"""Детектор исполняемого кода в комплекте модели (trust_remote_code).

Разбирает факты :class:`BundleScanner` (директория-комплект) и репортит ДВА
уровня РАЗНОЙ серьёзностью:

* **Уровень 1 (INFO)** — факт присутствия исполняемого Python в комплекте:
  - ``MLS-BUNDLE-001`` — на ``.py`` ССЫЛАЕТСЯ загрузочный конфиг
    (``config.json`` → ``auto_map`` / ``custom_pipeline`` / ``auto_modelcard``):
    код исполнится при загрузке через механизм Hugging Face trust_remote_code;
  - ``MLS-BUNDLE-003`` — ``.py`` лежит в комплекте БЕЗ ссылки из конфига:
    более слабый сигнал (код присутствует, но автозагрузка не подтверждена).

* **Уровень 2 (HIGH)** — ``MLS-BUNDLE-002``: внутри кода, НА КОТОРЫЙ ССЫЛАЕТСЯ
  конфиг, найден опасный паттерн из :data:`rules/bundle/code_patterns.yaml`
  (exec/eval/subprocess/socket/…), срабатывающий при загрузке.

ГРАНИЦА (жёсткий non-goal): репортим ФАКТ присутствия паттерна, НЕ доказываем
вредоносность и НЕ делаем data-flow. Вердикт «опасен ли код на самом деле» — за
человеком. Опасный паттерн в коде БЕЗ ссылки из конфига остаётся Уровнем 1
(MLS-BUNDLE-003), НЕ эскалируется в HIGH: это и отличает «код исполнится при
загрузке» от «просто лежит .py».

Срабатывает только для RawScanData от BundleScanner (scanner_name == 'bundle').
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import yaml  # type: ignore[import-untyped]

from poison_check._paths import RULES_DIR
from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Reference,
    Severity,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.bundle_scanner import METADATA_BUNDLE

logger = logging.getLogger(__name__)

# Namespace MLS-BUNDLE-NNN закреплён за комплектом модели.
_CODE_REF_CODE: str = "MLS-BUNDLE-001"       # Уровень 1: код по ссылке из конфига
_CODE_DANGEROUS: str = "MLS-BUNDLE-002"      # Уровень 2: опасный паттерн в коде по ссылке
_CODE_CODE_NOREF: str = "MLS-BUNDLE-003"     # Уровень 1 (слабый): .py без ссылки
_CODE_OBFUSCATION: str = "MLS-BUNDLE-004"    # Признаки обфускации/динамич. разрешения имён

_BUNDLE_SCANNER_NAME: str = "bundle"

_DEFAULT_PATTERNS_PATH = RULES_DIR / "bundle" / "code_patterns.yaml"

_COMPLIANCE_TAGS_RCE: tuple[str, ...] = (
    "owasp-ml:ml03",
    "owasp-ml:ml10",
    "fstec:ubi-067",
    "gost:56939-2024:5.3",
)


@DetectorRegistry.register
class BundleCodeDetector(BaseDetector):
    """Эмитит MLS-BUNDLE-* по фактам BundleScanner (код в комплекте модели)."""

    name: ClassVar[str] = "bundle_code"
    description: ClassVar[str] = (
        "Детектор исполняемого кода в комплекте модели (trust_remote_code)"
    )
    severity_range: ClassVar[tuple[Severity, Severity]] = (
        Severity.INFO,
        Severity.HIGH,
    )

    def __init__(self, patterns_path: Any = None) -> None:
        """Создаёт детектор, загружая список опасных паттернов Уровня 2 из YAML.

        :param patterns_path: Путь к YAML с паттернами (для тестов). По умолчанию
            ``rules/bundle/code_patterns.yaml``. При ошибке загрузки список
            пуст — Уровень 2 не эмитится, Уровень 1 (факт наличия кода) работает.
        """
        path = patterns_path if patterns_path is not None else _DEFAULT_PATTERNS_PATH
        self._patterns = _load_patterns(path)

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Возвращает Issues по фактам комплекта.

        :param raw_data: Результат BundleScanner.
        :param context: ML-контекст (не используется — вектор не зависит от фреймворка).
        :return: Список Issues; пустой если scanner_name != 'bundle' или фактов нет.
        """
        if raw_data.scanner_name != _BUNDLE_SCANNER_NAME:
            return []
        metadata = raw_data.metadata or {}
        facts = metadata.get(METADATA_BUNDLE)
        if not isinstance(facts, dict):
            return []

        root = str(facts.get("root", raw_data.file_path))
        py_files = facts.get("py_files", {})
        if not isinstance(py_files, dict):
            return []

        issues: list[Issue] = []
        for py_name, pf in sorted(py_files.items()):
            if not isinstance(pf, dict):
                continue
            referenced = bool(pf.get("referenced"))
            names = pf.get("names") or []
            obfs = pf.get("obfuscations") or []
            hits = self._match(names)
            location = f"{root}/{py_name}"

            if referenced:
                # Уровень 1 (INFO): код исполнится при загрузке (trust_remote_code).
                issues.append(_make_ref_issue(location, py_name))
                # Уровень 2: серьёзность по КЛАССУ вызова и ПОЗИЦИИ узла (факт +
                # позиция, без data-flow):
                #   - code-exec (захват выполнения) в загрузочной точке → HIGH;
                #   - side-effect (сеть/ФС/native) в загрузочной точке → LOW
                #     (законный сигнал «делает вызов при загрузке», виден в
                #     отчёте, но не гейтит НИ ОДНУ политику, включая strict);
                #   - любой класс в обычном методе → INFO (детект 3).
                load_exec = [h for h in hits
                             if h.get("pos") == "load" and h.get("class") != "side-effect"]
                load_side = [h for h in hits
                             if h.get("pos") == "load" and h.get("class") == "side-effect"]
                method_hits = [h for h in hits if h.get("pos") != "load"]
                if load_exec:
                    issues.append(
                        _make_dangerous_issue(location, py_name, load_exec, Severity.HIGH)
                    )
                if load_side:
                    issues.append(
                        _make_dangerous_issue(location, py_name, load_side, Severity.LOW)
                    )
                if method_hits:
                    issues.append(
                        _make_dangerous_issue(location, py_name, method_hits, Severity.INFO)
                    )
                # Детект 4 (INFO): признаки обфускации/динамического разрешения
                # имён в загрузочном коде. Содержимое НЕ раскрывается.
                if obfs:
                    issues.append(_make_obfuscation_issue(location, py_name, obfs))
            else:
                # Уровень 1 (слабый): .py без ссылки. Опасные паттерны / обфускация
                # тут — НЕ HIGH: автозагрузка не подтверждена (фиксируем в details).
                issues.append(_make_noref_issue(location, py_name, hits, obfs))

        return issues

    def _match(self, names: list[Any]) -> list[dict[str, Any]]:
        """Сопоставляет извлечённые имена с паттернами Уровня 2.

        :return: список hits ``{pattern_id, name, line, pos, description}`` — по
            одному на пару (pattern_id, pos): один и тот же паттерн в загрузочной
            и в обычной позиции даёт две записи (для детекта 3).
        """
        hits_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in names:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            line = entry.get("line")
            pos = entry.get("pos", "load")
            if not isinstance(name, str):
                continue
            for pat in self._patterns:
                key = (pat["id"], pos)
                if key in hits_by_key:
                    continue
                if _name_matches(name, pat["targets"]):
                    hits_by_key[key] = {
                        "pattern_id": pat["id"],
                        "name": name,
                        "line": line,
                        "pos": pos,
                        "description": pat["description"],
                        "class": pat.get("class", "code-exec"),
                    }
        return [hits_by_key[k] for k in sorted(hits_by_key)]


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def _name_matches(name: str, targets: list[str]) -> bool:
    """True, если извлечённое имя совпадает с target или начинается с ``target.``.

    Префиксный матч нужен для модулей: target ``subprocess`` ловит
    ``subprocess.run``; target ``os.system`` (полное имя) НЕ ловит ``os.path``.
    """
    return any(name == target or name.startswith(target + ".") for target in targets)


def _load_patterns(path: Any) -> list[dict[str, Any]]:
    """Загружает паттерны Уровня 2 из YAML. При ошибке — пустой список.

    :return: список ``{id, targets, description}``.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Не удалось загрузить паттерны комплекта %s: %s — Уровень 2 отключён",
            path,
            exc,
        )
        return []
    if not isinstance(raw, dict):
        return []
    patterns: list[dict[str, Any]] = []
    for item in raw.get("patterns", []) or []:
        if not isinstance(item, dict):
            continue
        pid = item.get("id")
        targets = item.get("targets")
        if not isinstance(pid, str) or not isinstance(targets, list):
            continue
        cls = item.get("class", "code-exec")
        patterns.append(
            {
                "id": pid,
                "targets": [t for t in targets if isinstance(t, str)],
                "description": str(item.get("description", pid)),
                "class": cls if cls in ("code-exec", "side-effect") else "code-exec",
            }
        )
    return patterns


def _make_ref_issue(location: str, py_name: str) -> Issue:
    """MLS-BUNDLE-001 (INFO) — код по ссылке из конфига (Уровень 1)."""
    return Issue(
        code=_CODE_REF_CODE,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        message=(
            f"Комплект модели содержит исполняемый код {py_name}, на который "
            "ссылается config.json (auto_map / custom_pipeline). При загрузке с "
            "trust_remote_code этот код будет исполнен."
        ),
        location=location,
        details={"py_file": py_name, "referenced": True, "level": 1},
        why=(
            "Механизм Hugging Face trust_remote_code импортирует и исполняет "
            "локальный .py из комплекта при from_pretrained(..., "
            "trust_remote_code=True). Сам факт автозагружаемого кода — то, о чём "
            "службе ИБ нужно знать; вердикт о безопасности кода — за человеком."
        ),
        remediation=(
            "Убедитесь, что источник модели доверенный. Загружайте комплекты с "
            "auto_map только с trust_remote_code=True для проверенных репозиториев; "
            "для недоверенных — используйте safetensors без кастомного кода."
        ),
        compliance_tags=["owasp-ml:ml03", "gost:56939-2024:5.3"],
    )


def _make_dangerous_issue(
    location: str, py_name: str, hits: list[dict[str, Any]], severity: Severity
) -> Issue:
    """MLS-BUNDLE-002 — паттерн Уровня 2 в коде по ссылке.

    Серьёзность задаётся вызывающим по КЛАССУ вызова и ПОЗИЦИИ узла (факт +
    позиция, без data-flow):
      - code-exec (захват выполнения) в загрузочной точке → HIGH;
      - side-effect (сеть/ФС/native) в загрузочной точке → LOW (фактическая
        формулировка «делает вызов при загрузке — проверьте», не «опасно»;
        виден в отчёте, но ниже порога любой политики — не гейтит);
      - любой класс в обычном методе → INFO (при загрузке не вызывается).
    Все hits в одном вызове — одного класса и позиции (analyze их так делит).
    """
    cls = hits[0].get("class", "code-exec") if hits else "code-exec"
    is_load = hits[0].get("pos") == "load" if hits else True
    first_line = next((h.get("line") for h in hits if h.get("line")), None)
    anchor = f"{location}:{first_line}" if first_line else location
    pattern_list = ", ".join(f"{h['pattern_id']} ({h['name']}:{h['line']})" for h in hits)
    where_load = "в загрузочной точке (module-level/dunder auto_map-класса/декоратор)"
    where_method = "в обычном методе — при загрузке НЕ вызывается (позиция понижает серьёзность)"

    if cls == "side-effect":
        if is_load:
            message = (
                f"Автозагружаемый код {py_name} выполняет сетевой/файловый/нативный "
                f"вызов {where_load} — сработает при загрузке модели: {pattern_list}. "
                "Это ФАКТ, а не подтверждённая угроза — проверьте назначение вызова."
            )
        else:
            message = (
                f"Код {py_name} содержит сетевой/файловый/нативный вызов {where_method}: "
                f"{pattern_list}."
            )
        why = (
            "Код, на который ссылается config.json, импортируется при загрузке "
            "модели (trust_remote_code). Сетевой/файловый/native-вызов (socket, "
            "urllib, requests, http.client, ctypes) при загрузке — это побочный "
            "эффект (загрузка ресурсов, обращение к сети/ФС), а НЕ захват "
            "выполнения. Законный сигнал, который стоит показать (модель ходит в "
            "сеть/ФС при загрузке), но сам по себе не «стоп»: severity LOW в "
            "загрузочной точке (виден, но не гейтит ни одну политику), INFO вне "
            "её. Репортится ФАКТ; назначение вызова определяет человек при ревью."
        )
        remediation = (
            "Проверьте, зачем автозагружаемый код обращается к сети/ФС при загрузке "
            "(скачивание весов/словаря — норма; обращение к внешнему хосту — повод "
            "для внимания). Для недоверенного источника не запускайте from_pretrained "
            "с trust_remote_code=True."
        )
    else:  # code-exec
        where = (
            f"{where_load} — исполнится при загрузке модели" if is_load else where_method
        )
        message = (
            f"В автозагружаемом коде {py_name} найдены примитивы выполнения кода "
            f"{where}: {pattern_list}."
        )
        why = (
            "Код, на который ссылается config.json, импортируется при загрузке "
            "модели (trust_remote_code). Перечисленные конструкции (exec/eval/"
            "compile, запуск процессов, десериализация) — захват выполнения, "
            "типичные примитивы RCE. Severity по ПОЗИЦИИ: в загрузочной точке "
            "исполнится при загрузке (HIGH); в обычном методе — только при явном "
            "вызове метода, что не анализируется (INFO). Репортится ФАКТ; вердикт — "
            "за человеком."
        )
        remediation = (
            "Проведите ручной код-ревью указанного .py до загрузки модели. Не "
            "запускайте from_pretrained с trust_remote_code=True из недоверенного "
            "источника. При возможности используйте модель без кастомного кода."
        )
    return Issue(
        code=_CODE_DANGEROUS,
        severity=severity,
        confidence=Confidence.MEDIUM,
        message=message,
        location=anchor,
        details={
            "py_file": py_name,
            "referenced": True,
            "level": 2 if severity >= Severity.HIGH else 1,
            "position": "load" if is_load else "method",
            "class": cls,
            "patterns": hits,
        },
        why=why,
        remediation=remediation,
        references=[
            Reference(type="cwe", id="CWE-94"),
            Reference(type="cwe", id="CWE-829"),
        ],
        compliance_tags=list(_COMPLIANCE_TAGS_RCE),
    )


def _make_obfuscation_issue(
    location: str, py_name: str, obfs: list[dict[str, Any]]
) -> Issue:
    """MLS-BUNDLE-004 (INFO) — признаки обфускации/динамич. разрешения имён.

    Детект 4 (НОВЫЙ КЛАСС): репортит сам ФАКТ присутствия обфускации в загрузочном
    коде. Содержимое НЕ раскрывается. Всегда INFO: опасность не доказана (одна ось
    из двух недоказана) — это сигнал к ручному ревью, не подтверждённая угроза.
    """
    first_line = next((o.get("line") for o in obfs if o.get("line")), None)
    anchor = f"{location}:{first_line}" if first_line else location
    kinds = ", ".join(f"{o['kind']} (строка {o['line']})" for o in obfs)
    return Issue(
        code=_CODE_OBFUSCATION,
        severity=Severity.INFO,
        confidence=Confidence.LOW,
        message=(
            f"В автозагружаемом коде {py_name} — признаки динамического/"
            f"обфусцированного разрешения имён: {kinds}. Содержимое не раскрывается."
        ),
        location=anchor,
        details={
            "py_file": py_name,
            "referenced": True,
            "level": 1,
            "obfuscation": obfs,
        },
        why=(
            "Имя вызываемого объекта собирается/разрешается динамически "
            "(getattr/__import__ с не-литеральным аргументом, base64/hex/codecs-"
            "декодирование, сборка из chr(), exec/eval от не-литерала) в коде, "
            "исполняемом при загрузке. Статический анализатор НЕ раскрывает, какой "
            "именно глобал получается — это потребовало бы трекинга значений "
            "(data-flow), вне границы. Факт обфускации сам по себе — повод для "
            "ручного ревью; доказанной опасности здесь нет, поэтому INFO."
        ),
        remediation=(
            "Проверьте вручную, какой объект разрешается этими конструкциями. "
            "Легитимному загрузочному коду обфускация имён не нужна."
        ),
        references=[Reference(type="cwe", id="CWE-94")],
        compliance_tags=["owasp-ml:ml03"],
    )


def _make_noref_issue(
    location: str,
    py_name: str,
    hits: list[dict[str, Any]],
    obfs: list[dict[str, Any]],
) -> Issue:
    """MLS-BUNDLE-003 (INFO, слабый) — .py без ссылки из конфига (Уровень 1).

    Опасные паттерны / обфускация, если есть, фиксируются в details, но severity
    остаётся INFO: автозагрузка не подтверждена (граница «исполнится» vs «лежит»).
    """
    parts: list[str] = []
    if hits:
        parts.append(
            "паттерны " + ", ".join(f"{h['pattern_id']} ({h['name']}:{h['line']})" for h in hits)
        )
    if obfs:
        parts.append("признаки обфускации " + ", ".join(f"{o['kind']}:{o['line']}" for o in obfs))
    hit_suffix = ""
    if parts:
        hit_suffix = (
            " В нём присутствуют " + "; ".join(parts)
            + ", но ссылки из config.json на этот файл нет — автозагрузка не подтверждена."
        )
    return Issue(
        code=_CODE_CODE_NOREF,
        severity=Severity.INFO,
        confidence=Confidence.LOW,
        message=(
            f"Комплект модели содержит .py ({py_name}) БЕЗ ссылки из config.json. "
            f"Более слабый сигнал, чем автозагружаемый код.{hit_suffix}"
        ),
        location=location,
        details={
            "py_file": py_name,
            "referenced": False,
            "level": 1,
            "patterns": hits,
            "obfuscation": obfs,
        },
        why=(
            "Python-файл лежит рядом с весами, но config.json на него не ссылается "
            "— он не будет автоматически исполнен механизмом trust_remote_code. "
            "Присутствие кода всё же стоит отметить: его может импортировать "
            "пользовательский загрузчик или другой модуль комплекта."
        ),
        remediation=(
            "Проверьте, зачем в комплекте модели лежит .py, если он не заявлен в "
            "config.json. Удалите лишний код или подтвердите его происхождение."
        ),
        compliance_tags=["owasp-ml:ml03"],
    )

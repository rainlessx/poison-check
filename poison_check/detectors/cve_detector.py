"""Детектор известных CVE-паттернов в ML-моделях.

В отличие от BlocklistDetector (точные совпадения module+name),
CVEDetector работает по комбинированным паттернам:
опкоды + глобалы + ML-контекст.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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

# Тот же safe-literal-факт getattr, что и в MLS-PKL-001 (детекторы одного слоя;
# прецедент — AllowlistDetector использует _HARDCODED_BLOCKLIST из blocklist).
from poison_check.detectors.blocklist_detector import _getattr_safe_literal
from poison_check.i18n.loader import I18n

logger = logging.getLogger(__name__)

_DEFAULT_RULES_PATH = RULES_DIR / "cve" / "ml_cves.yaml"

# Sentinel: если в правиле не указан context_frameworks — подходит любой фреймворк
_ANY_FRAMEWORK = frozenset[str]()


@dataclass
class _GlobalRef:
    """Один (module, name) из списка affected_globals."""

    module: str
    name: str


@dataclass
class _CveRef:
    """Ссылка на CVE / CWE / БДУ."""

    type: str
    id: str


@dataclass
class _CvePattern:
    """Полное CVE-правило с паттерном для детекции.

    Поля trigger_opcodes и affected_globals объединяются через AND:
    чтобы паттерн сработал, нужно обнаружить хотя бы один опкод из
    trigger_opcodes И хотя бы один глобал из affected_globals.

    context_frameworks — опциональный фильтр по ML-фреймворку.
    Пустое множество означает «любой фреймворк».
    """

    id: str
    title_ru: str
    description_ru: str
    trigger_opcodes: frozenset[str]       # REDUCE, GLOBAL, STACK_GLOBAL, ...
    affected_globals: list[_GlobalRef]    # хотя бы один должен совпасть
    severity: Severity
    confidence: Confidence
    references: list[_CveRef]
    remediation_ru: str
    title_en: str = ""
    description_en: str = ""
    remediation_en: str = ""
    context_frameworks: frozenset[str] = field(default_factory=frozenset)
    # Дополнительный контекст для Issue.details
    extra_details: dict[str, Any] = field(default_factory=dict)


@DetectorRegistry.register
class CVEDetector(BaseDetector):
    """Детектор известных CVE-паттернов в ML-моделях.

    Работает по комбинированным паттернам из YAML:
    trigger_opcodes AND affected_globals, опционально — контекст фреймворка.

    Отличие от BlocklistDetector:
    — BlocklistDetector: точное совпадение (module, name) → issue.
    — CVEDetector: паттерн = набор опкодов + набор глобалов + контекст.
      Один CVE-паттерн срабатывает, если в файле найдена комбинация,
      а не только один конкретный глобал.

    Применение:
    — CVE-2025-32434: REDUCE + torch._utils + PyTorch-контекст.
    — PATTERN-OS-SYSTEM: REDUCE/GLOBAL + os.system + любой контекст.
    """

    name: ClassVar[str] = "cve"
    description: ClassVar[str] = "Детектор известных CVE-паттернов в ML-моделях"
    severity_range: ClassVar[tuple[Severity, Severity]] = (Severity.HIGH, Severity.CRITICAL)

    def __init__(self, rules_path: Path | None = None) -> None:
        """Загружает CVE-паттерны из YAML.

        rules_path: путь к YAML-файлу с правилами. По умолчанию —
        rules/cve/ml_cves.yaml относительно корня проекта.
        """
        path = rules_path if rules_path is not None else _DEFAULT_RULES_PATH
        self._patterns: list[_CvePattern] = _load_patterns(path)

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Проверяет RawScanData на соответствие CVE-паттернам.

        Алгоритм для каждого паттерна:
        1. Проверяет наличие trigger_opcodes в raw_data.opcodes (OR по набору).
        2. Проверяет наличие хотя бы одного affected_global в raw_data.globals (OR).
        3. Если context_frameworks задан — проверяет совпадение с context.framework.
        4. При совпадении всех условий — создаёт Issue с severity=CRITICAL/HIGH,
           confidence=CERTAIN, кодом "MLS-CVE-XXXX" или "MLS-PATTERN-XXXX".

        Один паттерн → один Issue (не дублируется по количеству совпавших глобалов).
        """
        if raw_data.error is not None and raw_data.globals is None and raw_data.opcodes is None:
            # Файл не удалось разобрать — нечего анализировать
            return []

        present_opcodes = _extract_opcode_names(raw_data)
        present_globals = raw_data.globals or set()

        issues: list[Issue] = []
        seen_cve_ids: set[str] = set()  # не дублировать один CVE если несколько глобалов совпало

        for pattern in self._patterns:
            if pattern.id in seen_cve_ids:
                continue

            # 1. Проверка trigger_opcodes (хотя бы один)
            if pattern.trigger_opcodes and not (pattern.trigger_opcodes & present_opcodes):
                continue

            # 2. Проверка affected_globals (хотя бы один совпал)
            matched_globals = _find_matching_globals(pattern.affected_globals, present_globals)
            if not matched_globals:
                continue

            # Сужение getattr (калибровка правки 2, симметрично MLS-PKL-001):
            # если единственный совпавший глобал — builtins.getattr с БЕЗОПАСНЫМ
            # литеральным именем (getattr(m,"Cls")), это легит-реконструкция, не
            # indirect-RCE. При наличии __import__ или опасного/динамич. getattr
            # паттерн срабатывает как прежде (цепочка не ослаблена).
            if (
                matched_globals == [("builtins", "getattr")]
                and _getattr_safe_literal(raw_data)
            ):
                continue

            # 3. Проверка контекста фреймворка (если задан)
            if pattern.context_frameworks:
                framework = context.framework.lower()
                if framework not in pattern.context_frameworks:
                    continue

            # Все условия выполнены — создаём Issue
            issue = _make_cve_issue(raw_data, pattern, matched_globals, context)
            issues.append(issue)
            seen_cve_ids.add(pattern.id)

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _extract_opcode_names(raw_data: RawScanData) -> frozenset[str]:
    """Возвращает множество имён опкодов из raw_data.opcodes."""
    if not raw_data.opcodes:
        return frozenset()
    return frozenset(op.opcode for op in raw_data.opcodes)


def _find_matching_globals(
    affected: list[_GlobalRef],
    present: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Возвращает список (module, name) из affected, которые есть в present."""
    return [
        (ref.module, ref.name)
        for ref in affected
        if (ref.module, ref.name) in present
    ]


def _find_location(
    raw_data: RawScanData,
    matched_globals: list[tuple[str, str]],
) -> str:
    """Возвращает строку location с позицией первого совпавшего reduce_call."""
    if raw_data.reduce_calls and matched_globals:
        for rc in raw_data.reduce_calls:
            for module, name in matched_globals:
                if rc.module == module and rc.name == name:
                    return f"{raw_data.file_path}:offset {rc.position}"
    return str(raw_data.file_path)


def _make_cve_issue(
    raw_data: RawScanData,
    pattern: _CvePattern,
    matched_globals: list[tuple[str, str]],
    context: MLContext,
) -> Issue:
    """Создаёт Issue для найденного CVE-паттерна."""
    # Формат кода: MLS-CVE-2025-32434 или MLS-PATTERN-OS-SYSTEM
    code = f"MLS-{pattern.id}"

    matched_str = ", ".join(f"{m}.{n}" for m, n in matched_globals)
    details: dict[str, Any] = {
        "cve_id": pattern.id,
        "matched_globals": matched_str,
        "framework": context.framework,
        **pattern.extra_details,
    }

    # Локализация текста находки: en, если запрошена локаль en и для правила есть
    # перевод; иначе — русский (локаль по умолчанию). Обёртка CLI локализуется
    # отдельно через i18n, а тексты правил живут в самом YAML (title/*_en и *_ru).
    if I18n.get().locale == "en" and pattern.title_en:
        message = pattern.title_en
        why = pattern.description_en or pattern.description_ru
        remediation = pattern.remediation_en or pattern.remediation_ru
    else:
        message = pattern.title_ru
        why = pattern.description_ru
        remediation = pattern.remediation_ru

    return Issue(
        code=code,
        severity=pattern.severity,
        confidence=pattern.confidence,
        message=message,
        location=_find_location(raw_data, matched_globals),
        details=details,
        why=why,
        remediation=remediation,
        references=[Reference(type=r.type, id=r.id) for r in pattern.references],
        compliance_tags=[
            "owasp-ml:ml03",
            "owasp-ml:ml10",
            "fstec:ubi-067",
            "fstec:ubi-179",
            "gost:56939-2024:5.3",
            "gost:56939-2024:6.1",
            "gost:56939-2024:7.1",
        ],
    )


# ---------------------------------------------------------------------------
# Загрузка паттернов из YAML
# ---------------------------------------------------------------------------


def _parse_pattern(entry: dict[str, Any]) -> _CvePattern:
    """Разбирает одну запись из YAML в _CvePattern.

    Бросает KeyError, ValueError или TypeError если запись некорректна.
    """
    affected = [
        _GlobalRef(module=str(g["module"]), name=str(g["name"]))
        for g in entry["affected_globals"]
    ]
    refs = [
        _CveRef(type=str(r["type"]), id=str(r["id"]))
        for r in entry.get("references", [])
    ]

    raw_opcodes = entry.get("trigger_opcodes", [])
    if not isinstance(raw_opcodes, list):
        raise TypeError(f"trigger_opcodes должен быть списком, получен: {type(raw_opcodes)}")
    trigger_opcodes = frozenset(str(op) for op in raw_opcodes)

    raw_frameworks = entry.get("context_frameworks", [])
    if not isinstance(raw_frameworks, list):
        raise TypeError(f"context_frameworks должен быть списком, получен: {type(raw_frameworks)}")
    context_frameworks = frozenset(str(fw).lower() for fw in raw_frameworks)

    return _CvePattern(
        id=str(entry["id"]),
        title_ru=str(entry["title_ru"]),
        description_ru=str(entry["description_ru"]),
        trigger_opcodes=trigger_opcodes,
        affected_globals=affected,
        severity=Severity(entry["severity"]),
        confidence=Confidence(entry["confidence"]),
        references=refs,
        remediation_ru=str(entry["remediation_ru"]),
        title_en=str(entry.get("title", "")),
        description_en=str(entry.get("description_en", "")),
        remediation_en=str(entry.get("remediation_en", "")),
        context_frameworks=context_frameworks,
    )


def _load_patterns(path: Path) -> list[_CvePattern]:
    """Загружает CVE-паттерны из YAML-файла.

    При отсутствии файла или ошибках разбора возвращает пустой список —
    детектор продолжает работать (graceful degradation), но ничего не находит.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info(
            "CVE-паттерны не найдены: %s — CVEDetector не будет срабатывать",
            path,
        )
        return []
    except OSError as exc:
        logger.warning("Не удалось прочитать CVE YAML %s: %s", path, exc)
        return []
    except yaml.YAMLError as exc:
        logger.warning("Ошибка разбора CVE YAML %s: %s", path, exc)
        return []

    if not isinstance(raw, list):
        logger.warning(
            "Неожиданный формат CVE YAML %s: ожидается список, получен %s",
            path,
            type(raw).__name__,
        )
        return []

    patterns: list[_CvePattern] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            patterns.append(_parse_pattern(entry))
        except (KeyError, ValueError, TypeError) as exc:
            rule_id = entry.get("id", "?") if isinstance(entry, dict) else "?"
            logger.warning(
                "Пропускаем некорректную CVE-запись %s: %s", rule_id, exc
            )

    logger.debug("Загружено CVE-паттернов: %d из %s", len(patterns), path)
    return patterns

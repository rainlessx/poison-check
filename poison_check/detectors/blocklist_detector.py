"""Детектор по списку запрещённых глобалов (legacy, для CVE matching)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from poison_check.core.detector_base import BaseDetector
from poison_check.core.known_dangerous import (
    KNOWN_DANGEROUS_GLOBALS,
    is_py2_alias,
    normalize_global,
)
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Reference,
    Severity,
)
from poison_check.core.scanner_base import RawScanData


@dataclass
class _BlocklistRule:
    """Детальное правило для одной пары (module, name) в hardcoded blocklist."""

    severity: Severity
    description: str
    remediation: str


@DetectorRegistry.register
class BlocklistDetector(BaseDetector):
    """Детектор по списку запрещённых глобалов (legacy, для CVE matching).

    Проверяет globals против hardcoded blocklist
    (:data:`~poison_check.core.known_dangerous.KNOWN_DANGEROUS_GLOBALS`),
    а также сигналы уровня pickle-потока: homoglyph-атаки, PERSID,
    parse-stop и embedded payload.

    CVE-правила из ``rules/cve/ml_cves.yaml`` здесь НЕ читаются — этим
    целиком занимается :class:`~poison_check.detectors.cve_detector.CVEDetector`.
    Если один глобал попал и сюда, и в CVE-паттерн, лишний Issue снимается
    на этапе :func:`~poison_check.core.result.dedupe_issues`.
    """

    name = "blocklist"
    description = "Детектор по списку запрещённых глобалов (legacy, для CVE matching)"
    severity_range = (Severity.INFO, Severity.CRITICAL)

    # Полный hardcoded blocklist — синхронизирован с KNOWN_DANGEROUS_GLOBALS.
    # Источник истины для множества пар — core/known_dangerous.py.
    _HARDCODED_BLOCKLIST: ClassVar[frozenset[tuple[str, str]]] = KNOWN_DANGEROUS_GLOBALS

    # Детальные правила для конкретных пар — severity, описание, remediation.
    # Пары без явного правила получают дефолтный Issue (MLS001, CRITICAL).
    _BLOCKLIST_RULES: ClassVar[dict[tuple[str, str], _BlocklistRule]] = {
        # --- Выполнение системных команд (POSIX / Windows) ---
        ("os", "system"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "os.system позволяет выполнять произвольные команды ОС при "
                "десериализации pickle-файла (RCE, CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите у источника "
                "безопасный формат (safetensors или ONNX)."
            ),
        ),
        ("os", "popen"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "os.popen открывает подпроцесс и возвращает канал ввода-вывода — "
                "позволяет выполнять команды ОС при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите у источника "
                "безопасный формат (safetensors или ONNX)."
            ),
        ),
        ("posix", "system"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "posix.system — низкоуровневый Linux-эквивалент os.system. "
                "Позволяет выполнять произвольные команды ОС при десериализации."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите у источника "
                "безопасный формат (safetensors или ONNX)."
            ),
        ),
        ("posix", "popen"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "posix.popen — низкоуровневый Linux-эквивалент os.popen. "
                "Открывает подпроцесс через shell при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите у источника "
                "безопасный формат (safetensors или ONNX)."
            ),
        ),
        ("nt", "system"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "nt.system — низкоуровневый Windows-эквивалент os.system. "
                "Позволяет выполнять произвольные команды при десериализации."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите у источника "
                "безопасный формат (safetensors или ONNX)."
            ),
        ),
        # --- exec-семейство ---
        ("os", "execv"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "os.execv заменяет текущий процесс новой программой — "
                "полная замена процесса произвольным исполняемым файлом (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("os", "execve"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "os.execve заменяет текущий процесс новой программой с "
                "произвольным окружением — полная замена процесса (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("os", "execvp"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "os.execvp заменяет текущий процесс, выполняя поиск программы "
                "в PATH — позволяет запустить любой исполняемый файл (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- spawn-семейство ---
        ("os", "spawnl"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "os.spawnl запускает дочерний процесс с произвольными аргументами "
                "— позволяет выполнять команды ОС при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("os", "spawnve"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "os.spawnve запускает дочерний процесс с произвольным окружением "
                "— позволяет выполнять команды ОС при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- Псевдотерминал ---
        ("pty", "spawn"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "pty.spawn запускает процесс в псевдотерминале — позволяет "
                "выполнять интерактивные команды ОС при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- subprocess ---
        ("subprocess", "Popen"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "subprocess.Popen запускает дочерний процесс — полноценное "
                "выполнение произвольных команд ОС при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("subprocess", "call"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "subprocess.call запускает команду и ожидает её завершения — "
                "произвольное выполнение кода при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("subprocess", "run"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "subprocess.run запускает команду — произвольное выполнение "
                "кода при десериализации pickle-файла (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("subprocess", "check_output"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "subprocess.check_output выполняет команду и возвращает вывод — "
                "произвольное выполнение кода + утечка данных (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("subprocess", "check_call"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "subprocess.check_call выполняет команду и бросает исключение "
                "при ненулевом коде возврата — RCE при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- Py2-модуль commands и его Py3-наследники в subprocess ---
        ("commands", "getoutput"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "commands.getoutput (Python 2) выполняет строку через shell "
                "и возвращает её вывод — прямой аналог os.system с утечкой "
                "результата. Модуль удалён в Python 3, его появление в "
                "ML-файле означает либо legacy-сериализацию, либо "
                "преднамеренную попытку обхода сканеров (CWE-502, CWE-78)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("commands", "getstatusoutput"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "commands.getstatusoutput (Python 2) выполняет строку через "
                "shell и возвращает (код возврата, вывод) — RCE при "
                "десериализации. Модуль удалён в Python 3 (CWE-502, CWE-78)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("subprocess", "getoutput"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "subprocess.getoutput выполняет команду через shell и "
                "возвращает её вывод — произвольное выполнение кода плюс "
                "канал для утечки данных (CWE-502, CWE-78)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("subprocess", "getstatusoutput"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "subprocess.getstatusoutput выполняет команду через shell и "
                "возвращает (код возврата, вывод) — RCE при десериализации "
                "(CWE-502, CWE-78)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- Python builtins ---
        ("builtins", "eval"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "builtins.eval выполняет произвольный Python-код, переданный "
                "в виде строки — прямой RCE при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("builtins", "exec"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "builtins.exec выполняет произвольный блок Python-кода — "
                "прямой RCE при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("builtins", "compile"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "builtins.compile компилирует строку в code object — "
                "первый шаг многих RCE-цепочек при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("builtins", "breakpoint"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "builtins.breakpoint вызывает sys.breakpointhook. По умолчанию "
                "это pdb.set_trace (интерактивная сессия внутри процесса), а "
                "переменная окружения PYTHONBREAKPOINT позволяет подменить hook "
                "на любую импортируемую функцию — например, os.system (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- Интерактивный интерпретатор ---
        ("code", "interact"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "code.interact запускает интерактивный интерпретатор Python "
                "внутри процесса и может выполнить произвольный код из "
                "параметров banner/local — RCE при десериализации (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- Динамический импорт ---
        ("importlib", "import_module"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "importlib.import_module динамически импортирует произвольный "
                "модуль — может быть использован для загрузки вредоносного кода "
                "(CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("builtins", "__import__"): _BlocklistRule(
            severity=Severity.HIGH,
            description=(
                "builtins.__import__ динамически импортирует произвольный модуль. "
                "В pickle-цепочке __import__('os') + getattr(os, 'system') + call "
                "— это полноценный indirect RCE без прямого вызова os.system (CWE-502)."
            ),
            remediation=(
                "ML-модели не должны содержать __import__. "
                "Запросите безопасный формат (safetensors или ONNX)."
            ),
        ),
        ("builtins", "getattr"): _BlocklistRule(
            severity=Severity.HIGH,
            description=(
                "builtins.getattr извлекает произвольный атрибут объекта. "
                "В pickle-цепочке используется для получения опасных функций "
                "через рефлексию без явного GLOBAL (CWE-502)."
            ),
            remediation=(
                "ML-модели не должны содержать буiltins.getattr. "
                "Запросите безопасный формат (safetensors или ONNX)."
            ),
        ),
        # --- Вторичная десериализация ---
        ("marshal", "loads"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "marshal.loads десериализует байты в Python code object — "
                "позволяет встроить произвольный байт-код Python внутрь pickle "
                "(RCE-цепочка pickle→marshal, CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("pickle", "loads"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "pickle.loads внутри pickle — двойная десериализация. Позволяет "
                "обходить поверхностную проверку сканерами (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        ("dill", "loads"): _BlocklistRule(
            severity=Severity.CRITICAL,
            description=(
                "dill.loads внутри pickle — dill поддерживает сериализацию "
                "lambda и code objects, что обходит ряд защит (CWE-502)."
            ),
            remediation=(
                "Не загружайте этот файл. Запросите безопасный формат."
            ),
        ),
        # --- Сетевые подключения ---
        ("socket", "socket"): _BlocklistRule(
            severity=Severity.HIGH,
            description=(
                "socket.socket создаёт сетевой сокет — признак попытки "
                "установить C2-соединение или exfiltrate данные при десериализации."
            ),
            remediation=(
                "Не загружайте этот файл без тщательного анализа. "
                "Запросите безопасный формат."
            ),
        ),
    }

    def __init__(self, rules_path: Path | None = None) -> None:
        """Инициализирует детектор.

        ``rules_path`` — устаревший параметр, оставлен только для обратной
        совместимости конструктора. CVE-правила из YAML теперь полностью
        обрабатываются в :class:`CVEDetector` — двойной emit (см. историю)
        порождал дубликаты вида ``PATTERN-OS-SYSTEM`` + ``MLS-PATTERN-OS-SYSTEM``
        в консольном выводе.
        """
        # Параметр сохранён для сигнатуры, но не используется. Явно "пометим",
        # чтобы линтер не жаловался, при этом не тянем YAML при инициализации.
        _ = rules_path

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Ищет hardcoded blocklist-глобалы, PERSID, homoglyph-атаки.

        CVE-паттерны из ``rules/cve/ml_cves.yaml`` целиком обрабатываются
        в :class:`CVEDetector` — здесь не проверяем, чтобы избежать дубликатов.
        context не используется (blocklist не зависит от ML-фреймворка).
        """
        issues: list[Issue] = []

        # --- Homoglyph-атаки: не-ASCII байты в GLOBAL/INST opcode ---
        if raw_data.metadata:
            homoglyphs = raw_data.metadata.get("homoglyph_globals")
            if homoglyphs and isinstance(homoglyphs, list):
                for item in homoglyphs:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        mod_str, name_str = str(item[0]), str(item[1])
                        issues.append(_make_homoglyph_issue(raw_data, mod_str, name_str))
            # --- Parse-stop атака: прерванный и возобновлённый pickle-поток ---
            if raw_data.metadata.get("parse_stop_attack"):
                issues.append(_make_parse_stop_issue(raw_data))
            embedded = raw_data.metadata.get("embedded_strings")
            if embedded and isinstance(embedded, list):
                issues.append(_make_embedded_payload_issue(raw_data, embedded))
            # --- Аномально длинный поток: анализ усечён сканером (защита от OOM) ---
            if raw_data.metadata.get("opcode_limit_exceeded") == "true":
                limit = raw_data.metadata.get("opcode_limit")
                issues.append(_make_opcode_limit_issue(raw_data, limit))

        if raw_data.globals is None:
            return issues

        for module, name in raw_data.globals:
            if module == "__persid__":
                issues.append(_make_persid_issue(raw_data, name))
                continue
            # Нормализуем Py2/legacy имена перед lookup, но в issue пишем СЫРУЮ
            # пару — чтобы аналитик видел реальный опасный опкод (forensics).
            canonical_pair = normalize_global(module, name)
            if canonical_pair in self._HARDCODED_BLOCKLIST:
                # Сужение MLS-PKL-001 для getattr (калибровка правки 2):
                # getattr(obj, "<безопасное_имя>") — легитимная реконструкция
                # (ultralytics/YOLO .pt). НЕ флагаем, если 2-й аргумент — литерал,
                # разрешающийся в безопасное имя. getattr от не-литерала ИЛИ к
                # опасному имени (getattr(os,"system")) остаётся HIGH.
                if canonical_pair == ("builtins", "getattr") and _getattr_safe_literal(
                    raw_data
                ):
                    continue
                rule = self._BLOCKLIST_RULES.get(canonical_pair)
                issues.append(_make_blocklist_issue(raw_data, module, name, rule))

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции (не методы — не загромождают публичный API класса)
# ---------------------------------------------------------------------------

#: Имена атрибутов, получение которых через getattr(obj, "<name>") — сигнал
#: indirect-RCE-цепочки (getattr(os,"system"), getattr(builtins,"exec")). Набор
#: — из name-частей блок-списка плюс базовые примитивы исполнения/рефлексии.
#: getattr с БЕЗОПАСНЫМ литеральным именем (не из набора) — легитимная
#: реконструкция объекта (ultralytics/YOLO), не эскалируется (калибровка правки 2).
_DANGEROUS_GETATTR_ATTRS: frozenset[str] = frozenset(
    {name for _mod, name in KNOWN_DANGEROUS_GLOBALS}
    | {
        "system", "popen", "exec", "eval", "compile", "__import__", "import_module",
        "getattr", "setattr", "__getattribute__", "spawn", "run", "call",
        "check_output", "check_call", "getoutput", "getstatusoutput",
        "Popen", "loads", "load", "fromhex", "b64decode",
    }
)


def _getattr_safe_literal(raw_data: RawScanData) -> bool:
    """True, если ВСЕ вызовы getattr в файле — с безопасным ЛИТЕРАЛЬНЫМ именем.

    Опирается на факты сканера (metadata): ``getattr_literal_attrs`` (литеральные
    2-е аргументы) и ``getattr_dynamic`` (был не-литеральный аргумент / getattr
    не вызывался). Возвращает True только когда есть хотя бы один литерал, нет
    динамики и ни одно имя не входит в :data:`_DANGEROUS_GETATTR_ATTRS`. Это
    факт+имя, без трекинга значений — не ослабляет детект обхода через getattr.
    """
    meta = raw_data.metadata or {}
    if meta.get("getattr_dynamic") == "true":
        return False
    literals = meta.get("getattr_literal_attrs")
    if not literals:
        return False  # нет данных о литералах → консервативно оставляем HIGH
    attrs = [a for a in str(literals).split(",") if a]
    if not attrs:
        return False
    return not any(a in _DANGEROUS_GETATTR_ATTRS for a in attrs)


def _make_homoglyph_issue(raw_data: RawScanData, module_repr: str, name_repr: str) -> Issue:
    """Создаёт Issue для homoglyph-атаки: не-ASCII символы в GLOBAL/INST opcode (MLS-PKL-002)."""
    return Issue(
        code="MLS-PKL-002",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=(
            f"Не-ASCII символы в GLOBAL/INST opcode: модуль {module_repr!r}"
            " — возможная homoglyph-атака"
        ),
        location=str(raw_data.file_path),
        details={"module_repr": module_repr, "name_repr": name_repr, "opcode": "GLOBAL/INST"},
        why=(
            "Не-ASCII символы в имени модуля/функции используются для обхода сканеров "
            "путём визуальной подмены: например, Cyrillic 'о' (U+043E) вместо ASCII 'o'. "
            "Большинство детекторов сравнивают строки как ASCII — подменённый вызов "
            "остаётся незамеченным (homoglyph/confusable attack)."
        ),
        remediation=(
            "Не загружайте этот файл. Легитимные ML-модели не содержат не-ASCII символов "
            "в именах модулей Python. Признак преднамеренной попытки обхода защиты."
        ),
        references=[Reference(type="cwe", id="CWE-502"), Reference(type="cwe", id="CWE-1007")],
        compliance_tags=[
            "owasp-ml:ml03",
            "owasp-ml:ml10",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )


def _make_persid_issue(raw_data: RawScanData, pid: str) -> Issue:
    """Создаёт Issue для PERSID/BINPERSID opcode (MLS-PKL-003)."""
    return Issue(
        code="MLS-PKL-003",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=f"Обнаружен PERSID opcode с id={pid!r} — не должен встречаться в ML-файлах",
        location=str(raw_data.file_path),
        details={"persid_value": pid, "opcode": "PERSID/BINPERSID"},
        why=(
            "PERSID/BINPERSID — pickle-опкоды для персистентных ссылок. "
            "В ML-моделях они не используются никогда. Их наличие означает "
            "либо кастомный Unpickler с persistent_load (потенциальный RCE), "
            "либо преднамеренную попытку обхода сканеров через нестандартный опкод."
        ),
        remediation=(
            "ML-файл содержит нестандартный pickle-опкод. Не загружайте его. "
            "Запросите у источника стандартный формат (safetensors или ONNX)."
        ),
        references=[Reference(type="cwe", id="CWE-502")],
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )


def _make_parse_stop_issue(raw_data: RawScanData) -> Issue:
    """Создаёт Issue для parse-stop атаки (MLS-PKL-004)."""
    return Issue(
        code="MLS-PKL-004",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=(
            "Parse-stop атака: pickle-поток прерван нестандартными байтами "
            "и продолжен в следующем фрейме — обход статического анализа"
        ),
        location=str(raw_data.file_path),
        details={"technique": "parse-stop / NumpyArrayWrapper inline data"},
        why=(
            "Атакующий вставляет raw-байты (например, данные numpy-массива) "
            "в pickle-поток: статические анализаторы на pickletools.genops "
            "останавливаются и не видят вредоносный payload в следующем фрейме. "
            "Python продолжает десериализацию через кастомный persistent_load."
        ),
        remediation=(
            "Файл содержит несколько pickle-фреймов с прерыванием потока — "
            "стандартные ML-модели так не сохраняются. "
            "Не загружайте. Запросите у источника формат safetensors или ONNX."
        ),
        references=[
            Reference(type="cwe", id="CWE-502"),
            Reference(type="cwe", id="CWE-693"),
        ],
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )


def _make_embedded_payload_issue(raw_data: RawScanData, patterns: list[str]) -> Issue:
    """Создаёт Issue для embedded source-code payload (MLS-PKL-005)."""
    found_str = ", ".join(f"{p!r}" for p in patterns[:5])
    return Issue(
        code="MLS-PKL-005",
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        message=(
            f"Обнаружен embedded source-code payload после ошибки парсинга pickle: {found_str}"
        ),
        location=str(raw_data.file_path),
        details={"patterns": patterns},
        why=(
            "После прерывания pickle-потока в файле обнаружен Python source code "
            "в виде ASCII-текста. Реальный пример: adithyanm-defender/security-research-pickle-rce "
            "на HuggingFace — reverse shell payload скрыт после NumpyArrayWrapper raw bytes."
        ),
        remediation=(
            "Файл содержит встроенный исполняемый код. "
            "Не загружайте. Запросите безопасный формат."
        ),
        references=[Reference(type="cwe", id="CWE-502")],
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
        ],
    )


def _make_opcode_limit_issue(raw_data: RawScanData, limit: object | None) -> Issue:
    """Создаёт Issue для аномально длинного pickle-потока (MLS-PKL-006).

    Эмитируется, когда сканер выставил ``metadata["opcode_limit_exceeded"]``:
    число накопленных опкодов/строк достигло предела, и анализ был усечён во
    избежание OOM. Опасные глобалы, собранные ДО лимита, остаются в globals_set
    и по-прежнему детектируются отдельными Issue — это предупреждение лишь о том,
    что «хвост» потока не проанализирован.
    """
    limit_str = str(limit) if limit is not None else "лимит"
    return Issue(
        code="MLS-PKL-006",
        severity=Severity.MEDIUM,
        confidence=Confidence.MEDIUM,
        message=(
            "Аномально длинный pickle-поток: число накопленных элементов "
            f"достигло лимита {limit_str}, анализ усечён — возможная DoS-заготовка"
        ),
        location=str(raw_data.file_path),
        details={"opcode_limit": limit_str, "technique": "opcode flooding / DoS"},
        why=(
            "Pickle-поток содержит аномально большое число опкодов. Сканер усёк "
            "анализ на лимите, чтобы не исчерпать память (защита от OOM). "
            "Легитимные ML-модели не содержат миллионов опкодов в одном pickle. "
            "Такой поток может быть DoS-заготовкой (opcode flooding), заготовкой "
            "zip/pickle-бомбы, либо попыткой спрятать вредоносный payload за "
            "пределами усечения."
        ),
        remediation=(
            "Проверьте происхождение файла. Аномально длинный pickle-поток — "
            "признак либо повреждения, либо преднамеренной DoS-атаки. "
            "Запросите у источника безопасный формат (safetensors или ONNX)."
        ),
        references=[
            Reference(type="cwe", id="CWE-400"),
            Reference(type="cwe", id="CWE-502"),
        ],
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )


def _find_location(raw_data: RawScanData, module: str, name: str) -> str:
    """Возвращает строку location с позицией reduce_call если найдена."""
    if raw_data.reduce_calls:
        for rc in raw_data.reduce_calls:
            if rc.module == module and rc.name == name:
                return f"{raw_data.file_path}:offset {rc.position}"
    return str(raw_data.file_path)


def _make_blocklist_issue(
    raw_data: RawScanData,
    module: str,
    name: str,
    rule: _BlocklistRule | None = None,
) -> Issue:
    """Создаёт Issue для совпадения с hardcoded blocklist (MLS001).

    Если передан _BlocklistRule — использует его severity, description и
    remediation. Иначе применяет дефолтные CRITICAL-значения.
    """
    severity = rule.severity if rule is not None else Severity.CRITICAL
    why = (
        rule.description
        if rule is not None
        else (
            "Данный глобал позволяет выполнять произвольный код "
            "при десериализации pickle-файла (CWE-502)."
        )
    )
    remediation = (
        rule.remediation
        if rule is not None
        else (
            "Не загружайте этот файл. Запросите у источника "
            "безопасный формат (safetensors или ONNX)."
        )
    )
    # Py2/legacy алиас — форензик-подсказка в сообщении и в why.
    if is_py2_alias(module):
        canonical_module, canonical_name = normalize_global(module, name)
        alias_hint = (
            f" (Python 2 alias для {canonical_module}.{canonical_name})"
        )
        message = f"Обнаружен запрещённый глобал {module}.{name}{alias_hint}"
        why = (
            f"{why} Использование Py2-имени '{module}' вместо '{canonical_module}' "
            f"— типичная тактика обхода сканеров: детектор, проверяющий только "
            f"Python 3 форму, промолчит."
        )
    else:
        message = f"Обнаружен запрещённый глобал {module}.{name}"

    return Issue(
        code="MLS-PKL-001",
        severity=severity,
        confidence=Confidence.CERTAIN,
        message=message,
        location=_find_location(raw_data, module, name),
        details={"module": module, "name": name, "opcode": "REDUCE/GLOBAL"},
        why=why,
        remediation=remediation,
        references=[Reference(type="cwe", id="CWE-502")],
        compliance_tags=[
            "owasp-ml:ml03",
            "owasp-ml:ml10",
            "fstec:ubi-067",
            "fstec:ubi-068",
            "gost:56939-2024:5.3",
            "gost:56939-2024:6.1",
        ],
    )

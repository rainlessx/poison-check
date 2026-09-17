"""Типы данных для результатов сканирования ML-файлов."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(Enum):
    """Уровень опасности найденной проблемы.

    Порядок: CRITICAL > HIGH > MEDIUM > LOW > INFO.
    Поддерживает операторы сравнения для сортировки и фильтрации.
    """

    CRITICAL = "critical"  # Прямое исполнение кода в загрузочной позиции — макс. потенциал RCE
    HIGH = "high"          # Опасная конструкция, высокий потенциал эксплуатации
    MEDIUM = "medium"      # Подозрительная конструкция, требует ручной проверки
    LOW = "low"            # Низкий потенциал риска, информационно
    INFO = "info"          # Для полноты картины

    @property
    def level(self) -> int:
        """Числовой уровень опасности (выше = опаснее)."""
        return _SEVERITY_LEVELS[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.level < other.level

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.level <= other.level

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.level > other.level

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.level >= other.level


# Определяется после класса, чтобы можно было использовать Severity-члены как ключи
_SEVERITY_LEVELS: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Confidence(Enum):
    """Уверенность детектора в найденной проблеме."""

    CERTAIN = "certain"   # Известный паттерн CVE — ложных срабатываний нет
    HIGH = "high"         # Высокая уверенность
    MEDIUM = "medium"     # Требует проверки человеком
    LOW = "low"           # Эвристика, возможен false positive


@dataclass
class Reference:
    """Ссылка на внешний источник: CVE, CWE, БДУ ФСТЭК и т.д."""

    type: str  # "cve", "cwe", "capec", "bdu"
    id: str    # "CVE-2025-32434", "УБИ.067"


@dataclass
class OpcodeInfo:
    """Один опкод из pickle-потока."""

    position: int         # Смещение в байтах
    opcode: str           # Имя опкода: "REDUCE", "STACK_GLOBAL", ...
    arg: Any | None = None  # Аргумент опкода (если есть)


@dataclass
class StringInfo:
    """Строка, извлечённая из файла при сканировании."""

    value: str
    position: int
    encoding: str = "utf-8"


@dataclass
class ReduceCall:
    """Вызов REDUCE-опкода: module.name(*args)."""

    module: str
    name: str
    position: int


@dataclass
class TensorInfo:
    """Метаданные тензора (для safetensors и подобных форматов)."""

    name: str
    dtype: str
    shape: list[int] = field(default_factory=list)


@dataclass
class EmbeddedSignature:
    """Сигнатура встроенного исполняемого файла (PE/ELF/Mach-O)."""

    signature_type: str  # "PE", "ELF", "Mach-O"
    offset: int          # Смещение в байтах в родительском файле
    size: int            # Размер встроенного файла


@dataclass
class MLContext:
    """Определённый ML-фреймворк и уверенность детекции."""

    framework: str               # "pytorch", "sklearn", "tensorflow", "unknown"
    confidence: float            # 0.0 – 1.0
    detected_patterns: list[str] = field(default_factory=list)


@dataclass
class Issue:
    """Одна найденная проблема безопасности."""

    code: str               # MLS001, MLS002, ... (по аналогии с ESLint / Ruff)
    severity: Severity
    confidence: Confidence
    message: str            # На текущем языке локали
    location: str           # "model.pt:data.pkl (offset 1234)"

    details: dict[str, Any] = field(default_factory=dict)  # Технические данные

    # Человекочитаемые поля
    why: str | None = None             # Почему это опасно
    remediation: str | None = None     # Что делать
    decompiled_code: str | None = None # Декомпилированный Python для pickle

    # Ссылки и теги
    references: list[Reference] = field(default_factory=list)
    compliance_tags: list[str] = field(default_factory=list)  # "owasp-ml:ml03"


@dataclass
class FileResult:
    """Результат сканирования одного файла."""

    file_path: Path
    scanner_name: str
    issues: list[Issue] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None  # Заполняется при ошибке парсинга


@dataclass
class Summary:
    """Агрегированные счётчики по severity для всего прогона."""

    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    info: int = 0
    worst_severity: Severity | None = None
    blocked_by_policy: bool = False


@dataclass(frozen=True)
class PolicyThresholds:
    """Пороги политики, с которыми был выполнен прогон.

    Едет вместе с :class:`ScanResult`, чтобы слой Output мог применить
    ``severity_threshold`` (порог внимания) без обратной зависимости на
    загрузчик политик: результат сканирования сам несёт настройки, которые его
    породили. Благодаря этому Python API (``JsonFormatter().format(result)``)
    ведёт себя так же, как CLI, без дополнительных аргументов.

    :ivar severity_threshold: Порог ВНИМАНИЯ. Находки ниже него остаются в
        выводе, но помечаются как «ниже порога» (см.
        :mod:`poison_check.output.severity_threshold`). ``None`` — порог не
        задан, ни одна находка не помечается.
    :ivar fail_on_severity: Порог ДЕЙСТВИЯ — с какого уровня CLI возвращает
        exit code 1. На видимость находок не влияет.
    """

    severity_threshold: Severity | None = None
    fail_on_severity: Severity | None = None


@dataclass(frozen=True)
class ReportMetadata:
    """Метаданные шапки отчёта: для кого и кем выполнено сканирование.

    Едет вместе с :class:`ScanResult` по той же причине, что и
    :class:`PolicyThresholds`: результат сканирования сам несёт данные, которые
    нужны для его оформления, и слой Output не обязан получать их отдельными
    аргументами. Благодаря этому Python API
    (``JsonFormatter().format(result)``) выводит шапку так же, как CLI.

    Пустое значение хранится как ``None``, а не как пустая строка: форматтеры
    ОПУСКАЮТ незаданное поле, а не пишут его пустым (см.
    :mod:`poison_check.output.report_metadata`).

    :ivar client: Заказчик аудита — титульный лист отчёта по ГОСТ Р 56939-2024.
    :ivar auditor: ФИО аудитора — лист подписи отчёта.
    """

    client: str | None = None
    auditor: str | None = None


@dataclass
class ComplianceReport:
    """Маппинг находок на регуляторные требования."""

    owasp_ml_top_10: list[str] = field(default_factory=list)  # ["ML03", "ML10"]
    fstec_ubi: list[str] = field(default_factory=list)         # ["УБИ.067"]
    gost_references: list[str] = field(default_factory=list)   # ["ГОСТ Р 56939-2024 п.5.3"]

    #: Дисклеймер о статусе верификации compliance-маппинга (аудит #30).
    #: Маппинг УБИ ФСТЭК / OWASP ML / ГОСТ был построен на основании публичных
    #: описаний угроз. ИБ-консультант не привлекался — для banking/government
    #: использования рекомендуется верификация. Поле не None по умолчанию,
    #: чтобы исключить молчаливое отображение compliance как «verified».
    disclaimer: str | None = None


@dataclass
class ScanResult:
    """Итоговый результат всего прогона сканирования."""

    tool_version: str
    timestamp: datetime
    duration_ms: float

    scanned_paths: list[Path] = field(default_factory=list)
    policy: str = "default"
    results_per_file: dict[Path, FileResult] = field(default_factory=dict)
    summary: Summary = field(default_factory=Summary)
    compliance_report: ComplianceReport | None = None

    #: Пороги политики, применённые к этому прогону. ``None`` означает «порогов
    #: нет» — форматтеры выводят всё без разделения на основные и подпороговые
    #: находки (поведение до появления severity_threshold).
    policy_thresholds: PolicyThresholds | None = None

    #: Метаданные шапки отчёта (заказчик / аудитор). ``None`` — не заданы, и
    #: тогда ни один формат не выводит соответствующие поля (а не выводит их
    #: пустыми).
    report_metadata: ReportMetadata | None = None

    # --- вычисляемые свойства ---

    @property
    def worst_severity(self) -> Severity | None:
        """Максимальный severity среди всех issues во всех файлах."""
        severities = [
            issue.severity
            for fr in self.results_per_file.values()
            for issue in fr.issues
        ]
        return max(severities) if severities else None

    @property
    def has_critical(self) -> bool:
        """True если есть хотя бы одна проблема с уровнем CRITICAL."""
        return self.has_issues_above(Severity.CRITICAL)

    @property
    def has_errors(self) -> bool:
        """True если хотя бы один файл не удалось разобрать."""
        return any(fr.error is not None for fr in self.results_per_file.values())

    # --- публичные методы ---

    def has_issues_above(self, severity: Severity) -> bool:
        """Возвращает True если есть хотя бы одна issue с severity >= указанного."""
        return any(
            issue.severity >= severity
            for fr in self.results_per_file.values()
            for issue in fr.issues
        )

    def issues_by_severity(self) -> dict[Severity, list[Issue]]:
        """Группирует все issues из всех файлов по уровню severity."""
        groups: dict[Severity, list[Issue]] = {s: [] for s in Severity}
        for file_result in self.results_per_file.values():
            for issue in file_result.issues:
                groups[issue.severity].append(issue)
        return groups


_CONFIDENCE_RANK: dict[str, int] = {
    "certain": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


_SPECIFICITY_RANK: dict[str, int] = {
    # Более высокое число = более специфичный код.
    # Правило: чем длиннее и уже cкоуп кода, тем специфичнее.
    "MLS-PATTERN-": 40,   # CVE-паттерны с opcode+global+context — уже
    "MLS-CVE-": 30,       # конкретный CVE-идентификатор
    "MLS-ALW-": 20,       # allowlist-исключение с указанием framework
    "MLS-PKL-": 10,       # protocol-specific pickle-проблема
}
_GENERIC_HARDCODED_CODE = "MLS-PKL-001"  # генерик hardcoded blocklist


def _code_rank(code: str) -> int:
    """Возвращает числовой ранг специфичности кода.

    ``MLS-PKL-001`` — самый generic (hardcoded blocklist без CVE-контекста).
    Все остальные ``MLS-*`` коды информативнее в равной мере, но
    ``MLS-PATTERN-*`` и ``MLS-CVE-*`` при этом связаны с YAML-правилами
    (детальный remediation, CWE-refs) — им ранг ещё выше.
    """
    if code == _GENERIC_HARDCODED_CODE:
        return 0
    for prefix, rank in _SPECIFICITY_RANK.items():
        if code.startswith(prefix):
            return rank
    return 5  # неизвестный формат — выше generic'а, ниже специфики


#: Позиционные «якоря» в location. Детекторы, указывающие на КОНКРЕТНУЮ точку
#: файла, всегда добавляют к пути один из этих маркеров:
#:   ``{path}:offset 42``            — BlocklistDetector / CVEDetector (reduce_call)
#:   ``{path} (offset 42)``          — NetworkDetector / SecretsDetector / ExecutableDetector
#:   ``{path}#module.name``          — AllowlistDetector (конкретный global)
#:   ``{path} (metadata key: 'k')``  — SecretsDetector по метаданным
#: Только для таких location две записи от разных детекторов гарантированно
#: описывают ОДНУ угрозу и могут быть слиты (аудит: схлопывание независимых
#: находок на голом file_path скрывало вторую находку).
_LOCATION_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ":offset 42" / " (offset 42)" — позиция в байтах
    re.compile(r"offset\s+\d+"),
    # "#module.name" в конце строки — конкретный global (dotted-идентификаторы)
    re.compile(r"#[^\W\d]\w*(?:\.\w+)+$"),
    # " (metadata key: 'k')" — конкретный ключ метаданных
    re.compile(r"metadata key:"),
)


def _has_position_anchor(location: str) -> bool:
    """Есть ли в ``location`` позиционный якорь (offset / global / metadata key).

    Голый путь к файлу якоря не имеет: разные детекторы пишут туда
    независимые угрозы (homoglyph, PERSID, zip-бомба, ...), и сливать их
    в одну запись нельзя — это скрытие находок.
    """
    return any(pattern.search(location) for pattern in _LOCATION_ANCHOR_PATTERNS)


def dedupe_issues(issues: list[Issue]) -> list[Issue]:
    """Убирает дубликаты Issue в два прохода.

    **Проход 1 — точные повторы по (code, location):**
    Один детектор может выдать один и тот же Issue дважды, если совпало
    два правила из YAML на одну сущность.

    **Проход 2 — семантические дубликаты по location с позиционным якорем:**
    Разные детекторы часто «видят» одну и ту же угрозу под разным углом:
    ``BlocklistDetector`` эмитит generic ``MLS-PKL-001`` для ``os.system``,
    ``CVEDetector`` — специфичный ``MLS-PATTERN-OS-SYSTEM`` для той же
    reduce_call. Один location, две записи — шум.

    Слияние применяется ТОЛЬКО когда ``location`` несёт позиционный якорь
    (``offset N``, ``#module.name``, ``metadata key:`` — см.
    :func:`_has_position_anchor`). Совпадения одной угрозы от разных
    детекторов всегда указывают на одну и ту же точку файла и такой якорь
    имеют. Если же ``location`` — голый путь к файлу (``MLS-PKL-002``
    homoglyph, ``MLS-PKL-003`` PERSID, ``MLS-PKL-004`` parse-stop,
    ``MLS-PKL-005`` embedded payload, ``MLS-CMP-*``, GGUF-метаданные, а
    также blocklist/CVE без совпавшего reduce_call), то разные ``code`` —
    это РАЗНЫЕ угрозы. Такая группа не мержится: остаётся по одному
    представителю на каждый уникальный ``code`` (дубли уже сняты проходом 1).
    Слияние здесь означало бы скрытие находок.

    Стратегия слияния (не «выбор победителя»):
      - Ключ — ``location`` (только с позиционным якорем).
      - ``severity`` = максимальный уровень в группе (не даунгрейдим).
      - ``code``/``message``/``why``/``remediation``/``details`` — от
        записи с максимальным ``_code_rank`` (наиболее специфичной).
      - ``references``/``compliance_tags`` — объединение по всем в группе
        (не теряем CVE-refs даже если победил не CVE-код).
      - ``confidence`` — максимальная в группе.

    Так HIGH generic + CRITICAL specific → CRITICAL specific (правильно);
    CRITICAL generic + HIGH specific → CRITICAL с текстом specific
    (severity сохранён, форензик тоже).

    Сохраняет порядок первого появления в списке — важно для CI/CD
    воспроизводимости.
    """
    # Проход 1: точные дубликаты (code, location).
    seen: set[tuple[str, str]] = set()
    stage1: list[Issue] = []
    for issue in issues:
        key = (issue.code, issue.location)
        if key in seen:
            continue
        seen.add(key)
        stage1.append(issue)

    # Проход 2: группируем по location, мержим только якорные группы.
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, issue in enumerate(stage1):
        if issue.location not in groups:
            groups[issue.location] = []
            order.append(issue.location)
        groups[issue.location].append(idx)

    merged: list[Issue] = []
    for loc in order:
        group_idxs = groups[loc]
        if len(group_idxs) == 1:
            merged.append(stage1[group_idxs[0]])
            continue

        if not _has_position_anchor(loc):
            # Голый путь без якоря — независимые угрозы. Проход 1 уже оставил
            # по одной записи на каждый уникальный code, сохраняем все.
            merged.extend(stage1[i] for i in group_idxs)
            continue

        # Мерж: severity=max, шаблон текста от макс-rank записи,
        # refs/tags/confidence — агрегация по всем.
        max_severity = max(stage1[i].severity for i in group_idxs)
        template_idx = max(group_idxs, key=lambda i: _code_rank(stage1[i].code))
        template = stage1[template_idx]

        merged_refs: list[Reference] = []
        seen_refs: set[tuple[str, str]] = set()
        merged_tags: list[str] = []
        seen_tags: set[str] = set()
        max_confidence = template.confidence
        for i in group_idxs:
            candidate = stage1[i]
            if _CONFIDENCE_RANK.get(candidate.confidence.value, 0) > _CONFIDENCE_RANK.get(
                max_confidence.value, 0
            ):
                max_confidence = candidate.confidence
            for ref in candidate.references:
                key_ref = (ref.type, ref.id)
                if key_ref not in seen_refs:
                    seen_refs.add(key_ref)
                    merged_refs.append(ref)
            for tag in candidate.compliance_tags:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    merged_tags.append(tag)

        merged_issue = Issue(
            code=template.code,
            severity=max_severity,
            confidence=max_confidence,
            message=template.message,
            location=template.location,
            details=dict(template.details),
            why=template.why,
            remediation=template.remediation,
            references=merged_refs,
            compliance_tags=merged_tags,
        )
        merged.append(merged_issue)

    return merged

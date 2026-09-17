"""Точка входа CLI для poison-check — ML Supply Chain Security Scanner."""

from __future__ import annotations

import importlib.util
import io
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# Импортируем пакеты scanners и detectors, чтобы декораторы
# @ScannerRegistry.register / @DetectorRegistry.register выполнились.
import poison_check.detectors  # noqa: F401 — side-effect import
import poison_check.scanners  # noqa: F401 — side-effect import
from poison_check._paths import POLICIES_DIR, RULES_DIR
from poison_check.analysis.bundle import analyze_bundle, bundle_companion_files
from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry, ScannerRegistry
from poison_check.core.result import (
    ComplianceReport,
    FileResult,
    Issue,
    MLContext,
    ReportMetadata,
    ScanResult,
    Severity,
    Summary,
    dedupe_issues,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.parse_error_detector import annotate_suppressed_parse_error
from poison_check.i18n.loader import I18n
from poison_check.output.console import ConsoleFormatter
from poison_check.output.json_format import JsonFormatter
from poison_check.output.pdf_report import PdfReportFormatter
from poison_check.output.report_metadata import build_report_metadata
from poison_check.output.sarif import SarifFormatter
from poison_check.output.sbom import SbomFormatter
from poison_check.policies import (
    BUILTIN_POLICY_NAMES,
    PolicyLoader,
    instantiate_detectors,
    policy_fail_on_severity,
    policy_max_file_size_bytes,
    policy_severity_threshold,
    policy_thresholds,
    unregistered_extra_rule_keys,
    unregistered_policy_keys,
)
from poison_check.scanners.format_facts import attach_format_facts

app = typer.Typer(
    name="poison-check",
    help="ML Supply Chain Security Scanner — статический анализ безопасности ML-моделей.",
    add_completion=False,
)

console = Console()
_verbose_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

_POLICIES_DIR = POLICIES_DIR
_VERSION = "0.1.0"


class OutputFormat(str, Enum):
    """Формат отчёта для опции ``--format``.

    Перечисление, а не свободная строка: typer сам печатает список допустимых
    значений в ``scan --help`` и отвергает опечатку ДО начала сканирования
    внятной ошибкой click, а не трейсбеком и не сообщением «неизвестный формат»
    после того, как каталог уже прочитан целиком.

    Наследование от ``str`` сохраняет сравнение со строками — значения
    совпадают с именами форматтеров слоя Output.
    """

    CONSOLE = "console"
    JSON = "json"
    SARIF = "sarif"
    SBOM = "sbom"
    HTML = "html"
    PDF = "pdf"


#: Форматы, чей отчёт — текст. Для них ``--output`` пишется через
#: ``write_text``, для PDF — через ``write_bytes`` (бинарный формат).
_TEXT_FORMATS: frozenset[OutputFormat] = frozenset(
    {
        OutputFormat.CONSOLE,
        OutputFormat.JSON,
        OutputFormat.SARIF,
        OutputFormat.SBOM,
        OutputFormat.HTML,
    }
)


def _load_policy(name: str) -> dict[str, object]:
    """Загружает политику по имени или пути к YAML-файлу через PolicyLoader.

    Поддерживает:
    - встроенные политики: "default", "banking", "government", "strict"
    - путь к пользовательскому YAML: "/path/to/custom.yaml"

    При ошибке выводит предупреждение и возвращает политику "default".
    """
    try:
        loaded = PolicyLoader.load(name)
    except ValueError as exc:
        console.print(f"[yellow]⚠  {exc}[/yellow]")
        console.print("[yellow]⚠  Используется встроенная политика default[/yellow]")
        try:
            return PolicyLoader.load("default")
        except ValueError:
            return {}
    _warn_unregistered_keys(loaded, name)
    return loaded


def _warn_unregistered_keys(policy: dict[str, object], name: str) -> None:
    """Предупреждает о ключах политики, которых нет в реестре ключей.

    Реестр (``policies.POLICY_KEYS`` / ``EXTRA_RULE_KEYS``) знает все ключи,
    которые инструмент действительно читает. Ключ вне реестра ничего не делает —
    чаще всего это опечатка (``severity_treshold``) или настройка из чужой
    версии. Молча игнорировать её нельзя: пользователь будет уверен, что
    правило работает. Предупреждение идёт в stderr, чтобы не ломать
    ``--format json/sarif`` на stdout.
    """
    unknown = unregistered_policy_keys(policy) + [
        f"extra_rules.{key}" for key in unregistered_extra_rule_keys(policy)
    ]
    if unknown:
        _verbose_console.print(
            f"[yellow]⚠  Политика '{name}': ключи не распознаны и не применяются: "
            f"{', '.join(unknown)}[/yellow]"
        )


def _detect_ml_context(raw_data: RawScanData) -> MLContext:
    """Определяет ML-фреймворк через MLContextAnalyzer (неделя 10).

    Анализирует globals из RawScanData и возвращает MLContext
    с определённым фреймворком (pytorch/sklearn/tensorflow/numpy/unknown).
    """
    from poison_check.analysis.ml_context import MLContextAnalyzer

    return MLContextAnalyzer().analyze(raw_data)


#: id(policy) → (сама политика, её детекторы). Ссылка на dict политики
#: хранится намеренно: пока она жива, CPython не переиспользует её id
#: под другой объект — иначе кеш вернул бы детекторы чужой политики
#: (например, с включённой эскалацией URL от banking).
_DETECTOR_CACHE: dict[int, tuple[dict[str, object], list[BaseDetector]]] = {}


def _get_cached_detectors(policy: dict[str, object]) -> list[BaseDetector]:
    """Возвращает кешированные экземпляры детекторов для политики.

    Кеш ключуется по id(policy) — разные политики получают разные кеши,
    повторные вызовы для одной политики возвращают тот же список инстансов.
    Это устраняет O(N) парсинг YAML при сканировании N файлов:
    AllowlistDetector читает 727 строк YAML, SecretsDetector компилирует
    regex'ы — обе операции ненужны при каждом файле (аудит #10).

    Параметры конструкторов берутся из ``extra_rules`` политики через
    ``policies.instantiate_detectors`` — тот же путь, что и у Python API
    (``Scanner._get_detectors``), поэтому CLI и API ведут себя одинаково.
    """
    key = id(policy)
    cached = _DETECTOR_CACHE.get(key)
    if cached is None:
        detectors = instantiate_detectors(
            DetectorRegistry.enabled_for_policy(policy), policy
        )
        _DETECTOR_CACHE[key] = (policy, detectors)
        return detectors
    return cached[1]


def _scan_single_file(
    path: Path,
    policy: dict[str, object],
    max_file_size: int | None = None,
) -> tuple[FileResult, MLContext, RawScanData]:
    """Сканирует один файл и возвращает FileResult + MLContext + RawScanData.

    Алгоритм сканирования:
    1. FormatDetector → формат
    2. ScannerRegistry.find_scanner()
    3. scanner.scan() → RawScanData
    4. MLContext
    5. Все зарегистрированные детекторы → Issues
    6. Собираем FileResult

    Args:
        path:          Путь к файлу для сканирования.
        policy:        Загруженная политика сканирования.
        max_file_size: Максимальный размер файла в байтах. ``None`` —
            каждый сканер использует свой ``DEFAULT_MAX_FILE_SIZE``
            (GGUFScanner: 100 ГБ, остальные: 10 ГБ — аудит #24).
            Передавайте число для глобального override через ``--max-file-size``.
    """
    t_start = time.monotonic()

    scanner_class = ScannerRegistry.find_scanner(path)
    if scanner_class is None:
        # Неподдерживаемый формат. Файл всё равно проходит через детекторы:
        # расхождение «расширение ↔ содержимое» видно именно здесь (pickle-файл
        # с именем .safetensors сканер по расширению возьмёт, а safetensors с
        # именем .pkl — нет). Шума это не добавляет: MLS-PARSE-001 по такому
        # файлу подавлен флагом META_FORMAT_UNSUPPORTED, а остальные детекторы
        # без opcodes/строк молчат — поведение прежнее, кроме MLS-FMT-001.
        i18n = I18n.get()
        err = i18n.t("errors.unsupported_format", path=str(path))
        scanner_name = "unknown"
        raw_data = RawScanData(
            file_path=path,
            file_hash={},
            file_size=0,
            scanner_name=scanner_name,
            error=err,
        )
        attach_format_facts(raw_data, path, unsupported=True)
        context = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])
    else:
        scanner = scanner_class(max_file_size=max_file_size)
        scanner_name = scanner.name
        raw_data = scanner.scan(path)
        # Факты о формате (расхождение с расширением, класс безопасности) —
        # единая точка на оба пути, до запуска детекторов.
        attach_format_facts(raw_data, path)
        context = _detect_ml_context(raw_data)

    # Запускаем все разрешённые политикой детекторы. Кешируем экземпляры —
    # AllowlistDetector/SecretsDetector/CVEDetector парсят YAML при инициализации
    # (аудит #10).
    all_issues = []
    for detector in _get_cached_detectors(policy):
        try:
            issues = detector.analyze(raw_data, context)
            all_issues.extend(issues)
        except Exception as exc:  # noqa: BLE001
            # Детектор упал — логируем и продолжаем
            console.print(
                f"[dim]Детектор {type(detector).name} завершился с ошибкой: {exc}[/dim]"
            )

    duration_ms = (time.monotonic() - t_start) * 1000

    # Дедупликация (аудит #11): убираем точные повторы (code, location).
    # До фикса CLI выдавал дубликаты, в отличие от Python API через Scanner-фасад.
    all_issues = dedupe_issues(all_issues)

    # Краевой случай «payload + оборванный хвост»: факт обрыва разбора едет в
    # details находок (иначе теряется в SARIF). См. annotate_suppressed_parse_error.
    annotate_suppressed_parse_error(all_issues, raw_data)

    file_result = FileResult(
        file_path=path,
        scanner_name=scanner_name,
        issues=all_issues,
        duration_ms=duration_ms,
        error=raw_data.error,  # None если сканирование прошло без ошибок
    )
    return file_result, context, raw_data


def _collect_paths(path: Path, recursive: bool) -> list[Path]:
    """Собирает список файлов для сканирования.

    Если path — файл, возвращает [path] (если это обычный файл, не symlink/FIFO).
    Если path — директория, обходит рекурсивно (если recursive=True)
    или только верхний уровень.

    Безопасность (регрессия аудита #5): symbolic links, FIFO, сокеты и
    устройства отбрасываются. Symlink на ``/dev/zero`` возвращает st_size == 0
    в обход ``_check_file_size``, и любой потоковый сканер начнёт читать
    бесконечный поток — это DoS-вектор сам по себе.

    Поведение симметрично ``Scanner._collect_files``: одинаковая фильтрация
    через CLI и через Python API.
    """
    if path.is_symlink():
        # Корневой путь — symlink. Не следуем за ним: пользователь должен
        # явно указать целевой файл, иначе непонятно, что он сканирует.
        return []

    if path.is_file() and _is_regular_file(path):
        return [path]

    if path.is_dir():
        if recursive:
            return [
                p
                for p in sorted(path.rglob("*"))
                if p.is_file()
                and not p.is_symlink()
                and _is_regular_file(p)
                and not any(part.startswith(".") for part in p.parts)
            ]
        return sorted(
            p
            for p in path.iterdir()
            if p.is_file() and not p.is_symlink() and _is_regular_file(p)
        )

    return []


def _is_regular_file(path: Path) -> bool:
    """True только для обычных файлов (не FIFO/socket/character device).

    ``Path.is_file()`` возвращает True для FIFO и сокетов на некоторых
    платформах — этого недостаточно для безопасного сканирования.
    """
    import stat as _stat

    try:
        st = path.stat()
    except OSError:
        return False
    return _stat.S_ISREG(st.st_mode)


def _compute_exit_code(
    result: ScanResult,
    policy: dict[str, object],
) -> int:
    """Возвращает exit code: 0=чисто, 1=найдены issues, 2=ошибка сканирования.

    Порог severity для exit code 1 берётся из политики — ключ ``fail_on_severity``
    (новый формат) или ``exit_on_severity`` (устаревший, для совместимости).
    По умолчанию — critical.

    ``severity_threshold`` здесь НЕ участвует намеренно: это порог ВНИМАНИЯ
    (подача отчёта), а гейт CI определяется отдельным порогом ДЕЙСТВИЯ. Поэтому
    подпороговая находка, которая по ``fail_on_severity`` обязана валить сборку,
    её валит — помечена ≠ проигнорирована.
    """
    if result.has_errors:
        return 2

    threshold = policy_fail_on_severity(policy)

    for file_result in result.results_per_file.values():
        for issue in file_result.issues:
            if issue.severity >= threshold:
                return 1

    return 0


def _build_compliance_report(
    result: ScanResult,
    policy: dict[str, object],
) -> ComplianceReport | None:
    """Собирает ComplianceReport если политика требует compliance-проверок.

    Если ключ «compliance» в политике пустой или отсутствует — возвращает None.
    Иначе запускает нужные маперы (fstec, owasp_ml, gost_56939_2024) и
    собирает уникальные идентификаторы УБИ / OWASP ML / ГОСТ для итогового отчёта.

    Параметры:
        result: Частично собранный ScanResult (без compliance_report).
        policy: Загруженная политика сканирования.

    Возвращает:
        ComplianceReport или None если compliance не требуется.
    """
    compliance_list = policy.get("compliance", [])
    if not isinstance(compliance_list, list) or not compliance_list:
        return None

    owasp_ids: list[str] = []
    fstec_ids: list[str] = []
    gost_refs: list[str] = []

    if "fstec" in compliance_list:
        from poison_check.compliance.fstec_mapping import FstecMapper  # noqa: PLC0415

        fstec_mapper = FstecMapper()
        seen_fstec: set[str] = set()
        for file_result in result.results_per_file.values():
            for issue in file_result.issues:
                for ubi_id in fstec_mapper.map_issue(issue):
                    if ubi_id not in seen_fstec:
                        seen_fstec.add(ubi_id)
                        fstec_ids.append(ubi_id)

    if "owasp_ml" in compliance_list:
        from poison_check.compliance.owasp_ml_top10 import OwaspMapper  # noqa: PLC0415

        owasp_mapper = OwaspMapper()
        seen_owasp: set[str] = set()
        for file_result in result.results_per_file.values():
            for issue in file_result.issues:
                for cat_id in owasp_mapper.map_issue(issue):
                    if cat_id not in seen_owasp:
                        seen_owasp.add(cat_id)
                        owasp_ids.append(cat_id)

    if "gost_56939_2024" in compliance_list:
        from poison_check.compliance.gost_mapping import GostMapper  # noqa: PLC0415

        gost_mapper = GostMapper()
        seen_gost: set[str] = set()
        for file_result in result.results_per_file.values():
            for issue in file_result.issues:
                for section in gost_mapper.map_issue(issue):
                    ref = gost_mapper.format_reference(section)
                    if ref not in seen_gost:
                        seen_gost.add(ref)
                        gost_refs.append(ref)

    from poison_check.compliance.fstec_mapping import COMPLIANCE_DISCLAIMER  # noqa: PLC0415

    return ComplianceReport(
        owasp_ml_top_10=sorted(owasp_ids),
        fstec_ubi=sorted(fstec_ids),
        gost_references=sorted(gost_refs),
        disclaimer=COMPLIANCE_DISCLAIMER,
    )


def _build_scan_result(
    scanned_paths: list[Path],
    results_per_file: dict[Path, FileResult],
    policy_name: str,
    duration_ms: float,
    policy: dict[str, object] | None = None,
    report_metadata: ReportMetadata | None = None,
) -> ScanResult:
    """Собирает итоговый ScanResult из результатов по файлам.

    :param report_metadata: Реквизиты шапки отчёта (``--client``/``--auditor``).
        Едут внутри результата, поэтому доступны всем форматтерам одинаково.
    """
    # Пересчитываем summary
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for fr in results_per_file.values():
        for issue in fr.issues:
            key = issue.severity.value
            if key in counts:
                counts[key] += 1

    summary = Summary(
        critical=counts["critical"],
        high=counts["high"],
        medium=counts["medium"],
        low=counts["low"],
        info=counts["info"],
    )

    # Временно собираем ScanResult без compliance_report для передачи в маперы
    partial_result = ScanResult(
        tool_version=_VERSION,
        timestamp=datetime.now(timezone.utc),
        duration_ms=duration_ms,
        scanned_paths=scanned_paths,
        policy=policy_name,
        results_per_file=results_per_file,
        summary=summary,
        compliance_report=None,
        # Пороги едут вместе с результатом: форматтеры применяют
        # severity_threshold, не зная про загрузчик политик.
        policy_thresholds=policy_thresholds(policy or {}),
        report_metadata=report_metadata,
    )

    compliance_report = _build_compliance_report(partial_result, policy or {})
    partial_result.compliance_report = compliance_report
    return partial_result


# ---------------------------------------------------------------------------
# Verbose-вывод (--verbose / -v)
# ---------------------------------------------------------------------------


def _print_verbose_globals(
    raw_data: RawScanData,
    context: MLContext,
    file_result: FileResult,
) -> None:
    """Выводит breakdown глобалов для отладки false positive (--verbose).

    Каждый глобал помечается по наиболее критичному Issue, которое он вызвал:
    🚨 red   → HIGH/CRITICAL (blocklist, CVE-паттерны)
    ⚠ yellow → MEDIUM (allowlist miss)
    ℹ dim    → INFO/LOW (информационные)
    ✓ green  → нет issues

    Вывод идёт в stderr, чтобы не загрязнять --format json/sarif.
    """
    # Строим карту (module, name) → наиболее критичный Issue
    flagged: dict[tuple[str, str], Issue] = {}

    for issue in file_result.issues:
        m = issue.details.get("module", "")
        n = issue.details.get("name", "")
        pairs: list[tuple[str, str]] = []
        if m and n:
            pairs.append((str(m), str(n)))
        # CVEDetector пишет matched_globals как "mod.name, mod2.name2"
        mg_str = issue.details.get("matched_globals", "")
        if isinstance(mg_str, str) and mg_str:
            for mg in mg_str.split(", "):
                parts = mg.rsplit(".", 1)
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        for pair in pairs:
            existing = flagged.get(pair)
            if existing is None or issue.severity > existing.severity:
                flagged[pair] = issue

    globals_set = raw_data.globals or set()

    _verbose_console.print(
        f"  [bold dim]verbose[/bold dim]  "
        f"scanner=[cyan]{raw_data.scanner_name}[/cyan]  "
        f"framework=[cyan]{context.framework}[/cyan]  "
        f"confidence=[cyan]{context.confidence:.2f}[/cyan]"
    )

    if not globals_set:
        is_pickle = raw_data.scanner_name in ("pickle", "pytorch")
        reason = "нет GLOBAL opcodes в pickle-потоке" if is_pickle else "не pickle-формат"
        _verbose_console.print(f"  [dim]globals: пусто ({reason})[/dim]")
        return

    _verbose_console.print(f"  [dim]globals ({len(globals_set)}):[/dim]")
    for module, name in sorted(globals_set):
        pair = (module, name)
        hit: Issue | None = flagged.get(pair)
        label = f"{module}.{name}"
        if hit is None:
            _verbose_console.print(f"    [green]✓[/green] {label}")
        elif hit.severity >= Severity.HIGH:
            _verbose_console.print(
                f"    [red]✗[/red] {label}  [dim red][{hit.code}][/dim red]"
            )
        elif hit.severity >= Severity.MEDIUM:
            _verbose_console.print(
                f"    [yellow]⚠[/yellow] {label}  [dim yellow][{hit.code}][/dim yellow]"
            )
        else:
            _verbose_console.print(
                f"    [dim]ℹ[/dim] {label}  [dim][{hit.code}][/dim]"
            )


# ---------------------------------------------------------------------------
# Команды CLI
# ---------------------------------------------------------------------------


@app.command()
def scan(
    path: Annotated[Path, typer.Argument(help="Файл или директория для сканирования")],
    policy: Annotated[
        str,
        typer.Option(
            help=(
                "Политика сканирования: имя встроенной "
                f"({', '.join(sorted(BUILTIN_POLICY_NAMES))}) "
                "или путь к своему YAML-файлу (.yaml/.yml)."
            ),
        ),
    ] = "default",
    format: Annotated[  # noqa: A002 — shadowing builtin, но так требует задание
        OutputFormat,
        typer.Option(
            help=(
                "Формат отчёта. console — цветной вывод в терминал, "
                "json — отчёт по схеме v1.0, sarif — SARIF 2.1.0 для CI/CD, "
                "sbom — CycloneDX 1.4, html — HTML-отчёт по ГОСТ-шаблону, "
                "pdf — тот же отчёт в PDF (требует weasyprint)."
            ),
        ),
    ] = OutputFormat.CONSOLE,
    output: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Путь для сохранения отчёта. Работает для ВСЕХ форматов, "
                "включая console (пишется текст без ANSI-разметки). "
                "Без флага текстовые форматы идут в stdout, а PDF — в файл "
                "рядом со сканируемым путём."
            ),
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            "-r/-R",
            help=(
                "Обходить вложенные каталоги. По умолчанию ВЫКЛЮЧЕНО: без флага "
                "сканируется только верхний уровень указанного каталога. "
                "С флагом обход рекурсивный, но скрытые пути (.git, .venv, "
                "и любые компоненты с точки) пропускаются."
            ),
        ),
    ] = False,
    locale: Annotated[
        str, typer.Option(help="Язык вывода: ru, en")
    ] = "ru",
    client: Annotated[
        str | None,
        typer.Option(
            help=(
                "Заказчик аудита для шапки отчёта. Попадает во все форматы "
                "(console, json, sarif, sbom, pdf). Без флага поле опускается."
            ),
        ),
    ] = None,
    auditor: Annotated[
        str | None,
        typer.Option(
            help=(
                "ФИО аудитора для шапки отчёта. Попадает во все форматы "
                "(console, json, sarif, sbom, pdf). Без флага поле опускается."
            ),
        ),
    ] = None,
    max_file_size: Annotated[
        int,
        typer.Option(
            help=(
                "Максимальный размер файла в ГБ (override для всех форматов). "
                "0 = взять лимит из политики (extra_rules.max_file_size_gb), "
                "а если его нет — дефолт каждого сканера: pickle/joblib/numpy/"
                "safetensors → 10 ГБ, GGUF → 100 ГБ (для LLM 50–90 ГБ)."
            ),
        ),
    ] = 0,
    no_emoji: Annotated[
        bool,
        typer.Option(
            "--no-emoji",
            help=(
                "Использовать ASCII-иконки [!]/[*]/[i] вместо эмодзи. "
                "Авто-детект для не-TTY, не-UTF-8 локали и Windows cmd."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help=(
                "Подробный вывод для отладки: показывает глобалы каждого файла "
                "с разметкой allowlist/blocklist статуса. "
                "Полезно для диагностики false positive. "
                "Вывод идёт в stderr, не загрязняет --format json/sarif."
            ),
        ),
    ] = False,
    bundle_check: Annotated[
        bool,
        typer.Option(
            "--bundle-check/--no-bundle-check",
            help=(
                "Проверка КОМПЛЕКТА модели (код рядом с весами: trust_remote_code, "
                "весь префикс MLS-BUNDLE-*). По умолчанию ВКЛЮЧЕНА. "
                "С --no-bundle-check проверка не выполняется вовсе: находки "
                "MLS-BUNDLE-* не идут ни в отчёт, ни в exit-код гейта. Флаг имеет "
                "приоритет над enabled_detectors политики и только ВЫКЛЮЧАЕТ "
                "(включить проверку, отключённую политикой, он не может)."
            ),
        ),
    ] = True,
    audit_log: Annotated[
        Path | None,
        typer.Option(
            "--audit-log",
            help=(
                "Путь к JSON Lines audit-логу. Каждое сканирование добавляет одну строку "
                "(timestamp, user, host, путь, политика, exit code, счётчики severity). "
                "Также читается переменная окружения POISON_CHECK_AUDIT_LOG."
            ),
        ),
    ] = None,
) -> None:
    """Сканировать ML-модель на наличие угроз безопасности.

    Лимит размера файла определяется по приоритету:

    1. Флаг ``--max-file-size`` (в ГБ), если он больше нуля — имеет приоритет
       над политикой: явное указание пользователя всегда сильнее конфигурации.
    2. Ключ ``extra_rules.max_file_size_gb`` активной политики (banking → 50 ГБ,
       government → 100 ГБ, strict → 10 ГБ).
    3. ``DEFAULT_MAX_FILE_SIZE`` конкретного сканера (аудит #24):
       GGUFScanner = 100 ГБ, остальные = 10 ГБ.

    Обход каталога НЕ рекурсивен по умолчанию: без ``--recursive`` проверяются
    только файлы верхнего уровня. Рекурсия — явное решение пользователя, потому
    что каталог модели часто лежит рядом с кешем HuggingFace, ``.git`` и
    виртуальным окружением, и молчаливый обход вглубь превращает быструю
    проверку в многочасовую. При ``--recursive`` скрытые пути (любой компонент,
    начинающийся с точки) пропускаются по той же причине.
    """
    # Устанавливаем локаль
    I18n.set_locale(locale)

    # Шапка отчёта (--client / --auditor). Собирается один раз и едет в
    # ScanResult — оттуда её берут ВСЕ форматтеры через единую точку
    # poison_check.output.report_metadata. Пустые значения → None (поле
    # опускается, а не выводится пустым).
    report_metadata = build_report_metadata(client=client, auditor=auditor)

    # Шаг 1: Проверяем, что путь существует
    if not path.exists():
        i18n = I18n.get()
        err_msg = i18n.t("errors.file_not_found", path=str(path))
        console.print(f"[bold red]Ошибка:[/bold red] {err_msg}")
        raise typer.Exit(code=2)

    # Шаг 2: Загружаем политику
    loaded_policy = _load_policy(policy)

    # Шаг 2.1: Лимит размера файла. CLI-флаг сильнее политики; при
    # --max-file-size 0 берём extra_rules.max_file_size_gb, а если и его нет —
    # None, то есть DEFAULT_MAX_FILE_SIZE каждого сканера.
    max_file_size_bytes: int | None
    if max_file_size > 0:
        max_file_size_bytes = max_file_size * 1_000_000_000
    else:
        max_file_size_bytes = policy_max_file_size_bytes(loaded_policy)
        if max_file_size_bytes is not None and format is OutputFormat.CONSOLE:
            console.print(
                f"[dim]Лимит размера файла из политики "
                f"'{loaded_policy.get('name', policy)}': "
                f"{max_file_size_bytes / 1_000_000_000:g} ГБ[/dim]"
            )

    # Шаг 3: Собираем список файлов
    files_to_scan = _collect_paths(path, recursive)

    # Directory-level проверка комплекта модели (код рядом с весами). Вычисляем
    # ДО early-return «нет файлов»: комплект может состоять только из config.json
    # и modeling.py без отдельно сканируемых весов. companion-файлы (config.json
    # / *.py) исключаются из пофайлового цикла — их разбирает bundle-анализ, иначе
    # каждый из них дал бы «Неподдерживаемый формат» → ошибку → exit 2.
    bundle_result: FileResult | None = None
    if path.is_dir():
        companions = bundle_companion_files(files_to_scan)
        files_to_scan = [f for f in files_to_scan if f not in companions]
        # --no-bundle-check ПОЛНОСТЬЮ отключает проверку комплекта: analyze_bundle
        # не вызывается, MLS-BUNDLE-* не появляются ни в отчёте, ни в гейте. Флаг
        # сильнее enabled_detectors политики (но только выключает).
        if bundle_check:
            bundle_result = analyze_bundle(path, _get_cached_detectors(loaded_policy))

    if not files_to_scan and bundle_result is None:
        console.print(f"[yellow]Не найдено файлов для сканирования в: {path}[/yellow]")
        raise typer.Exit(code=0)

    # Шаг 4: Сканируем каждый файл.
    # При `--format console --output FILE` отчёт рендерится в буфер, а не в
    # терминал: раньше --output для console молча игнорировался и файл не
    # создавался вовсе. Буферная консоль без цвета и без ANSI — отчёт в файле
    # должен читаться, а не содержать escape-последовательности.
    console_buffer: io.StringIO | None = None
    report_console = console
    if format is OutputFormat.CONSOLE and output is not None:
        console_buffer = io.StringIO()
        report_console = Console(
            file=console_buffer, force_terminal=False, no_color=True, width=100
        )

    formatter = ConsoleFormatter(
        console=report_console,
        no_emoji=no_emoji if no_emoji else None,
        severity_threshold=policy_severity_threshold(loaded_policy),
    )
    if format is OutputFormat.CONSOLE:
        formatter.format_scan_start(path)
        formatter.format_report_header(report_metadata)

    t_global_start = time.monotonic()
    results_per_file: dict[Path, FileResult] = {}
    ml_contexts: dict[Path, MLContext] = {}

    for file_path in files_to_scan:
        if format is OutputFormat.CONSOLE:
            report_console.print(
                I18n.get().t("cli.scanning_file", path=str(file_path)), style="dim"
            )
        file_result, ctx, raw_data = _scan_single_file(
            file_path, loaded_policy, max_file_size=max_file_size_bytes
        )
        results_per_file[file_path] = file_result
        ml_contexts[file_path] = ctx
        if verbose:
            _print_verbose_globals(raw_data, ctx, file_result)

    # Bundle-анализ (уровень директории) — отдельная запись, привязанная к корню.
    scanned_paths = list(files_to_scan)
    if bundle_result is not None:
        if format is OutputFormat.CONSOLE:
            report_console.print(
                I18n.get().t("cli.scanning_file", path=str(path)), style="dim"
            )
        results_per_file[path] = bundle_result
        ml_contexts[path] = MLContext(
            framework="unknown", confidence=0.0, detected_patterns=[]
        )
        scanned_paths.append(path)

    total_duration_ms = (time.monotonic() - t_global_start) * 1000

    # Шаг 5: Собираем ScanResult
    result = _build_scan_result(
        scanned_paths=scanned_paths,
        results_per_file=results_per_file,
        policy_name=policy,
        duration_ms=total_duration_ms,
        policy=loaded_policy,
        report_metadata=report_metadata,
    )

    # Шаг 6: Форматируем и выводим
    if format is OutputFormat.CONSOLE:
        for file_result in results_per_file.values():
            formatter.format_file_result(file_result)
        formatter.format_summary(result)
        # Текст отчёта берём из буфера, если он есть (режим --output).
        report_str: str | None = (
            console_buffer.getvalue() if console_buffer is not None else None
        )

    elif format is OutputFormat.JSON:
        json_fmt = JsonFormatter()
        report_str = json_fmt.format(result, ml_contexts=ml_contexts)
        if output is None:
            # Выводим без Rich, чтобы не сломать JSON управляющими символами
            typer.echo(report_str)

    elif format is OutputFormat.SARIF:
        sarif_fmt = SarifFormatter()
        report_str = sarif_fmt.format(result)
        if output is None:
            typer.echo(report_str)

    elif format is OutputFormat.SBOM:
        sbom_fmt = SbomFormatter()
        report_str = sbom_fmt.format(result)
        if output is None:
            typer.echo(report_str)

    elif format is OutputFormat.HTML:
        # Тот же шаблон, что и у PDF, но без WeasyPrint: нужен только jinja2.
        try:
            html_fmt = PdfReportFormatter()
        except ImportError as exc:
            console.print(f"[red]Ошибка: {exc}[/red]")
            raise typer.Exit(code=2) from exc
        report_str = html_fmt.format_html(result)
        if output is None:
            typer.echo(report_str)

    else:
        # OutputFormat.PDF — единственное оставшееся значение: остальные click
        # уже отверг при разборе аргументов, поэтому ветки «неизвестный формат»
        # здесь больше нет.
        pdf_fmt = PdfReportFormatter()
        try:
            # Реквизиты шапки едут в result.report_metadata — тем же каналом,
            # что и для остальных форматов.
            pdf_bytes = pdf_fmt.format(result)
        except ImportError as exc:
            console.print(f"[red]Ошибка: {exc}[/red]")
            raise typer.Exit(code=2) from exc

        if output is None:
            # PDF — бинарный формат, без явного --output записываем рядом с входным файлом
            default_output = path.with_suffix(".pdf") if path.is_file() else Path("report.pdf")
            try:
                default_output.write_bytes(pdf_bytes)
                console.print(
                    I18n.get().t("cli.output_written", path=str(default_output))
                )
            except OSError as exc:
                console.print(f"[red]Ошибка записи PDF {default_output}: {exc}[/red]")
                raise typer.Exit(code=2) from exc
        else:
            try:
                output.write_bytes(pdf_bytes)
                console.print(I18n.get().t("cli.output_written", path=str(output)))
            except OSError as exc:
                console.print(f"[red]Ошибка записи PDF {output}: {exc}[/red]")
                raise typer.Exit(code=2) from exc

        report_str = None  # PDF не является текстом

    # Шаг 7: Сохраняем в файл если указан --output. PDF записан выше
    # (бинарный формат), здесь — все текстовые форматы, включая console.
    if output is not None and format in _TEXT_FORMATS:
        assert report_str is not None  # гарантировано ветками выше
        try:
            output.write_text(report_str, encoding="utf-8")
            console.print(I18n.get().t("cli.output_written", path=str(output)))
        except OSError as exc:
            console.print(f"[red]Ошибка записи файла {output}: {exc}[/red]")
            raise typer.Exit(code=2) from exc

    # Шаг 8: Exit code по результатам
    exit_code = _compute_exit_code(result, loaded_policy)

    # Шаг 9: Audit-лог (аудит #31). Запись метаданных сканирования в
    # JSON Lines файл — нужно для ИБ-аудитов в банках/госструктурах.
    # Тихо игнорируется если ни --audit-log, ни POISON_CHECK_AUDIT_LOG не заданы.
    from poison_check.audit import write_audit_record  # noqa: PLC0415

    write_audit_record(
        result=result,
        scanned_path=path,
        exit_code=exit_code,
        audit_log_path=audit_log,
    )

    raise typer.Exit(code=exit_code)


@app.command(name="list-scanners")
def list_scanners() -> None:
    """Показать список поддерживаемых форматов."""
    all_scanners = ScannerRegistry.all_scanners()

    if not all_scanners:
        console.print("[yellow]Нет зарегистрированных сканеров.[/yellow]")
        return

    table = Table(
        title="Поддерживаемые форматы ML-файлов",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Сканер", style="bold")
    table.add_column("Расширения")
    table.add_column("Описание")

    for scanner_class in sorted(all_scanners, key=lambda s: s.name):
        exts = ", ".join(scanner_class.supported_extensions)
        table.add_row(scanner_class.name, exts, scanner_class.description)

    console.print(table)


@app.command()
def doctor() -> None:
    """Диагностика окружения.

    Проверяет версию Python, наличие зависимостей, YAML-правила,
    зарегистрированные сканеры и детекторы.
    Быстрая самодиагностика установки — распространённый паттерн CLI-инструментов.
    """
    all_ok = True

    # --- Python версия ---
    console.print("\n[bold cyan]== Диагностика окружения poison-check ==[/bold cyan]\n")

    py_version = sys.version_info
    py_str = f"Python {py_version.major}.{py_version.minor}.{py_version.micro}"
    if py_version >= (3, 10):
        console.print(f"[green]✓[/green]  {py_str} (требуется >= 3.10)")
    else:
        console.print(f"[red]✗[/red]  {py_str} (требуется >= 3.10) — [red]ОБНОВИТЕ PYTHON[/red]")
        all_ok = False

    # --- Версия poison-check ---
    console.print(f"[green]✓[/green]  poison-check версия {_VERSION}")

    # --- Обязательные зависимости ---
    # Список, а не цепочка and — нужно выполнить и напечатать ВСЕ проверки,
    # а не остановиться на первой провалившейся.
    required_checks = [
        _check_dep("typer", "CLI-фреймворк", required=True),
        _check_dep("yaml", "Загрузка YAML-правил", required=True),
        _check_dep("rich", "Цветной вывод", required=True),
        _check_dep(
            "lz4",
            "Декомпрессия joblib (lz4)",
            required=True,
            impact="joblib-модели с lz4-сжатием не распаковываются и остаются непроверенными",
            install="pip install lz4",
        ),
        _check_dep(
            "zstandard",
            "Декомпрессия joblib (zstd)",
            required=True,
            impact="joblib-модели с zstd-сжатием не распаковываются и остаются непроверенными",
            install="pip install zstandard",
        ),
        _check_dep(
            "h5py",
            "Разбор HDF5-моделей Keras (.h5/.hdf5)",
            required=True,
            impact="форматы .h5/.hdf5 не проверяются на Lambda-RCE (CVE-2025-1550)",
            install="pip install h5py (или переустановите poison-check целиком)",
        ),
    ]
    if not all(required_checks):
        all_ok = False

    # --- Опциональные зависимости ---
    # Сканер читает .pt/.pth/.bin и .safetensors побайтово — сами
    # библиотеки torch и safetensors ему не нужны. Опциональны только PDF.
    _check_dep("jinja2", "PDF-отчёты (шаблоны)", required=False)
    _check_dep("weasyprint", "PDF-отчёты (рендер)", required=False)

    # --- YAML-правила ---
    rules_dir = RULES_DIR
    cve_yaml = rules_dir / "cve" / "ml_cves.yaml"
    allowlist_dir = rules_dir / "allowlist"

    console.print()
    if cve_yaml.is_file():
        console.print(f"[green]✓[/green]  CVE-правила: {cve_yaml}")
    else:
        console.print(f"[yellow]⚠[/yellow]  CVE-правила не найдены: {cve_yaml}")

    if allowlist_dir.is_dir():
        yamls = list(allowlist_dir.glob("*.yaml"))
        console.print(f"[green]✓[/green]  Allowlist-файлов: {len(yamls)} в {allowlist_dir}")
    else:
        console.print(f"[yellow]⚠[/yellow]  Директория allowlist не найдена: {allowlist_dir}")

    # --- Политики ---
    # Печатаем не только количество, но и name/description каждой политики:
    # иначе эти ключи YAML нигде не показываются пользователю и превращаются
    # в декоративные (см. реестр ключей в poison_check.policies).
    policies_dir = _POLICIES_DIR
    if policies_dir.is_dir():
        policy_files = sorted(policies_dir.glob("*.yaml"))
        console.print(f"[green]✓[/green]  Политик найдено: {len(policy_files)}")
        for policy_path in policy_files:
            try:
                loaded = PolicyLoader.load(policy_path.stem)
            except ValueError as exc:
                console.print(f"    [yellow]⚠ {policy_path.name}: {exc}[/yellow]")
                continue
            name = str(loaded.get("name") or policy_path.stem)
            description = str(loaded.get("description") or "")
            threshold = policy_severity_threshold(loaded)
            console.print(
                f"    • {name}: {description}"
                f" [dim](severity_threshold: "
                f"{threshold.value if threshold else '—'}, "
                f"fail_on_severity: {policy_fail_on_severity(loaded).value})[/dim]"
            )
    else:
        console.print(f"[yellow]⚠[/yellow]  Директория политик не найдена: {policies_dir}")

    # --- Зарегистрированные сканеры ---
    console.print()
    scanners = ScannerRegistry.all_scanners()
    console.print(f"[green]✓[/green]  Сканеров зарегистрировано: {len(scanners)}")
    for sc in scanners:
        console.print(f"    • {sc.name}: {', '.join(sc.supported_extensions)}")

    # --- Зарегистрированные детекторы ---
    detectors = DetectorRegistry.all_detectors()
    console.print(f"[green]✓[/green]  Детекторов зарегистрировано: {len(detectors)}")
    for dc in detectors:
        console.print(f"    • {dc.name}: {dc.description}")

    # --- Итог ---
    console.print()
    if all_ok:
        console.print("[bold green]✅ Всё готово к работе![/bold green]")
    else:
        console.print(
            "[bold red]❌ Обнаружены проблемы — исправьте их перед использованием.[/bold red]"
        )
        raise typer.Exit(code=1)


def _check_dep(
    module_name: str,
    description: str,
    required: bool,
    impact: str | None = None,
    install: str | None = None,
) -> bool:
    """Проверяет наличие Python-модуля и выводит статус.

    Для обязательной зависимости дополнительно печатает КОНКРЕТНОЕ последствие
    её отсутствия (что именно перестаёт проверяться) и команду установки —
    «нет пакета» само по себе не объясняет пользователю, какой участок
    сканирования ослеп.

    :param module_name: Импортируемое имя модуля (``h5py``, ``yaml``, ...).
    :param description: Назначение зависимости для вывода.
    :param required: Обязательна ли зависимость.
    :param impact: Последствие отсутствия; печатается только при провале.
    :param install: Команда установки; печатается только при провале.
    :return: True, если проверка пройдена (модуль есть либо он опционален).
    """
    available = importlib.util.find_spec(module_name) is not None
    if available:
        console.print(f"[green]✓[/green]  {module_name}: {description}")
        return True
    if required:
        console.print(
            f"[red]✗[/red]  {module_name}: {description} — [red]ОТСУТСТВУЕТ (обязательно)[/red]"
        )
        if impact:
            console.print(f"      [red]Последствие:[/red] {impact}")
        if install:
            console.print(f"      Установка: {install}")
        return False
    console.print(f"[dim]–  {module_name}: {description} (опционально, не установлен)[/dim]")
    return True


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

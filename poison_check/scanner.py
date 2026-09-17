"""Высокоуровневый фасад для сканирования ML-файлов.

Реализует принцип «Python API первого класса»:
«Каждый сканер и детектор должен быть вызываем программно с тем же качеством,
что и через CLI».

Полный цикл сканирования:
  1. FormatDetector → определение формата
  2. ScannerRegistry.find_scanner() → подбор сканера
  3. scanner.scan() → RawScanData
  4. MLContextAnalyzer.analyze() → MLContext
  5. Все детекторы из DetectorRegistry → Issues
  6. Сборка FileResult / ScanResult

Пример использования:
    from poison_check import Scanner

    result = Scanner().scan("model.pt")
    print(result.summary.critical)  # 2

    result = Scanner().scan("models/", recursive=True)
    for path, file_result in result.results_per_file.items():
        for issue in file_result.issues:
            print(f"{path}: [{issue.severity.value}] {issue.message}")

    file_result = Scanner().scan_bytes(b"...", filename="suspect.pkl")
    print(file_result.issues)
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Импортируем пакеты scanners и detectors, чтобы декораторы
# @ScannerRegistry.register / @DetectorRegistry.register выполнились.
import poison_check.detectors  # noqa: F401 — side-effect import
import poison_check.scanners  # noqa: F401 — side-effect import
from poison_check._paths import POLICIES_DIR
from poison_check.analysis.bundle import analyze_bundle, bundle_companion_files
from poison_check.analysis.ml_context import MLContextAnalyzer
from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry, ScannerRegistry
from poison_check.core.result import (
    ComplianceReport,
    FileResult,
    MLContext,
    ReportMetadata,
    ScanResult,
    Summary,
    dedupe_issues,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.parse_error_detector import annotate_suppressed_parse_error
from poison_check.i18n.loader import I18n
from poison_check.policies import (
    instantiate_detectors,
    policy_max_file_size_bytes,
    policy_thresholds,
)
from poison_check.scanners.format_facts import attach_format_facts

_VERSION = "0.1.0"


# _dedupe_issues перенесён в poison_check.core.result.dedupe_issues (аудит #11),
# чтобы использоваться и в Scanner-фасаде, и в CLI. Локальное имя оставлено
# как алиас для обратной совместимости в коде/тестах.
_dedupe_issues = dedupe_issues


def _build_compliance_report(
    result: ScanResult,
    policy: dict[str, Any],
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


class Scanner:
    """Высокоуровневый фасад для сканирования ML-моделей.

    Инкапсулирует полный цикл сканирования:
    подбор сканера → извлечение данных → определение контекста → детекция угроз.

    Параметры:
        policy:  Имя политики из каталога policies/ ("default", "banking",
                 "government", "strict"). Влияет на набор детекторов, правила
                 из ``extra_rules`` (``max_file_size_gb`` — лимит размера файла,
                 ``no_external_urls`` — эскалация внешних URL до HIGH,
                 ``no_unverified_files`` — эскалация непроверенного файла) и на
                 пороги, которые едут в ``ScanResult.policy_thresholds``:
                 ``severity_threshold`` (порог внимания, применяется
                 форматтерами) и ``fail_on_severity`` (порог действия).
        locale:  Язык сообщений ("ru" или "en"). По умолчанию "ru".
        report_metadata: Реквизиты шапки отчёта (заказчик / аудитор) — то же,
                 что CLI-опции ``--client`` и ``--auditor``. Едут в
                 ``ScanResult.report_metadata`` и выводятся всеми форматтерами.

    Пример базового использования:
        result = Scanner().scan("model.pt")
        print(result.summary.critical)

    Пример рекурсивного сканирования:
        result = Scanner(policy="banking").scan("models/", recursive=True)

    Пример сканирования из байтов (для CI/CD пайплайнов):
        result = Scanner().scan_bytes(raw_bytes, filename="uploaded.pkl")
    """

    def __init__(
        self,
        policy: str = "default",
        locale: str = "ru",
        report_metadata: ReportMetadata | None = None,
    ) -> None:
        """Инициализирует фасад с указанной политикой и локалью.

        :param policy: Имя политики. Если файл не найден — используется пустая политика.
        :param locale: Локаль для сообщений.
        :param report_metadata: Реквизиты шапки отчёта. Паритет с CLI
            (``--client``/``--auditor``): Python API формирует такой же отчёт,
            что и командная строка (Python API первого класса).
        """
        self._policy_name = policy
        self._locale = locale
        self._report_metadata = report_metadata
        self._loaded_policy: dict[str, Any] = self._load_policy(policy)
        # Кеш экземпляров детекторов — создаётся один раз при первом вызове scan()
        self._detectors: list[BaseDetector] | None = None
        # Инициализируем i18n с нужной локалью
        I18n.set_locale(locale)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def scan(
        self, path: str | Path, recursive: bool = False, bundle_check: bool = True
    ) -> ScanResult:
        """Сканирует файл или директорию и возвращает ScanResult.

        Полный цикл сканирования.

        :param path: Путь к файлу или директории.
        :param recursive: Если True — рекурсивно сканирует все файлы в директории.
        :param bundle_check: Если False — проверка комплекта модели (код рядом с
            весами, префикс MLS-BUNDLE-*) не выполняется вовсе: находки не идут ни
            в отчёт, ни в exit-код. По умолчанию включена. Паритет с CLI-флагом
            ``--no-bundle-check``.
        :return: ScanResult с результатами по всем файлам и агрегированным summary.
        :raises FileNotFoundError: Если указанный путь не существует.

        Пример:
            result = Scanner().scan("model.pt")
            result = Scanner().scan("models/", recursive=True)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Путь не найден: {path}")

        files_to_scan = self._collect_files(path, recursive)
        t_start = time.monotonic()

        # Directory-level проверка комплекта модели (код рядом с весами).
        # companion-файлы (config.json / *.py) исключаются из пофайлового цикла:
        # их разбирает bundle-анализ, и ошибка «неподдерживаемый формат» по ним
        # была бы ложной (см. poison_check.analysis.bundle).
        bundle_result: FileResult | None = None
        if path.is_dir():
            companions = bundle_companion_files(files_to_scan)
            files_to_scan = [f for f in files_to_scan if f not in companions]
            if bundle_check:
                bundle_result = analyze_bundle(
                    path, self._get_detectors(self._loaded_policy)
                )

        results_per_file: dict[Path, FileResult] = {}
        for file_path in files_to_scan:
            file_result, _ctx = self._scan_single_file(file_path)
            results_per_file[file_path] = file_result

        scanned_paths = list(files_to_scan)
        if bundle_result is not None:
            results_per_file[path] = bundle_result
            scanned_paths.append(path)

        duration_ms = (time.monotonic() - t_start) * 1000
        return self._build_scan_result(
            scanned_paths=scanned_paths,
            results_per_file=results_per_file,
            duration_ms=duration_ms,
        )

    def scan_bytes(self, data: bytes, filename: str = "unknown") -> FileResult:
        """Сканирует байты ML-файла и возвращает FileResult.

        Удобно для интеграции в CI/CD пайплайны, где файл приходит
        в виде байтового потока (например, из объектного хранилища).

        Данные временно сохраняются во временный файл с указанным именем,
        чтобы сканер мог определить формат по расширению.
        Временный файл удаляется сразу после сканирования.

        :param data: Байты файла.
        :param filename: Имя файла с расширением (используется для определения формата).
        :return: FileResult с найденными issues.

        Пример:
            with open("model.pkl", "rb") as f:
                raw = f.read()
            result = Scanner().scan_bytes(raw, filename="model.pkl")
        """
        suffix = Path(filename).suffix or ".bin"
        # Создаём временный файл с правильным расширением
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="poison_check_"
        ) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            file_result, _ctx = self._scan_single_file(
                path=tmp_path,
                display_name=filename,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        return file_result

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_scanners_registered() -> None:
        """No-op — оставлен для обратной совместимости (аудит #12).

        Раньше этот метод вручную перечислял все сканеры и детекторы и
        перерегистрировал их если реестр был сброшен через ``_reset()`` в тестах.
        Это требовало синхронизации списка при добавлении нового класса —
        легко забыть. Теперь тесты используют snapshot/restore через копирование
        ``_scanners`` / ``_detectors`` (см. ``tests/test_core_registry.py``),
        благодаря чему глобальный реестр не разрушается между тестами.

        Если по какой-то причине реестры пусты на момент вызова — это значит,
        что ни один модуль ``scanners.*`` или ``detectors.*`` не был импортирован.
        Импорт ``poison_check.scanners`` и ``poison_check.detectors`` уже
        выполнен на уровне модуля ``poison_check.scanner`` (см. side-effect
        импорты выше) — то есть в нормальной ситуации реестры не могут быть пусты.
        """
        return

    def _get_detectors(self, policy: dict[str, Any]) -> list[BaseDetector]:
        """Возвращает кешированные экземпляры детекторов для текущей политики.

        Детекторы создаются только один раз при первом вызове.
        Это устраняет O(N) создание объектов при сканировании N файлов:
        AllowlistDetector читает YAML при инициализации, SecretsDetector
        компилирует regex — обе операции дорогие и не нужны при каждом файле.

        Параметры конструкторов берутся из ``extra_rules`` политики через
        ``policies.instantiate_detectors`` — та же функция используется в CLI
        (``cli._get_cached_detectors``), поэтому оба пути дают одинаковое
        поведение (например, эскалацию внешних URL при ``no_external_urls``).

        :param policy: Загруженная политика (определяет набор детекторов).
        :return: Список готовых экземпляров детекторов.
        """
        if self._detectors is None:
            self._ensure_scanners_registered()
            detector_classes = DetectorRegistry.enabled_for_policy(policy)
            self._detectors = instantiate_detectors(detector_classes, policy)
        return self._detectors

    def _scan_single_file(
        self,
        path: Path,
        display_name: str | None = None,
    ) -> tuple[FileResult, MLContext]:
        """Сканирует один файл. Возвращает FileResult + MLContext.

        :param path: Реальный путь к файлу.
        :param display_name: Отображаемое имя (для scan_bytes, где путь временный).
        """
        t_start = time.monotonic()
        display_path = Path(display_name) if display_name else path

        self._ensure_scanners_registered()
        scanner_class = ScannerRegistry.find_scanner(path)
        if scanner_class is None:
            # Неподдерживаемый формат. Детекторы всё равно запускаются: именно
            # здесь видно расхождение «расширение ↔ содержимое» для файлов,
            # которые ни один сканер не берёт (safetensors с именем .pkl).
            # Поведение прежнее — MLS-PARSE-001 подавлен флагом
            # META_FORMAT_UNSUPPORTED, прочие детекторы без данных молчат.
            i18n = I18n.get()
            err = i18n.t("errors.unsupported_format", path=str(display_path))
            scanner_name = "unknown"
            raw_data = RawScanData(
                file_path=display_path,
                file_hash={},
                file_size=0,
                scanner_name=scanner_name,
                error=err,
            )
            attach_format_facts(raw_data, path, unsupported=True)
            ctx = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])
        else:
            # Лимит размера берётся из политики (extra_rules.max_file_size_gb).
            # У Python API нет аналога флага --max-file-size, поэтому значение
            # политики применяется напрямую; None → DEFAULT_MAX_FILE_SIZE сканера.
            scanner = scanner_class(
                max_file_size=policy_max_file_size_bytes(self._loaded_policy)
            )
            scanner_name = scanner_class.name
            raw_data = scanner.scan(path)

            # Факты о формате (расхождение с расширением, класс безопасности) —
            # единая точка на оба пути, до запуска детекторов. Считаются по
            # реальному пути: у scan_bytes временный файл, но расширение
            # display_name сохраняется через суффикс временного файла.
            attach_format_facts(raw_data, path)

            # Для scan_bytes подменяем file_path в RawScanData на display_name
            if display_name:
                raw_data.file_path = display_path

            # Определяем ML-контекст
            ctx = MLContextAnalyzer().analyze(raw_data)

        # Запускаем все разрешённые политикой детекторы.
        # Экземпляры детекторов кешируются в self._detectors — O(1) при повторных вызовах.
        all_issues = []
        for detector in self._get_detectors(self._loaded_policy):
            try:
                issues = detector.analyze(raw_data, ctx)
                all_issues.extend(issues)
            except Exception as exc:  # noqa: BLE001
                # Детектор не должен ронять весь прогон — но молча терять ошибку
                # тоже нельзя: для banking-режима «детектор не нашёл секретов»
                # неотличимо от «детектор упал». Логируем с полным трейсбэком.
                import logging  # noqa: PLC0415

                logging.getLogger(__name__).warning(
                    "Детектор %s упал на файле %s: %s",
                    detector.name,
                    path,
                    exc,
                    exc_info=True,
                )

        # Дедупликация: BlocklistDetector и CVEDetector могут оба сработать на
        # одну и ту же CVE (например, blocklist выдаёт MLS-PKL-001 для os.system,
        # а CVEDetector — MLS-PATTERN-OS-SYSTEM с теми же location/details).
        # Пользователь не должен видеть две разные строки про один и тот же
        # вредоносный глобал. Дедупим по (code, location) — это ключ из спеки
        # SARIF, который пользователь и видит как «уникальное событие».
        all_issues = _dedupe_issues(all_issues)

        # Краевой случай «payload + оборванный хвост»: если MLS-PARSE-001 подавлен
        # (по файлу есть КРИТ по извлечённому глобалу), но поток опкодов оборвался
        # после payload — протаскиваем факт обрыва в details находок, иначе он
        # теряется в SARIF (несёт только Issues, не FileResult.error).
        annotate_suppressed_parse_error(all_issues, raw_data)

        duration_ms = (time.monotonic() - t_start) * 1000
        file_result = FileResult(
            file_path=display_path,
            scanner_name=scanner_name,
            issues=all_issues,
            duration_ms=duration_ms,
            error=raw_data.error,
        )

        return file_result, ctx

    def _build_scan_result(
        self,
        scanned_paths: list[Path],
        results_per_file: dict[Path, FileResult],
        duration_ms: float,
    ) -> ScanResult:
        """Собирает ScanResult из результатов по файлам."""
        counts: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
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
            policy=self._policy_name,
            results_per_file=results_per_file,
            summary=summary,
            compliance_report=None,
            # Те же пороги, что и в CLI: JsonFormatter().format(result) через
            # Python API даёт отчёт, идентичный `poison-check scan --format json`.
            policy_thresholds=policy_thresholds(self._loaded_policy),
            # Реквизиты шапки — тем же каналом, что и в CLI.
            report_metadata=self._report_metadata,
        )

        compliance_report = _build_compliance_report(partial_result, self._loaded_policy)

        partial_result.compliance_report = compliance_report
        return partial_result

    @staticmethod
    def _collect_files(path: Path, recursive: bool) -> list[Path]:
        """Собирает список файлов для сканирования.

        Если path — файл, возвращает [path] (если это обычный файл, не symlink/FIFO).
        Если path — директория:
            recursive=True  → все файлы рекурсивно (кроме скрытых директорий)
            recursive=False → только файлы в корне директории.

        Безопасность (аудит #5): отбрасываем symlink, FIFO, сокеты и устройства.
        Это DoS-защита: symlink на ``/dev/zero`` или FIFO даст бесконечный поток
        для любого потокового сканера.
        """
        import stat as _stat

        def _is_regular(p: Path) -> bool:
            try:
                st = p.stat()
            except OSError:
                return False
            return _stat.S_ISREG(st.st_mode)

        if path.is_symlink():
            return []

        if path.is_file() and _is_regular(path):
            return [path]

        if path.is_dir():
            if recursive:
                return [
                    p
                    for p in sorted(path.rglob("*"))
                    if p.is_file()
                    and not p.is_symlink()
                    and _is_regular(p)
                    and not any(part.startswith(".") for part in p.parts)
                ]
            return sorted(
                p
                for p in path.iterdir()
                if p.is_file() and not p.is_symlink() and _is_regular(p)
            )

        return []

    @staticmethod
    def _load_policy(name: str) -> dict[str, Any]:
        """Загружает политику из YAML-файла.

        При ошибке возвращает пустую политику (все детекторы включены).
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            return {}

        policy_path = POLICIES_DIR / f"{name}.yaml"
        if not policy_path.is_file():
            return {}
        try:
            raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            # Не падаем если политику не удалось прочитать — у пользователя
            # должна быть возможность запустить сканер на дефолтных настройках.
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                "Не удалось загрузить политику %s: %s — используется пустая политика",
                policy_path,
                exc,
            )
            return {}
        return raw if isinstance(raw, dict) else {}

"""Тесты Python API — высокоуровневый фасад Scanner и форматтеры SARIF/SBOM.

Покрывает высокоуровневый публичный API (Python API первого класса).

Тесты не зависят от сети и внешних сервисов.
Временные файлы создаются через tmp_path fixture pytest.
"""

from __future__ import annotations

import json
import pickle
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

from poison_check.core.result import (
    Confidence,
    FileResult,
    Issue,
    Reference,
    ScanResult,
    Severity,
    Summary,
)
from poison_check.output.sarif import SarifFormatter
from poison_check.output.sbom import SbomFormatter

# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

_SAFE_PICKLE = Path(__file__).parent / "fixtures" / "safe" / "simple_list.pkl"
_MALICIOUS_PICKLE = (
    Path(__file__).parent / "fixtures" / "malicious" / "payload_01_os_system.pkl"
)


def _make_scan_result(
    *,
    with_issues: bool = False,
    with_cve: bool = False,
    extra_file: Path | None = None,
) -> ScanResult:
    """Строит ScanResult для использования в тестах форматтеров.

    :param with_issues: Добавить тестовую Issue в результат.
    :param with_cve: Добавить CVE-ссылку к Issue.
    :param extra_file: Добавить второй файл в results_per_file.
    """
    main_path = Path("tests/fixtures/safe/simple_list.pkl")

    issues: list[Issue] = []
    if with_issues:
        refs: list[Reference] = []
        if with_cve:
            refs = [
                Reference(type="cve", id="CVE-2025-32434"),
                Reference(type="cwe", id="CWE-502"),
            ]
        issues.append(
            Issue(
                code="MLS-PKL-001",
                severity=Severity.CRITICAL,
                confidence=Confidence.CERTAIN,
                message="Обнаружен вызов системной команды в pickle-файле",
                location="simple_list.pkl (offset 0)",
                why="os.system позволяет выполнять произвольные команды ОС",
                remediation="Не загружать модель, пересохранить в safetensors",
                references=refs,
                compliance_tags=["owasp-ml:ml03", "fstec:ubi-067"],
                details={"module": "os", "function": "system"},
            )
        )

    results_per_file: dict[Path, FileResult] = {
        main_path: FileResult(
            file_path=main_path,
            scanner_name="pickle",
            issues=issues,
            duration_ms=12.5,
        )
    }

    if extra_file is not None:
        results_per_file[extra_file] = FileResult(
            file_path=extra_file,
            scanner_name="pytorch",
            issues=[],
            duration_ms=5.0,
        )

    return ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        duration_ms=42.0,
        scanned_paths=[main_path],
        policy="default",
        results_per_file=results_per_file,
        summary=Summary(
            critical=len([i for i in issues if i.severity == Severity.CRITICAL]),
            high=0,
            medium=0,
            low=0,
            info=0,
        ),
    )


# ---------------------------------------------------------------------------
# Тесты: импорт from poison_check import Scanner
# ---------------------------------------------------------------------------


def test_import_scanner_works() -> None:
    """from poison_check import Scanner — работает без ошибок."""
    from poison_check import Scanner  # noqa: PLC0415

    assert Scanner is not None


def test_scanner_instantiation() -> None:
    """Scanner() создаётся с параметрами по умолчанию без исключений."""
    from poison_check import Scanner  # noqa: PLC0415

    s = Scanner()
    assert s is not None


def test_scanner_custom_policy_and_locale() -> None:
    """Scanner(policy='strict', locale='en') создаётся без исключений."""
    from poison_check import Scanner  # noqa: PLC0415

    s = Scanner(policy="strict", locale="en")
    assert s is not None


# ---------------------------------------------------------------------------
# Тесты: Scanner.scan() → ScanResult
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SAFE_PICKLE.exists(),
    reason="Fixture tests/fixtures/safe/simple_list.pkl не найден",
)
def test_scanner_scan_returns_scan_result() -> None:
    """Scanner().scan(safe_pickle) → ScanResult с корректными полями."""
    from poison_check import Scanner  # noqa: PLC0415

    result = Scanner().scan(_SAFE_PICKLE)

    assert isinstance(result, ScanResult)
    assert result.tool_version == "0.1.0"
    assert isinstance(result.timestamp, datetime)
    assert result.duration_ms >= 0.0
    assert len(result.results_per_file) == 1
    assert result.summary is not None


@pytest.mark.skipif(
    not _SAFE_PICKLE.exists(),
    reason="Fixture tests/fixtures/safe/simple_list.pkl не найден",
)
def test_scanner_scan_with_str_path() -> None:
    """Scanner().scan() принимает str и Path одинаково."""
    from poison_check import Scanner  # noqa: PLC0415

    result_str = Scanner().scan(str(_SAFE_PICKLE))
    result_path = Scanner().scan(_SAFE_PICKLE)

    assert len(result_str.results_per_file) == len(result_path.results_per_file)


def test_scanner_scan_nonexistent_path_raises() -> None:
    """Scanner().scan('nonexistent/') → FileNotFoundError."""
    from poison_check import Scanner  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        Scanner().scan("nonexistent_path_xyz_12345.pkl")


def test_scanner_scan_directory(tmp_path: Path) -> None:
    """Scanner().scan(dir) сканирует все файлы в директории без рекурсии."""
    from poison_check import Scanner  # noqa: PLC0415

    # Создаём безопасный pickle
    pkl_file = tmp_path / "test.pkl"
    pkl_file.write_bytes(pickle.dumps([1, 2, 3], protocol=2))

    result = Scanner().scan(tmp_path)

    assert isinstance(result, ScanResult)
    assert len(result.results_per_file) >= 1


def test_scanner_scan_recursive(tmp_path: Path) -> None:
    """Scanner().scan(dir, recursive=True) рекурсивно обходит поддиректории."""
    from poison_check import Scanner  # noqa: PLC0415

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (tmp_path / "model1.pkl").write_bytes(pickle.dumps([1], protocol=2))
    (subdir / "model2.pkl").write_bytes(pickle.dumps([2], protocol=2))

    result = Scanner().scan(tmp_path, recursive=True)

    assert isinstance(result, ScanResult)
    assert len(result.results_per_file) == 2


# ---------------------------------------------------------------------------
# Тесты: Scanner.scan_bytes() → FileResult
# ---------------------------------------------------------------------------


def test_scanner_scan_bytes_safe_pickle() -> None:
    """Scanner().scan_bytes(pickle_bytes) → FileResult без ошибок."""
    from poison_check import Scanner  # noqa: PLC0415

    # Безопасный pickle: простой список
    data = pickle.dumps([1, 2, 3], protocol=2)
    file_result = Scanner().scan_bytes(data, filename="test.pkl")

    assert isinstance(file_result, FileResult)
    assert file_result.scanner_name != "unknown"
    assert file_result.error is None


def test_scanner_scan_bytes_uses_filename_extension() -> None:
    """scan_bytes использует расширение filename для определения формата.

    .pkl берёт либо pickle, либо joblib (оба поддерживают это расширение).
    Главное — не 'unknown', т.е. формат был опознан.
    """
    from poison_check import Scanner  # noqa: PLC0415

    data = pickle.dumps({"key": "value"}, protocol=2)
    file_result = Scanner().scan_bytes(data, filename="model.pkl")

    # Должен быть разобран каким-то сканером, не unknown
    assert file_result.scanner_name != "unknown"
    assert file_result.error is None


def test_scanner_scan_bytes_display_name_in_result() -> None:
    """FileResult.file_path содержит filename переданный в scan_bytes."""
    from poison_check import Scanner  # noqa: PLC0415

    data = pickle.dumps([42], protocol=2)
    file_result = Scanner().scan_bytes(data, filename="my_model.pkl")

    assert file_result.file_path.name == "my_model.pkl"


def test_scanner_scan_bytes_unknown_extension() -> None:
    """scan_bytes с неизвестным расширением → FileResult с error (не исключение)."""
    from poison_check import Scanner  # noqa: PLC0415

    file_result = Scanner().scan_bytes(b"\x00\x01\x02\x03", filename="unknown.xyz")

    # Сканер не упал — вернул FileResult
    assert isinstance(file_result, FileResult)


# ---------------------------------------------------------------------------
# Тесты: SarifFormatter
# ---------------------------------------------------------------------------


def test_sarif_formatter_returns_valid_json() -> None:
    """SarifFormatter.format() возвращает валидный JSON."""
    result = _make_scan_result()
    sarif_str = SarifFormatter().format(result)

    # Проверяем парсинг без исключений
    sarif = json.loads(sarif_str)
    assert isinstance(sarif, dict)


def test_sarif_formatter_has_schema_field() -> None:
    """SARIF-документ содержит поле '$schema'."""
    result = _make_scan_result()
    sarif = json.loads(SarifFormatter().format(result))

    assert "$schema" in sarif
    assert "sarif" in sarif["$schema"].lower()


def test_sarif_formatter_has_version_field() -> None:
    """SARIF-документ содержит поле 'version' == '2.1.0'."""
    result = _make_scan_result()
    sarif = json.loads(SarifFormatter().format(result))

    assert sarif["version"] == "2.1.0"


def test_sarif_formatter_has_runs() -> None:
    """SARIF-документ содержит поле 'runs' с одним прогоном."""
    result = _make_scan_result()
    sarif = json.loads(SarifFormatter().format(result))

    assert "runs" in sarif
    assert len(sarif["runs"]) == 1


def test_sarif_formatter_tool_driver_name() -> None:
    """runs[0].tool.driver.name == 'poison-check'."""
    result = _make_scan_result()
    sarif = json.loads(SarifFormatter().format(result))

    driver = sarif["runs"][0]["tool"]["driver"]
    assert driver["name"] == "poison-check"


def test_sarif_formatter_empty_results_on_clean_file() -> None:
    """Для чистого файла (без issues) SARIF-results пустой."""
    result = _make_scan_result(with_issues=False)
    sarif = json.loads(SarifFormatter().format(result))

    assert sarif["runs"][0]["results"] == []


def test_sarif_formatter_issue_mapped_to_result() -> None:
    """Каждый Issue → один SARIF result с корректными полями."""
    result = _make_scan_result(with_issues=True)
    sarif = json.loads(SarifFormatter().format(result))

    results = sarif["runs"][0]["results"]
    assert len(results) == 1

    r = results[0]
    assert r["ruleId"] == "MLS-PKL-001"
    assert r["level"] in ("error", "warning", "note")
    assert "message" in r
    assert "text" in r["message"]
    assert r["message"]["text"] != ""


def test_sarif_formatter_critical_maps_to_error() -> None:
    """CRITICAL severity → SARIF level 'error'."""
    result = _make_scan_result(with_issues=True)
    sarif = json.loads(SarifFormatter().format(result))

    # MLS001 имеет severity=CRITICAL → должен быть level=error
    results = sarif["runs"][0]["results"]
    assert any(r["level"] == "error" for r in results)


def test_sarif_formatter_rules_deduplicated() -> None:
    """Правила дедуплицированы по ruleId — один code → одно правило."""
    # Создаём результат с двумя Issues с одинаковым кодом
    issues = [
        Issue(
            code="MLS-PKL-001",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            message="Первый MLS001",
            location="a.pkl",
        ),
        Issue(
            code="MLS-PKL-001",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            message="Второй MLS001",
            location="b.pkl",
        ),
    ]
    file_path = Path("test.pkl")
    result = ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        duration_ms=1.0,
        results_per_file={
            file_path: FileResult(
                file_path=file_path,
                scanner_name="pickle",
                issues=issues,
            )
        },
        summary=Summary(high=2),
    )

    sarif = json.loads(SarifFormatter().format(result))
    rules = sarif["runs"][0]["tool"]["driver"]["rules"]

    rule_ids = [r["id"] for r in rules]
    assert rule_ids.count("MLS-PKL-001") == 1


def test_sarif_formatter_locations_present() -> None:
    """Каждый SARIF result содержит locations с physicalLocation."""
    result = _make_scan_result(with_issues=True)
    sarif = json.loads(SarifFormatter().format(result))

    r = sarif["runs"][0]["results"][0]
    assert "locations" in r
    assert len(r["locations"]) > 0
    loc = r["locations"][0]
    assert "physicalLocation" in loc
    assert "artifactLocation" in loc["physicalLocation"]
    assert "uri" in loc["physicalLocation"]["artifactLocation"]


def test_sarif_formatter_with_cve_references() -> None:
    """Issues с CVE-ссылками получают relatedLocations."""
    result = _make_scan_result(with_issues=True, with_cve=True)
    sarif = json.loads(SarifFormatter().format(result))

    r = sarif["runs"][0]["results"][0]
    assert "relatedLocations" in r
    assert len(r["relatedLocations"]) > 0


def test_sarif_formatter_artifacts_present() -> None:
    """SARIF-документ содержит artifacts с URI всех просканированных файлов."""
    result = _make_scan_result()
    sarif = json.loads(SarifFormatter().format(result))

    artifacts = sarif["runs"][0]["artifacts"]
    assert len(artifacts) == 1
    assert "location" in artifacts[0]
    assert "uri" in artifacts[0]["location"]


# ---------------------------------------------------------------------------
# Тесты: SbomFormatter
# ---------------------------------------------------------------------------


def test_sbom_formatter_returns_valid_json() -> None:
    """SbomFormatter.format() возвращает валидный JSON."""
    result = _make_scan_result()
    sbom_str = SbomFormatter().format(result)

    sbom = json.loads(sbom_str)
    assert isinstance(sbom, dict)


def test_sbom_formatter_has_bom_format() -> None:
    """SBOM-документ содержит поле 'bomFormat': 'CycloneDX'."""
    result = _make_scan_result()
    sbom = json.loads(SbomFormatter().format(result))

    assert sbom["bomFormat"] == "CycloneDX"


def test_sbom_formatter_spec_version() -> None:
    """SBOM-документ содержит поле 'specVersion': '1.4'."""
    result = _make_scan_result()
    sbom = json.loads(SbomFormatter().format(result))

    assert sbom["specVersion"] == "1.4"


def test_sbom_formatter_has_components() -> None:
    """SBOM-документ содержит список компонентов (по одному на ML-файл)."""
    result = _make_scan_result()
    sbom = json.loads(SbomFormatter().format(result))

    assert "components" in sbom
    assert len(sbom["components"]) == 1


def test_sbom_formatter_component_type() -> None:
    """Компоненты имеют type 'machine-learning-model'."""
    result = _make_scan_result()
    sbom = json.loads(SbomFormatter().format(result))

    comp = sbom["components"][0]
    assert comp["type"] == "machine-learning-model"


def test_sbom_formatter_component_has_name() -> None:
    """Компонент содержит имя файла в поле 'name'."""
    result = _make_scan_result()
    sbom = json.loads(SbomFormatter().format(result))

    comp = sbom["components"][0]
    assert "name" in comp
    assert comp["name"] == "simple_list.pkl"


def test_sbom_formatter_vulnerabilities_empty_on_clean_file() -> None:
    """Для чистого файла (без issues) список уязвимостей пустой."""
    result = _make_scan_result(with_issues=False)
    sbom = json.loads(SbomFormatter().format(result))

    assert sbom["vulnerabilities"] == []


def test_sbom_formatter_vulnerability_from_issue() -> None:
    """Каждый Issue → одна vulnerability с корректными полями."""
    result = _make_scan_result(with_issues=True)
    sbom = json.loads(SbomFormatter().format(result))

    vulns = sbom["vulnerabilities"]
    assert len(vulns) == 1

    v = vulns[0]
    assert "id" in v
    assert "ratings" in v
    assert len(v["ratings"]) > 0
    assert "severity" in v["ratings"][0]
    assert "description" in v
    assert "affects" in v


def test_sbom_formatter_cve_id_used_when_present() -> None:
    """Если Issue имеет CVE-ссылку, vulnerability.id = CVE-идентификатор."""
    result = _make_scan_result(with_issues=True, with_cve=True)
    sbom = json.loads(SbomFormatter().format(result))

    vuln_id = sbom["vulnerabilities"][0]["id"]
    assert vuln_id == "CVE-2025-32434"


def test_sbom_formatter_cwe_in_vuln() -> None:
    """CWE-ссылки попадают в vulnerability.cwes как числа."""
    result = _make_scan_result(with_issues=True, with_cve=True)
    sbom = json.loads(SbomFormatter().format(result))

    vuln = sbom["vulnerabilities"][0]
    assert "cwes" in vuln
    assert 502 in vuln["cwes"]


def test_sbom_formatter_metadata_tool_name() -> None:
    """SBOM-документ содержит метаданные с именем инструмента."""
    result = _make_scan_result()
    sbom = json.loads(SbomFormatter().format(result))

    assert "metadata" in sbom
    tools = sbom["metadata"]["tools"]
    assert any(t["name"] == "poison-check" for t in tools)


def test_sbom_formatter_serial_number_unique() -> None:
    """Каждый вызов format() генерирует уникальный serialNumber (UUID)."""
    result = _make_scan_result()
    formatter = SbomFormatter()

    sbom1 = json.loads(formatter.format(result))
    sbom2 = json.loads(formatter.format(result))

    assert sbom1["serialNumber"] != sbom2["serialNumber"]


def test_sbom_formatter_multiple_files() -> None:
    """Несколько файлов → несколько компонентов в SBOM."""
    extra = Path("tests/fixtures/safe/simple_model.pt")
    result = _make_scan_result(extra_file=extra)
    sbom = json.loads(SbomFormatter().format(result))

    assert len(sbom["components"]) == 2


# ---------------------------------------------------------------------------
# Интеграционные тесты: Scanner + форматтеры
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SAFE_PICKLE.exists(),
    reason="Fixture tests/fixtures/safe/simple_list.pkl не найден",
)
def test_scanner_result_to_sarif_valid() -> None:
    """Scanner().scan(safe_pkl) → SarifFormatter().format() → валидный SARIF."""
    from poison_check import Scanner  # noqa: PLC0415

    result = Scanner().scan(_SAFE_PICKLE)
    sarif_str = SarifFormatter().format(result)
    sarif = json.loads(sarif_str)

    assert sarif["version"] == "2.1.0"
    assert "$schema" in sarif
    assert "runs" in sarif


@pytest.mark.skipif(
    not _SAFE_PICKLE.exists(),
    reason="Fixture tests/fixtures/safe/simple_list.pkl не найден",
)
def test_scanner_result_to_sbom_valid() -> None:
    """Scanner().scan(safe_pkl) → SbomFormatter().format() → валидный CycloneDX SBOM."""
    from poison_check import Scanner  # noqa: PLC0415

    result = Scanner().scan(_SAFE_PICKLE)
    sbom_str = SbomFormatter().format(result)
    sbom = json.loads(sbom_str)

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.4"
    assert "components" in sbom


def test_scan_bytes_then_sarif(tmp_path: Path) -> None:
    """scan_bytes → ScanResult → SarifFormatter → валидный SARIF."""
    from poison_check import Scanner  # noqa: PLC0415

    # Простой безопасный pickle
    data = pickle.dumps([1, 2, 3], protocol=2)
    file_result = Scanner().scan_bytes(data, filename="inline.pkl")

    # Оборачиваем FileResult в ScanResult для форматтера
    scan_result = ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        duration_ms=1.0,
        results_per_file={file_result.file_path: file_result},
        summary=Summary(),
    )

    sarif = json.loads(SarifFormatter().format(scan_result))
    assert "$schema" in sarif
    assert sarif["version"] == "2.1.0"


def test_scan_bytes_then_sbom(tmp_path: Path) -> None:
    """scan_bytes → ScanResult → SbomFormatter → валидный CycloneDX SBOM."""
    from poison_check import Scanner  # noqa: PLC0415

    data = pickle.dumps({"key": "value"}, protocol=2)
    file_result = Scanner().scan_bytes(data, filename="model.pkl")

    scan_result = ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        duration_ms=1.0,
        results_per_file={file_result.file_path: file_result},
        summary=Summary(),
    )

    sbom = json.loads(SbomFormatter().format(scan_result))
    assert sbom["bomFormat"] == "CycloneDX"
    assert len(sbom["components"]) == 1


# ---------------------------------------------------------------------------
# Тесты: кеш детекторов — производительность
# ---------------------------------------------------------------------------


def test_scanner_detectors_instantiated_once(tmp_path: Path) -> None:
    """AllowlistDetector.__init__ вызывается только один раз при двух вызовах scan().

    Проверяет оптимизацию _get_detectors(): детекторы создаются при первом
    вызове scan() и кешируются в self._detectors, поэтому при сканировании
    N файлов нет O(N) создания объектов.
    """
    from unittest.mock import patch

    from poison_check import Scanner  # noqa: PLC0415
    from poison_check.detectors.allowlist_detector import AllowlistDetector  # noqa: PLC0415

    # Создаём два безопасных pickle-файла
    file1 = tmp_path / "model1.pkl"
    file2 = tmp_path / "model2.pkl"
    file1.write_bytes(pickle.dumps([1, 2, 3], protocol=2))
    file2.write_bytes(pickle.dumps({"a": 1}, protocol=2))

    scanner = Scanner()

    # Считаем вызовы __init__ через side_effect, чтобы избежать TypeError
    # возникающего при wraps=original_init (mock теряет self при вызове unbound метода).
    call_count = 0
    original_init = AllowlistDetector.__init__

    def counting_init(self: AllowlistDetector) -> None:  # type: ignore[misc]
        nonlocal call_count
        call_count += 1
        original_init(self)

    with patch.object(AllowlistDetector, "__init__", counting_init):
        # Первое сканирование — детекторы создаются и кешируются
        scanner.scan(file1)
        # Второе сканирование — детекторы берутся из кеша, __init__ не вызывается
        scanner.scan(file2)

    # AllowlistDetector.__init__ должен быть вызван ровно один раз
    assert call_count == 1, (
        f"AllowlistDetector.__init__ вызван {call_count} раз(а), "
        f"ожидалось 1 — детекторы не кешируются между вызовами scan()"
    )


# ---------------------------------------------------------------------------
# Регрессия аудита #22: расширенный публичный API
# ---------------------------------------------------------------------------


class TestExtendedPublicAPI:
    """Forensics-юзеры должны иметь прямой доступ к Issue/Severity и т.п."""

    def test_severity_imported_from_top_level(self) -> None:
        """`from poison_check import Severity` работает."""
        from poison_check import Severity  # noqa: PLC0415

        assert Severity.CRITICAL.value == "critical"

    def test_issue_constructible_from_top_level(self) -> None:
        """Issue/Confidence/Severity конструируются через top-level импорты."""
        from poison_check import Confidence as Conf
        from poison_check import Issue as Iss
        from poison_check import Reference as Ref
        from poison_check import Severity as Sev

        i = Iss(
            code="X-001",
            severity=Sev.HIGH,
            confidence=Conf.MEDIUM,
            message="test",
            location="x.pkl",
            references=[Ref(type="cwe", id="CWE-502")],
        )
        assert i.code == "X-001"
        assert i.severity is Sev.HIGH

    def test_scan_result_imported(self) -> None:
        """ScanResult/FileResult/Summary доступны из top-level."""
        from poison_check import (  # noqa: PLC0415
            ComplianceReport,
            FileResult,
            MLContext,
            ScanResult,
            Summary,
        )

        # Минимальный smoke: всё это dataclass'ы
        assert hasattr(ScanResult, "__dataclass_fields__")
        assert hasattr(FileResult, "__dataclass_fields__")
        assert hasattr(Summary, "__dataclass_fields__")
        assert hasattr(MLContext, "__dataclass_fields__")
        assert hasattr(ComplianceReport, "__dataclass_fields__")

    def test_dedupe_issues_callable(self) -> None:
        """dedupe_issues экспортирован и работает."""
        from poison_check import Confidence, Issue, Severity, dedupe_issues  # noqa: PLC0415

        dup = Issue(
            code="A", severity=Severity.HIGH, confidence=Confidence.HIGH,
            message="m", location="x",
        )
        result = dedupe_issues([dup, dup])
        assert len(result) == 1

    def test_all_listed_in_dunder_all(self) -> None:
        """Все экспорты документированы в __all__."""
        import poison_check  # noqa: PLC0415

        for name in (
            "Scanner", "Issue", "Severity", "Confidence", "Reference",
            "ScanResult", "FileResult", "Summary", "MLContext",
            "ComplianceReport", "dedupe_issues",
        ):
            assert name in poison_check.__all__, f"{name} не в __all__"


# ---------------------------------------------------------------------------
# Регрессия аудита #29: SARIF informationUri параметризован
# ---------------------------------------------------------------------------


class TestSarifInformationUri:
    """SarifFormatter позволяет переопределить informationUri."""

    def test_default_uri_used(self) -> None:
        """Без override — используется TOOL_INFORMATION_URI."""
        from poison_check.output.sarif import SarifFormatter as _SF  # noqa: PLC0415

        result = _make_scan_result()
        sarif = json.loads(_SF().format(result))
        uri = sarif["runs"][0]["tool"]["driver"]["informationUri"]
        # Дефолт — это TOOL_INFORMATION_URI или env override (если задан в CI)
        assert uri  # не пустой
        assert isinstance(uri, str)

    def test_constructor_argument_overrides(self) -> None:
        """Параметр конструктора имеет наивысший приоритет."""
        from poison_check.output.sarif import SarifFormatter as _SF  # noqa: PLC0415

        result = _make_scan_result()
        custom_uri = "https://example.org/poison-check"
        sarif = json.loads(_SF(information_uri=custom_uri).format(result))
        assert (
            sarif["runs"][0]["tool"]["driver"]["informationUri"] == custom_uri
        )

    def test_env_var_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POISON_CHECK_INFORMATION_URI в env → используется в SARIF."""
        from poison_check.output.sarif import SarifFormatter as _SF  # noqa: PLC0415

        env_uri = "https://my-fork.example/poison-check"
        monkeypatch.setenv("POISON_CHECK_INFORMATION_URI", env_uri)
        result = _make_scan_result()
        sarif = json.loads(_SF().format(result))
        assert sarif["runs"][0]["tool"]["driver"]["informationUri"] == env_uri

    def test_constructor_arg_beats_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Конструктор имеет приоритет над env."""
        from poison_check.output.sarif import SarifFormatter as _SF  # noqa: PLC0415

        monkeypatch.setenv("POISON_CHECK_INFORMATION_URI", "https://from-env.example")
        result = _make_scan_result()
        sarif = json.loads(_SF(information_uri="https://from-arg.example").format(result))
        assert (
            sarif["runs"][0]["tool"]["driver"]["informationUri"]
            == "https://from-arg.example"
        )

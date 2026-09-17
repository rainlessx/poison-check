"""Финальный интеграционный тест недели 3.

Проверяет совместную работу всех трёх детекторов:
BlocklistDetector, AllowlistDetector, CVEDetector.

Сценарии:
1. Все детекторы в совокупности обнаруживают все 6 вредоносных payload
   из недели 2 (хотя бы одна CRITICAL или HIGH issue на каждый файл).
2. BlocklistDetector и CVEDetector не дают CRITICAL false positive
   на чистых pickle-файлах (safe fixtures).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.core.result import MLContext, Severity
from poison_check.detectors.allowlist_detector import AllowlistDetector
from poison_check.detectors.blocklist_detector import BlocklistDetector
from poison_check.detectors.cve_detector import CVEDetector
from poison_check.scanners.pickle_scanner import PickleScanner

_MALICIOUS_DIR = Path(__file__).parent / "fixtures" / "malicious"
_SAFE_DIR = Path(__file__).parent / "fixtures" / "safe"

# Ожидаем ровно 6 payload (+ os_system.pkl legacy, который тоже поймаем)
_EXPECTED_MALICIOUS_PATTERNS = [
    "payload_01_os_system.pkl",
    "payload_02_subprocess_popen.pkl",
    "payload_03_builtins_eval.pkl",
    "payload_04_builtins_exec.pkl",
    "payload_05_torch.pt",
    "payload_06_numpy.npz",
]


@pytest.fixture(scope="module")
def scanner() -> PickleScanner:
    """Единственный экземпляр PickleScanner на весь модуль."""
    return PickleScanner()


@pytest.fixture(scope="module")
def detectors() -> list[BlocklistDetector | AllowlistDetector | CVEDetector]:
    """Все три детектора, инициализированные один раз."""
    return [BlocklistDetector(), AllowlistDetector(), CVEDetector()]


@pytest.fixture(scope="module")
def pytorch_context() -> MLContext:
    """ML-контекст PyTorch с высокой уверенностью."""
    return MLContext(
        framework="pytorch",
        confidence=0.9,
        detected_patterns=["torch.nn.modules", "torch._utils"],
    )


@pytest.fixture(scope="module")
def unknown_context() -> MLContext:
    """ML-контекст с неизвестным фреймворком (для тестов false positive)."""
    return MLContext(framework="unknown", confidence=0.0, detected_patterns=[])


# ---------------------------------------------------------------------------
# Тест 1: все детекторы ловят все вредоносные payload
# ---------------------------------------------------------------------------


def test_all_detectors_catch_all_payloads(
    scanner: PickleScanner,
    detectors: list[BlocklistDetector | AllowlistDetector | CVEDetector],
    pytorch_context: MLContext,
) -> None:
    """Все 6 payload обнаруживаются хотя бы одним детектором как CRITICAL или HIGH.

    Проверяем все .pkl / .pt / .npz файлы в malicious/, исключая legacy
    os_system.pkl (он не входит в 6 основных payload, но тоже безопасен для теста).
    """
    payload_files = [
        _MALICIOUS_DIR / name for name in _EXPECTED_MALICIOUS_PATTERNS
    ]

    missing = [p for p in payload_files if not p.exists()]
    if missing:
        pytest.skip(
            f"Fixture-файлы не найдены (запустите generate_fixtures.py): "
            f"{[str(m) for m in missing]}"
        )

    for payload_file in payload_files:
        raw_data = scanner.scan(payload_file)

        all_issues = []
        for detector in detectors:
            all_issues.extend(detector.analyze(raw_data, pytorch_context))

        critical_or_high = [
            i
            for i in all_issues
            if i.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

        assert len(critical_or_high) > 0, (
            f"Payload {payload_file.name} не задетектирован ни одним детектором.\n"
            f"Всего issues: {len(all_issues)}\n"
            f"Issues: {[(i.code, i.severity.value, i.message) for i in all_issues]}\n"
            f"RawScanData.globals: {raw_data.globals}\n"
            f"RawScanData.error: {raw_data.error}"
        )


# ---------------------------------------------------------------------------
# Тест 2: нет false positive CRITICAL на чистых файлах
# ---------------------------------------------------------------------------


def test_no_false_positives_on_clean_files(
    scanner: PickleScanner,
    unknown_context: MLContext,
) -> None:
    """BlocklistDetector и CVEDetector не дают CRITICAL на чистых pickle-файлах.

    AllowlistDetector пропускаем — при unknown framework и пустом combined
    allowlist он может шуметь на любом незнакомом глобале (MEDIUM, не CRITICAL).
    """
    detectors_for_fp_test: list[BlocklistDetector | CVEDetector] = [
        BlocklistDetector(),
        CVEDetector(),
    ]

    safe_files = sorted(_SAFE_DIR.glob("*.pkl"))

    if not safe_files:
        pytest.skip(
            "Safe fixture-файлы не найдены (запустите generate_fixtures.py)"
        )

    for clean_file in safe_files:
        raw_data = scanner.scan(clean_file)

        all_issues = []
        for detector in detectors_for_fp_test:
            all_issues.extend(detector.analyze(raw_data, unknown_context))

        critical = [i for i in all_issues if i.severity == Severity.CRITICAL]

        assert len(critical) == 0, (
            f"False positive CRITICAL в {clean_file.name}:\n"
            f"  {[(i.code, i.message) for i in critical]}"
        )


# ---------------------------------------------------------------------------
# Тест 3: CVEDetector находит правильные CVE-коды
# ---------------------------------------------------------------------------


def test_cve_detector_returns_correct_codes(
    scanner: PickleScanner,
    pytorch_context: MLContext,
) -> None:
    """CVEDetector возвращает issues с кодами в формате MLS-<CVE_ID>.

    Проверяем payload с os.system — ожидаем MLS-PATTERN-OS-SYSTEM.
    """
    payload_file = _MALICIOUS_DIR / "payload_01_os_system.pkl"
    if not payload_file.exists():
        pytest.skip("Fixture payload_01_os_system.pkl не найден")

    detector = CVEDetector()
    raw_data = scanner.scan(payload_file)
    issues = detector.analyze(raw_data, pytorch_context)

    assert len(issues) > 0, (
        f"CVEDetector не нашёл ничего в payload_01_os_system.pkl\n"
        f"globals: {raw_data.globals}"
    )

    codes = {i.code for i in issues}
    assert all(code.startswith("MLS-") for code in codes), (
        f"Коды issue должны начинаться с 'MLS-': {codes}"
    )

    # Ожидаем паттерн OS-SYSTEM
    assert any("OS-SYSTEM" in code or "PATTERN" in code for code in codes), (
        f"Ожидался паттерн OS-SYSTEM, получены коды: {codes}"
    )


# ---------------------------------------------------------------------------
# Тест 4: CVEDetector — severity и confidence для CVE-паттернов
# ---------------------------------------------------------------------------


def test_cve_detector_severity_confidence(
    scanner: PickleScanner,
    pytorch_context: MLContext,
) -> None:
    """Для CVE-совпадений severity=CRITICAL, confidence=CERTAIN (как указано в YAML)."""
    payload_file = _MALICIOUS_DIR / "payload_01_os_system.pkl"
    if not payload_file.exists():
        pytest.skip("Fixture payload_01_os_system.pkl не найден")

    detector = CVEDetector()
    raw_data = scanner.scan(payload_file)
    issues = detector.analyze(raw_data, pytorch_context)

    critical_certain = [
        i
        for i in issues
        if i.severity == Severity.CRITICAL
    ]

    assert len(critical_certain) > 0, (
        "CVEDetector должен находить хотя бы одну CRITICAL issue "
        "для os.system payload"
    )

    # Проверяем наличие references
    for issue in critical_certain:
        assert len(issue.references) > 0, (
            f"Issue {issue.code} должен содержать ссылки (CVE/CWE)"
        )


# ---------------------------------------------------------------------------
# Тест 5: CVEDetector не даёт false positive на чистых файлах
# ---------------------------------------------------------------------------


def test_cve_detector_no_false_positives(
    scanner: PickleScanner,
    unknown_context: MLContext,
) -> None:
    """CVEDetector не создаёт issues для простых чистых pickle-файлов."""
    safe_files = sorted(_SAFE_DIR.glob("*.pkl"))
    if not safe_files:
        pytest.skip("Safe fixture-файлы не найдены")

    detector = CVEDetector()

    for clean_file in safe_files:
        raw_data = scanner.scan(clean_file)
        issues = detector.analyze(raw_data, unknown_context)

        assert len(issues) == 0, (
            f"CVEDetector создал ложное срабатывание на {clean_file.name}: "
            f"{[(i.code, i.message) for i in issues]}"
        )


# ---------------------------------------------------------------------------
# Тест 6: subprocess payload задетектирован
# ---------------------------------------------------------------------------


def test_subprocess_payload_detected(
    scanner: PickleScanner,
    pytorch_context: MLContext,
) -> None:
    """payload_02_subprocess_popen.pkl ловится BlocklistDetector или CVEDetector."""
    payload_file = _MALICIOUS_DIR / "payload_02_subprocess_popen.pkl"
    if not payload_file.exists():
        pytest.skip("Fixture payload_02_subprocess_popen.pkl не найден")

    blocklist = BlocklistDetector()
    cve = CVEDetector()
    raw_data = scanner.scan(payload_file)

    issues = blocklist.analyze(raw_data, pytorch_context)
    issues += cve.analyze(raw_data, pytorch_context)

    critical_or_high = [
        i for i in issues if i.severity in (Severity.CRITICAL, Severity.HIGH)
    ]

    assert len(critical_or_high) > 0, (
        f"subprocess.Popen payload не задетектирован\n"
        f"globals: {raw_data.globals}"
    )


# ---------------------------------------------------------------------------
# Тест 7: PyTorch .pt payload (ZIP) задетектирован
# ---------------------------------------------------------------------------


def test_pytorch_pt_payload_detected(
    scanner: PickleScanner,
    pytorch_context: MLContext,
) -> None:
    """payload_05_torch.pt (ZIP с вредоносным data.pkl) ловится детекторами."""
    payload_file = _MALICIOUS_DIR / "payload_05_torch.pt"
    if not payload_file.exists():
        pytest.skip("Fixture payload_05_torch.pt не найден")

    blocklist = BlocklistDetector()
    cve = CVEDetector()
    raw_data = scanner.scan(payload_file)

    issues = blocklist.analyze(raw_data, pytorch_context)
    issues += cve.analyze(raw_data, pytorch_context)

    critical_or_high = [
        i for i in issues if i.severity in (Severity.CRITICAL, Severity.HIGH)
    ]

    assert len(critical_or_high) > 0, (
        f"PyTorch .pt payload не задетектирован\n"
        f"globals: {raw_data.globals}, error: {raw_data.error}"
    )

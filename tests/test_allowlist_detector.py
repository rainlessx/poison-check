"""Тесты для AllowlistDetector."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from poison_check.core.result import Confidence, MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.allowlist_detector import AllowlistDetector
from poison_check.detectors.blocklist_detector import BlocklistDetector

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

_DUMMY_PATH = Path("test.pkl")


def _make_raw(
    globals_set: set[tuple[str, str]] | None = None,
    file_path: Path = _DUMMY_PATH,
) -> RawScanData:
    """Минимальный RawScanData для тестов AllowlistDetector."""
    return RawScanData(
        file_path=file_path,
        file_hash={},
        file_size=100,
        scanner_name="pickle",
        globals=globals_set,
    )


def _pytorch_context() -> MLContext:
    """MLContext для PyTorch-модели."""
    return MLContext(framework="pytorch", confidence=0.95)


def _sklearn_context() -> MLContext:
    """MLContext для sklearn-модели."""
    return MLContext(framework="sklearn", confidence=0.90)


def _unknown_context() -> MLContext:
    """MLContext с неизвестным фреймворком."""
    return MLContext(framework="unknown", confidence=0.0)


@pytest.fixture
def detector() -> AllowlistDetector:
    """AllowlistDetector с дефолтными YAML из rules/allowlist/."""
    return AllowlistDetector()


# ---------------------------------------------------------------------------
# Тест 1: чистая PyTorch модель → пустой список issues
# ---------------------------------------------------------------------------


def test_clean_pytorch_globals_produce_no_issues(detector: AllowlistDetector) -> None:
    """Стандартные globals легитимной PyTorch-модели → пустой список issues.

    Все перечисленные глобалы должны быть в pytorch.yaml allowlist.
    """
    pytorch_globals: set[tuple[str, str]] = {
        ("torch", "Tensor"),
        ("torch", "FloatTensor"),
        ("torch", "LongTensor"),
        ("torch", "Size"),
        ("torch", "_utils"),
        ("torch.storage", "_load_from_bytes"),
        ("torch", "UntypedStorage"),
        ("torch.nn.modules.linear", "Linear"),
        ("torch.nn.modules.activation", "ReLU"),
        ("torch.nn.modules.normalization", "LayerNorm"),
        ("torch.nn.modules.container", "Sequential"),
        ("collections", "OrderedDict"),
        ("_codecs", "encode"),
        ("numpy.core.multiarray", "_reconstruct"),
    }
    raw = _make_raw(globals_set=pytorch_globals)
    issues = detector.analyze(raw, _pytorch_context())

    assert issues == [], (
        f"Ожидался пустой список, но получены issues: "
        f"{[(i.details.get('module'), i.details.get('name')) for i in issues]}"
    )


def test_clean_sklearn_globals_produce_no_issues(detector: AllowlistDetector) -> None:
    """Стандартные globals sklearn-модели → пустой список issues."""
    sklearn_globals: set[tuple[str, str]] = {
        ("sklearn.linear_model._logistic", "LogisticRegression"),
        ("sklearn.preprocessing._data", "StandardScaler"),
        ("sklearn.pipeline", "Pipeline"),
        ("numpy", "ndarray"),
        ("numpy", "float64"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("collections", "OrderedDict"),
        ("_codecs", "encode"),
    }
    raw = _make_raw(globals_set=sklearn_globals)
    issues = detector.analyze(raw, _sklearn_context())

    assert issues == []


def test_none_globals_produce_no_issues(detector: AllowlistDetector) -> None:
    """globals=None → пустой список (нечего проверять)."""
    raw = _make_raw(globals_set=None)
    issues = detector.analyze(raw, _pytorch_context())

    assert issues == []


def test_empty_globals_set_produces_no_issues(detector: AllowlistDetector) -> None:
    """Пустое множество globals → пустой список."""
    raw = _make_raw(globals_set=set())
    issues = detector.analyze(raw, _pytorch_context())

    assert issues == []


# ---------------------------------------------------------------------------
# Тест 2: os.system → AllowlistDetector НЕ дублирует BlocklistDetector
# ---------------------------------------------------------------------------


def test_os_system_not_duplicated_by_allowlist_detector(
    detector: AllowlistDetector,
) -> None:
    """os.system в globals → AllowlistDetector не создаёт Issue (это зона BlocklistDetector).

    Разделение обязанностей: BlocklistDetector поймает os.system с CRITICAL,
    AllowlistDetector должен промолчать — иначе пользователь получит два Issue
    об одном и том же.
    """
    raw = _make_raw(globals_set={("os", "system")})
    issues = detector.analyze(raw, _pytorch_context())

    assert issues == [], (
        "AllowlistDetector не должен создавать Issue для глобалов из hardcoded blocklist"
    )


def test_subprocess_popen_not_duplicated(detector: AllowlistDetector) -> None:
    """subprocess.Popen → AllowlistDetector молчит (покрыто blocklist)."""
    raw = _make_raw(globals_set={("subprocess", "Popen")})
    issues = detector.analyze(raw, _pytorch_context())

    assert issues == []


def test_builtins_eval_not_duplicated(detector: AllowlistDetector) -> None:
    """builtins.eval → AllowlistDetector молчит (покрыто blocklist)."""
    raw = _make_raw(globals_set={("builtins", "eval")})
    issues = detector.analyze(raw, _pytorch_context())

    assert issues == []


def test_blocklist_detector_still_catches_what_allowlist_skips(
    detector: AllowlistDetector,
) -> None:
    """Совместный запуск: os.system → только 1 CRITICAL от BlocklistDetector, не 2.

    Имитирует реальный пайплайн: оба детектора применяются к одним данным,
    суммарно должен быть ровно один Issue на os.system.
    """
    raw = _make_raw(globals_set={("os", "system")})
    context = _pytorch_context()

    blocklist_detector = BlocklistDetector()
    blocklist_issues = blocklist_detector.analyze(raw, context)
    allowlist_issues = detector.analyze(raw, context)

    all_issues = blocklist_issues + allowlist_issues
    # Должен быть ровно один Issue — от BlocklistDetector
    assert len(all_issues) == 1
    assert all_issues[0].severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Тест 3: неизвестный global → Issue с MEDIUM
# ---------------------------------------------------------------------------


def test_unknown_global_produces_medium_issue(detector: AllowlistDetector) -> None:
    """Неизвестный global ('mycorp.custom', 'Loader') → Issue с severity=MEDIUM."""
    raw = _make_raw(globals_set={("mycorp.custom", "Loader")})
    issues = detector.analyze(raw, _pytorch_context())

    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == Severity.MEDIUM
    assert issue.confidence == Confidence.LOW
    assert issue.code == "MLS-ALW-001"
    assert "mycorp.custom" in issue.message
    assert "Loader" in issue.message


def test_unknown_global_details_populated(detector: AllowlistDetector) -> None:
    """Issue для неизвестного global содержит корректные details."""
    raw = _make_raw(globals_set={("evil.corp", "steal_weights")})
    issues = detector.analyze(raw, _sklearn_context())

    assert len(issues) == 1
    issue = issues[0]
    assert issue.details["module"] == "evil.corp"
    assert issue.details["name"] == "steal_weights"
    assert issue.details["framework"] == "sklearn"


def test_unknown_global_has_why_and_remediation(detector: AllowlistDetector) -> None:
    """Issue для неизвестного global содержит непустые why и remediation."""
    raw = _make_raw(globals_set={("unknown_lib", "unknown_func")})
    issues = detector.analyze(raw, _pytorch_context())

    assert len(issues) == 1
    assert issues[0].why is not None and len(issues[0].why) > 0
    assert issues[0].remediation is not None and len(issues[0].remediation) > 0


def test_multiple_unknown_globals_produce_multiple_issues(
    detector: AllowlistDetector,
) -> None:
    """Несколько неизвестных globals → отдельный Issue для каждого."""
    raw = _make_raw(
        globals_set={
            ("corp.ml", "CustomLayer"),
            ("corp.util", "Loader"),
        }
    )
    issues = detector.analyze(raw, _pytorch_context())

    assert len(issues) == 2
    modules = {i.details["module"] for i in issues}
    assert "corp.ml" in modules
    assert "corp.util" in modules


def test_mix_of_known_and_unknown_globals(detector: AllowlistDetector) -> None:
    """Смесь allowlisted и неизвестных globals → Issue только для неизвестных."""
    raw = _make_raw(
        globals_set={
            ("torch", "Tensor"),           # в allowlist → тихо
            ("collections", "OrderedDict"), # в allowlist → тихо
            ("evil.lib", "RCE"),            # не в allowlist → Issue
        }
    )
    issues = detector.analyze(raw, _pytorch_context())

    assert len(issues) == 1
    assert issues[0].details["module"] == "evil.lib"
    assert issues[0].details["name"] == "RCE"


# ---------------------------------------------------------------------------
# Тест 4: unknown framework → используется объединённый allowlist
# ---------------------------------------------------------------------------


def test_unknown_framework_uses_combined_allowlist(detector: AllowlistDetector) -> None:
    """При framework='unknown' используется объединение всех allowlists.

    Globals из разных фреймворков должны проходить без Issues,
    так как они есть хотя бы в одном из загруженных allowlists.
    """
    # torch.Tensor из pytorch.yaml, sklearn.pipeline.Pipeline из sklearn.yaml,
    # numpy.ndarray из numpy.yaml — все должны быть в объединённом allowlist
    mixed_globals: set[tuple[str, str]] = {
        ("torch", "Tensor"),
        ("sklearn.pipeline", "Pipeline"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("collections", "OrderedDict"),
    }
    raw = _make_raw(globals_set=mixed_globals)
    issues = detector.analyze(raw, _unknown_context())

    assert issues == [], (
        f"При unknown framework объединённый allowlist должен молчать на известных "
        f"глобалах. Получены: "
        f"{[(i.details.get('module'), i.details.get('name')) for i in issues]}"
    )


def test_unknown_framework_still_flags_suspicious(detector: AllowlistDetector) -> None:
    """При framework='unknown' полностью неизвестный global → Issue."""
    raw = _make_raw(globals_set={("totally.unknown.lib", "DoEvil")})
    issues = detector.analyze(raw, _unknown_context())

    assert len(issues) == 1
    assert issues[0].code == "MLS-ALW-001"
    assert issues[0].severity == Severity.MEDIUM


def test_unknown_framework_framework_in_details(detector: AllowlistDetector) -> None:
    """Issue при unknown framework указывает 'unknown' в details.framework."""
    raw = _make_raw(globals_set={("weird.lib", "WeirdClass")})
    issues = detector.analyze(raw, _unknown_context())

    assert len(issues) == 1
    assert issues[0].details["framework"] == "unknown"


# ---------------------------------------------------------------------------
# Дополнительные тесты: robustness и edge cases
# ---------------------------------------------------------------------------


def test_detector_loads_without_exception() -> None:
    """AllowlistDetector инициализируется без исключений."""
    det = AllowlistDetector()
    assert det is not None


def test_detector_with_missing_dir_does_not_crash(tmp_path: Path) -> None:
    """При отсутствии директории allowlist детектор не падает.

    Все globals будут подозрительны (пустой объединённый allowlist),
    но исключений нет.
    """
    missing_dir = tmp_path / "nonexistent_allowlist"
    det = AllowlistDetector(allowlist_dir=missing_dir)

    raw = _make_raw(globals_set={("torch", "Tensor")})
    issues = det.analyze(raw, _pytorch_context())

    # Без allowlist — torch.Tensor тоже подозрителен, это ожидаемо
    assert len(issues) == 1
    assert issues[0].code == "MLS-ALW-001"


def test_detector_with_custom_allowlist_yaml(tmp_path: Path) -> None:
    """AllowlistDetector принимает кастомную директорию с YAML."""
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    custom_yaml = allowlist_dir / "custom.yaml"
    custom_yaml.write_text(
        yaml.dump({
            "framework": "custom",
            "description": "Тестовый allowlist",
            "version": "1.0",
            "globals": [
                {"module": "mycorp.ml", "name": "SafeModel", "reason": "Наша модель"},
                {"module": "mycorp.ml", "name": "SafeLayer", "reason": "Наш слой"},
            ],
        }),
        encoding="utf-8",
    )
    det = AllowlistDetector(allowlist_dir=allowlist_dir)
    context = MLContext(framework="custom", confidence=1.0)

    # Разрешённые globals → тихо
    raw_clean = _make_raw(globals_set={("mycorp.ml", "SafeModel")})
    assert det.analyze(raw_clean, context) == []

    # Неизвестный global → Issue
    raw_bad = _make_raw(globals_set={("mycorp.ml", "UnsafeLoader")})
    issues = det.analyze(raw_bad, context)
    assert len(issues) == 1
    assert issues[0].code == "MLS-ALW-001"


# ---------------------------------------------------------------------------
# Тест 5: severity tiers — trusted_prefixes → INFO vs MEDIUM
# ---------------------------------------------------------------------------


def test_trusted_prefix_module_produces_info_issue(detector: AllowlistDetector) -> None:
    """Неизвестный класс из доверенного пространства имён (sklearn.*) → INFO, MLS-ALW-002."""
    raw = _make_raw(globals_set={("sklearn.new_module", "NewEstimator")})
    issues = detector.analyze(raw, _sklearn_context())

    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == Severity.INFO
    assert issue.code == "MLS-ALW-002"
    assert issue.confidence == Confidence.LOW
    assert "sklearn.new_module" in issue.message
    assert "NewEstimator" in issue.message


def test_trusted_prefix_torch_produces_info_issue(detector: AllowlistDetector) -> None:
    """Неизвестный класс из torch.* (доверенный) → INFO."""
    raw = _make_raw(globals_set={("torch.nn.modules.custom", "CustomLayer")})
    issues = detector.analyze(raw, _pytorch_context())

    assert len(issues) == 1
    assert issues[0].severity == Severity.INFO
    assert issues[0].code == "MLS-ALW-002"


def test_untrusted_module_produces_medium_issue(detector: AllowlistDetector) -> None:
    """Полностью незнакомый модуль (не в trusted_prefixes) → MEDIUM, MLS-ALW-001."""
    raw = _make_raw(globals_set={("malicious.corp", "Exfiltrator")})
    issues = detector.analyze(raw, _sklearn_context())

    assert len(issues) == 1
    assert issues[0].severity == Severity.MEDIUM
    assert issues[0].code == "MLS-ALW-001"


def test_trusted_prefixes_loaded_from_yaml(tmp_path: Path) -> None:
    """trusted_prefixes из YAML корректно загружаются и применяются."""
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    (allowlist_dir / "myfw.yaml").write_text(
        yaml.dump({
            "framework": "myfw",
            "description": "Test",
            "version": "1.0",
            "trusted_prefixes": ["myfw.", "safe_lib."],
            "globals": [
                {"module": "myfw.core", "name": "BaseModel", "reason": "OK"},
            ],
        }),
        encoding="utf-8",
    )
    det = AllowlistDetector(allowlist_dir=allowlist_dir)
    context = MLContext(framework="myfw", confidence=1.0)

    # Разрешённый global → тихо
    raw_clean = _make_raw(globals_set={("myfw.core", "BaseModel")})
    assert det.analyze(raw_clean, context) == []

    # Неизвестный класс из trusted prefix → INFO
    raw_trusted = _make_raw(globals_set={("myfw.new_module", "NewModel")})
    issues_trusted = det.analyze(raw_trusted, context)
    assert len(issues_trusted) == 1
    assert issues_trusted[0].severity == Severity.INFO
    assert issues_trusted[0].code == "MLS-ALW-002"

    # Полностью незнакомый модуль → MEDIUM
    raw_unknown = _make_raw(globals_set={("evil.corp", "Payload")})
    issues_unknown = det.analyze(raw_unknown, context)
    assert len(issues_unknown) == 1
    assert issues_unknown[0].severity == Severity.MEDIUM
    assert issues_unknown[0].code == "MLS-ALW-001"


def test_allowlist_yaml_with_broken_entry_skips_it(tmp_path: Path) -> None:
    """Запись без module или name в YAML пропускается, остальные загружаются."""
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    (allowlist_dir / "test.yaml").write_text(
        yaml.dump({
            "framework": "test",
            "description": "Test",
            "version": "1.0",
            "globals": [
                {"module": "good.lib", "name": "GoodClass", "reason": "OK"},
                {"module": "broken_no_name", "reason": "Missing name field"},
                {"name": "no_module_here", "reason": "Missing module"},
            ],
        }),
        encoding="utf-8",
    )
    det = AllowlistDetector(allowlist_dir=allowlist_dir)
    context = MLContext(framework="test", confidence=1.0)

    # Корректная запись загрузилась
    raw = _make_raw(globals_set={("good.lib", "GoodClass")})
    assert det.analyze(raw, context) == []

    # Сломанные записи не попали в allowlist — но и падения нет
    raw_unknown = _make_raw(globals_set={("other.lib", "Other")})
    issues = det.analyze(raw_unknown, context)
    assert len(issues) == 1

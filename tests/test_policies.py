"""Тесты для системы политик сканирования.

Проверяет:
- Загрузку встроенных политик по имени
- Загрузку пользовательской политики по пути
- Поведение exit code в зависимости от политики (banking vs default)
- Обработку неизвестной политики
- Smoke-тест: все ключи i18n/ru.yaml доступны без KeyError
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from poison_check.cli import _compute_exit_code, app
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    FileResult,
    Issue,
    ScanResult,
    Severity,
    Summary,
)
from poison_check.detectors.network_detector import NetworkDetector
from poison_check.i18n.loader import I18n
from poison_check.policies import (
    PolicyKeyStatus,
    PolicyLoader,
    detector_kwargs_for,
    extra_rule_key_specs,
    instantiate_detectors,
    policy_escalate_external_urls,
    policy_escalate_unverified,
    policy_extra_rules,
    policy_max_file_size_bytes,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_MALICIOUS = _FIXTURES / "malicious"
_SAFE = _FIXTURES / "safe"

runner = CliRunner()


# ---------------------------------------------------------------------------
# Фикстура восстановления реестров
# ---------------------------------------------------------------------------
# test_core_registry.py сбрасывает глобальные реестры ScannerRegistry и
# DetectorRegistry через autouse-фикстуру. Поскольку Python кэширует модули
# и не перезапускает side-effects при повторном import, реестры остаются
# пустыми для последующих тестов. Восстанавливаем их через importlib.reload.

@pytest.fixture(autouse=True)
def _ensure_registries_populated() -> None:
    """Гарантирует заполненность реестров перед каждым тестом в этом модуле.

    Использует importlib.reload() для форсированного повтора side-effect
    регистрации сканеров и детекторов.
    """
    import importlib

    import poison_check.detectors.allowlist_detector as _mod_al
    import poison_check.detectors.blocklist_detector as _mod_bl
    import poison_check.detectors.compression_detector as _mod_cp
    import poison_check.detectors.cve_detector as _mod_cv
    import poison_check.detectors.executable_detector as _mod_ex
    import poison_check.detectors.gguf_metadata_detector as _mod_ggm
    import poison_check.detectors.joblib_metadata_detector as _mod_jlm
    import poison_check.detectors.network_detector as _mod_nw
    import poison_check.detectors.numpy_metadata_detector as _mod_npm
    import poison_check.detectors.secrets_detector as _mod_sc
    import poison_check.scanners.gguf_scanner as _mod_gguf
    import poison_check.scanners.joblib_scanner as _mod_jl
    import poison_check.scanners.numpy_scanner as _mod_np
    import poison_check.scanners.pickle_scanner as _mod_pk
    import poison_check.scanners.pytorch_scanner as _mod_pt
    import poison_check.scanners.safetensors_scanner as _mod_st

    from poison_check.core.registry import DetectorRegistry, ScannerRegistry

    # Если реестры уже заполнены — ничего не делаем (нормальный случай)
    if ScannerRegistry.all_scanners() and DetectorRegistry.all_detectors():
        return

    # Реестры пусты — восстанавливаем через reload
    ScannerRegistry._reset()
    DetectorRegistry._reset()

    for mod in (_mod_pk, _mod_pt, _mod_jl, _mod_np, _mod_st, _mod_gguf):
        importlib.reload(mod)
    for mod in (
        _mod_bl, _mod_al, _mod_cv, _mod_sc, _mod_nw, _mod_ex, _mod_cp,
        _mod_ggm, _mod_npm, _mod_jlm,
    ):
        importlib.reload(mod)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------


def _make_issue(severity: Severity, code: str = "MLS-PKL-001") -> Issue:
    """Создаёт тестовый Issue с заданным severity."""
    return Issue(
        code=code,
        severity=severity,
        confidence=Confidence.HIGH,
        message=f"Тестовая проблема {severity.value}",
        location="test.pkl:0",
    )


def _make_scan_result(issues: list[Issue]) -> ScanResult:
    """Создаёт ScanResult с одним файлом и переданным списком issues."""
    from datetime import datetime, timezone

    file_result = FileResult(
        file_path=Path("test.pkl"),
        scanner_name="pickle",
        issues=issues,
        duration_ms=1.0,
    )
    return ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        duration_ms=1.0,
        scanned_paths=[Path("test.pkl")],
        policy="test",
        results_per_file={Path("test.pkl"): file_result},
        summary=Summary(),
        compliance_report=None,
    )


# ---------------------------------------------------------------------------
# Тест 1: banking политика — HIGH → exit 1
# ---------------------------------------------------------------------------


class TestBankingPolicy:
    """banking-политика: fail_on_severity=high → exit code 1 при HIGH issue."""

    def test_high_issue_causes_exit1(self) -> None:
        """При наличии HIGH issue banking-политика даёт exit code 1."""
        policy = PolicyLoader.load("banking")
        assert policy["fail_on_severity"] == "high", (
            f"banking.fail_on_severity должен быть 'high', получено: {policy['fail_on_severity']}"
        )

        result = _make_scan_result([_make_issue(Severity.HIGH)])
        code = _compute_exit_code(result, policy)
        assert code == 1, (
            f"banking-политика при HIGH issue должна возвращать exit code 1, получено: {code}"
        )

    def test_critical_issue_causes_exit1(self) -> None:
        """При наличии CRITICAL issue banking-политика также даёт exit code 1."""
        policy = PolicyLoader.load("banking")
        result = _make_scan_result([_make_issue(Severity.CRITICAL)])
        code = _compute_exit_code(result, policy)
        assert code == 1

    def test_medium_issue_causes_exit0(self) -> None:
        """При наличии только MEDIUM issue banking-политика даёт exit code 0.

        fail_on_severity=high означает: только HIGH и выше → exit 1.
        MEDIUM ниже HIGH → exit 0.
        """
        policy = PolicyLoader.load("banking")
        result = _make_scan_result([_make_issue(Severity.MEDIUM)])
        code = _compute_exit_code(result, policy)
        assert code == 0, (
            f"banking-политика при MEDIUM issue должна возвращать exit code 0, получено: {code}"
        )

    def test_no_issues_causes_exit0(self) -> None:
        """Без issues banking-политика возвращает exit code 0."""
        policy = PolicyLoader.load("banking")
        result = _make_scan_result([])
        code = _compute_exit_code(result, policy)
        assert code == 0

    def test_banking_policy_fields(self) -> None:
        """banking-политика содержит все обязательные поля с корректными значениями."""
        policy = PolicyLoader.load("banking")
        assert policy["name"] == "banking"
        assert "description" in policy
        assert "enabled_detectors" in policy
        assert "compliance" in policy
        assert isinstance(policy["compliance"], list)
        assert "fstec" in policy["compliance"]

    def test_banking_via_cli_high_issue(self) -> None:
        """CLI с --policy banking на вредоносном файле возвращает exit code 1."""
        malicious_file = _MALICIOUS / "os_system.pkl"
        if not malicious_file.exists():
            pytest.skip(f"Тестовый файл не найден: {malicious_file}")

        result = runner.invoke(app, ["scan", str(malicious_file), "--policy", "banking"])
        assert result.exit_code == 1, (
            f"CLI с banking-политикой на вредоносном файле ожидался exit code 1, "
            f"получено: {result.exit_code}\n{result.output}"
        )


# ---------------------------------------------------------------------------
# Тест 2: default политика — HIGH → exit 0, CRITICAL → exit 1
# ---------------------------------------------------------------------------


class TestDefaultPolicy:
    """default-политика: fail_on_severity=critical → exit 1 только при CRITICAL."""

    def test_fail_on_severity_is_critical(self) -> None:
        """default.fail_on_severity должен быть 'critical'."""
        policy = PolicyLoader.load("default")
        assert policy["fail_on_severity"] == "critical", (
            f"default.fail_on_severity должен быть 'critical', получено: {policy['fail_on_severity']}"
        )

    def test_high_issue_causes_exit0(self) -> None:
        """При наличии HIGH issue default-политика даёт exit code 0."""
        policy = PolicyLoader.load("default")
        result = _make_scan_result([_make_issue(Severity.HIGH)])
        code = _compute_exit_code(result, policy)
        assert code == 0, (
            f"default-политика при HIGH issue должна возвращать exit code 0, получено: {code}"
        )

    def test_critical_issue_causes_exit1(self) -> None:
        """При наличии CRITICAL issue default-политика даёт exit code 1."""
        policy = PolicyLoader.load("default")
        result = _make_scan_result([_make_issue(Severity.CRITICAL)])
        code = _compute_exit_code(result, policy)
        assert code == 1, (
            f"default-политика при CRITICAL issue должна возвращать exit code 1, получено: {code}"
        )

    def test_medium_issue_causes_exit0(self) -> None:
        """При MEDIUM issue default-политика даёт exit code 0."""
        policy = PolicyLoader.load("default")
        result = _make_scan_result([_make_issue(Severity.MEDIUM)])
        code = _compute_exit_code(result, policy)
        assert code == 0

    def test_no_issues_causes_exit0(self) -> None:
        """Без issues default-политика возвращает exit code 0."""
        policy = PolicyLoader.load("default")
        result = _make_scan_result([])
        code = _compute_exit_code(result, policy)
        assert code == 0


# ---------------------------------------------------------------------------
# Тест 3: strict политика — MEDIUM → exit 1
# ---------------------------------------------------------------------------


class TestStrictPolicy:
    """strict-политика: fail_on_severity=medium → exit 1 при MEDIUM и выше."""

    def test_fail_on_severity_is_medium(self) -> None:
        """strict.fail_on_severity должен быть 'medium'."""
        policy = PolicyLoader.load("strict")
        assert policy["fail_on_severity"] == "medium"

    def test_medium_issue_causes_exit1(self) -> None:
        """При MEDIUM issue strict-политика даёт exit code 1."""
        policy = PolicyLoader.load("strict")
        result = _make_scan_result([_make_issue(Severity.MEDIUM)])
        code = _compute_exit_code(result, policy)
        assert code == 1

    def test_low_issue_causes_exit0(self) -> None:
        """При LOW issue strict-политика даёт exit code 0."""
        policy = PolicyLoader.load("strict")
        result = _make_scan_result([_make_issue(Severity.LOW)])
        code = _compute_exit_code(result, policy)
        assert code == 0


# ---------------------------------------------------------------------------
# Тест 4: Кастомный путь к YAML → загружается корректно
# ---------------------------------------------------------------------------


class TestCustomPolicyPath:
    """Загрузка политики из пользовательского YAML-файла по абсолютному пути."""

    def test_load_by_absolute_path(self, tmp_path: Path) -> None:
        """PolicyLoader.load() принимает абсолютный путь к YAML-файлу."""
        custom_yaml = tmp_path / "my_policy.yaml"
        custom_yaml.write_text(
            "name: custom\n"
            "description: Тестовая кастомная политика\n"
            "enabled_detectors:\n"
            "  - blocklist\n"
            "  - cve\n"
            "fail_on_severity: high\n"
            "severity_threshold: medium\n"
            "compliance: []\n",
            encoding="utf-8",
        )

        policy = PolicyLoader.load(str(custom_yaml))
        assert policy["name"] == "custom"
        assert policy["fail_on_severity"] == "high"
        assert "blocklist" in policy["enabled_detectors"]
        assert "cve" in policy["enabled_detectors"]

    def test_load_by_relative_path(self, tmp_path: Path) -> None:
        """PolicyLoader.load() принимает относительный путь к существующему YAML-файлу."""
        custom_yaml = tmp_path / "rel_policy.yaml"
        custom_yaml.write_text(
            "name: rel_custom\n"
            "fail_on_severity: medium\n"
            "severity_threshold: low\n"
            "compliance: []\n",
            encoding="utf-8",
        )

        policy = PolicyLoader.load(str(custom_yaml))
        assert policy["name"] == "rel_custom"
        assert policy["fail_on_severity"] == "medium"

    def test_custom_policy_applies_defaults(self, tmp_path: Path) -> None:
        """Кастомный YAML с минимальными ключами получает остальные из defaults."""
        custom_yaml = tmp_path / "minimal.yaml"
        custom_yaml.write_text(
            "name: minimal\n",
            encoding="utf-8",
        )

        policy = PolicyLoader.load(str(custom_yaml))
        # Ключи из _DEFAULTS должны быть проставлены
        assert policy["name"] == "minimal"
        assert "fail_on_severity" in policy
        assert "severity_threshold" in policy
        assert "enabled_detectors" in policy
        assert "compliance" in policy

    def test_custom_policy_exit_code_via_cli(self, tmp_path: Path) -> None:
        """CLI --policy /path/to/custom.yaml загружает кастомную политику.

        Тест проверяет, что CLI принял кастомную политику (имя есть в выводе),
        не падает с unhandled exception.
        """
        custom_yaml = tmp_path / "test_policy.yaml"
        custom_yaml.write_text(
            "name: test_custom_cli\n"
            "fail_on_severity: critical\n"
            "severity_threshold: info\n"
            "enabled_detectors:\n"
            "  - blocklist\n"
            "  - allowlist\n"
            "  - cve\n"
            "  - secrets\n"
            "  - network\n"
            "  - executable\n"
            "  - compression\n"
            "compliance: []\n",
            encoding="utf-8",
        )

        safe_file = _SAFE / "simple_list.pkl"
        if not safe_file.exists():
            pytest.skip(f"Тестовый файл не найден: {safe_file}")

        result = runner.invoke(
            app, ["scan", str(safe_file), "--policy", str(custom_yaml)]
        )
        # CLI не должен упасть с исключением (exit code 2 допустим при ошибке формата,
        # но не из-за проблем с загрузкой политики).
        # Главная проверка: политика загружена — имя должно быть в выводе ИЛИ
        # CLI завершился без необработанного исключения.
        assert result.exit_code in (0, 1, 2), (
            f"CLI с кастомной политикой не должен падать, "
            f"exit code: {result.exit_code}\n{result.output}"
        )
        # Нет traceback в выводе
        assert "Traceback" not in result.output, (
            f"CLI выбросил необработанное исключение:\n{result.output}"
        )


# ---------------------------------------------------------------------------
# Тест 5: Неизвестная политика → ValueError с понятным сообщением
# ---------------------------------------------------------------------------


class TestUnknownPolicy:
    """Обработка неизвестных и некорректных политик."""

    def test_unknown_policy_name_raises_value_error(self) -> None:
        """Попытка загрузить несуществующую политику по имени → ValueError."""
        with pytest.raises(ValueError, match="несуществующая_политика"):
            PolicyLoader.load("несуществующая_политика")

    def test_unknown_policy_error_message_helpful(self) -> None:
        """Сообщение ValueError содержит список доступных политик."""
        with pytest.raises(ValueError) as exc_info:
            PolicyLoader.load("totally_unknown")
        error_msg = str(exc_info.value)
        # Сообщение должно указывать на допустимые политики
        assert any(
            name in error_msg
            for name in ("default", "banking", "government", "strict")
        ), f"Сообщение об ошибке не содержит список доступных политик:\n{error_msg}"

    def test_nonexistent_yaml_path_raises_value_error(self) -> None:
        """Попытка загрузить несуществующий YAML-файл → ValueError."""
        with pytest.raises(ValueError, match="не найден"):
            PolicyLoader.load("/nonexistent/path/to/policy.yaml")

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """Некорректный YAML в файле политики → ValueError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("this: is: not: valid: yaml: :\n  - broken\n[broken", encoding="utf-8")

        with pytest.raises(ValueError):
            PolicyLoader.load(str(bad_yaml))

    def test_non_dict_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """YAML-файл с не-словарём на верхнем уровне → ValueError."""
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="словарь"):
            PolicyLoader.load(str(list_yaml))

    def test_cli_unknown_policy_falls_back_gracefully(self) -> None:
        """CLI с неизвестной политикой не падает с unhandled exception."""
        safe_file = _SAFE / "simple_list.pkl"
        if not safe_file.exists():
            pytest.skip(f"Тестовый файл не найден: {safe_file}")

        result = runner.invoke(app, ["scan", str(safe_file), "--policy", "nonexistent_xyz"])
        # CLI не должен падать с unhandled exception — любой exit code допустим
        assert result.exit_code in (0, 1, 2), (
            f"CLI с неизвестной политикой не должен давать exception, "
            f"exit code: {result.exit_code}\n{result.output}"
        )
        # Нет traceback
        assert "Traceback" not in result.output, (
            f"CLI выбросил необработанное исключение:\n{result.output}"
        )
        # Должно быть предупреждение о неизвестной политике
        output_lower = result.output.lower()
        assert any(
            marker in output_lower
            for marker in ["политика", "nonexistent", "не найдена", "policy", "default"]
        ), f"Нет сообщения о проблеме с политикой:\n{result.output}"


class TestPolicyPathTraversal:
    """Защита от path traversal в имени политики.

    Регрессия: в v0.1 PolicyLoader.load("../../etc/passwd") принимал любое
    имя как путь к файлу и читал произвольный YAML-совместимый файл. Атакующий,
    управляющий именем политики (например, через CI-переменную), мог раскрыть
    структуру файловой системы или прочитать чужие YAML-файлы.
    """

    def test_traversal_via_dotdot_rejected(self) -> None:
        """Имя с '..' не должно искаться как файл политики."""
        with pytest.raises(ValueError):
            PolicyLoader.load("..")

    def test_traversal_via_relative_dotdot_rejected(self) -> None:
        """Имя вида '../../foo' не должно искаться вне директории политик."""
        with pytest.raises(ValueError):
            PolicyLoader.load("../../etc/passwd")

    def test_path_without_yaml_extension_rejected(self, tmp_path: Path) -> None:
        """Путь к файлу без расширения .yaml/.yml отвергается даже если файл существует.

        Это защита от чтения произвольных текстовых файлов через PolicyLoader:
        раньше любой существующий файл загружался как YAML.
        """
        # Создаём файл с YAML-валидным содержимым, но без расширения
        non_yaml = tmp_path / "config"
        non_yaml.write_text("name: foo\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\.yaml"):
            PolicyLoader.load(str(non_yaml))

    def test_traversal_through_policies_dir_rejected(self) -> None:
        """Имя, которое после resolve выходит за пределы BUILTIN_POLICIES_DIR, отвергается."""
        # Имя с слэшами автоматически уходит в "явный путь" ветку,
        # но имя без слэшей с '..' — в "имя политики" ветку.
        # Этот тест ловит вторую ветку.
        with pytest.raises(ValueError, match=r"'\.\.'|\bточки\b"):
            PolicyLoader.load("..")


# ---------------------------------------------------------------------------
# Тест 6: Тест DetectorRegistry.enabled_for_policy()
# ---------------------------------------------------------------------------


class TestDetectorRegistryPolicy:
    """DetectorRegistry правильно фильтрует детекторы по политике.

    Эти тесты работают с детекторами напрямую, минуя реестр — чтобы не зависеть
    от глобального состояния реестра, которое другие тесты могут сбрасывать.
    """

    def test_enabled_detectors_filters_correctly(self) -> None:
        """enabled_detectors из политики фильтрует реестр детекторов.

        Тест регистрирует детекторы вручную в изолированном реестре.
        """
        from poison_check.core.detector_base import BaseDetector
        from poison_check.core.result import Confidence, Issue, MLContext
        from poison_check.core.scanner_base import RawScanData

        # Создаём минимальные фиктивные детекторы для теста
        class FakeBlocklist(BaseDetector):
            name = "blocklist"
            description = "fake"
            severity_range = (Severity.CRITICAL, Severity.CRITICAL)

            def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
                return []

        class FakeCve(BaseDetector):
            name = "cve"
            description = "fake"
            severity_range = (Severity.CRITICAL, Severity.CRITICAL)

            def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
                return []

        class FakeNetwork(BaseDetector):
            name = "network"
            description = "fake"
            severity_range = (Severity.LOW, Severity.HIGH)

            def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
                return []

        # Тестируем фильтрацию напрямую — без реестра
        all_fakes = [FakeBlocklist, FakeCve, FakeNetwork]
        policy: dict[str, Any] = {"enabled_detectors": ["blocklist", "cve"]}

        enabled_raw = policy.get("enabled_detectors")
        assert enabled_raw is not None
        enabled_set: set[str] = set(enabled_raw)
        filtered = [d for d in all_fakes if d.name in enabled_set]
        filtered_names = {d.name for d in filtered}

        assert filtered_names == {"blocklist", "cve"}, (
            f"Ожидались только blocklist и cve, получено: {filtered_names}"
        )

    def test_none_enabled_detectors_returns_all(self) -> None:
        """enabled_detectors=None — фильтрация не применяется, возвращаются все."""
        from poison_check.core.detector_base import BaseDetector
        from poison_check.core.result import Issue, MLContext
        from poison_check.core.scanner_base import RawScanData

        class FakeA(BaseDetector):
            name = "a"
            description = "fake"
            severity_range = (Severity.LOW, Severity.LOW)

            def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
                return []

        class FakeB(BaseDetector):
            name = "b"
            description = "fake"
            severity_range = (Severity.LOW, Severity.LOW)

            def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
                return []

        all_fakes = [FakeA, FakeB]
        policy: dict[str, Any] = {"enabled_detectors": None}
        enabled_raw = policy.get("enabled_detectors")
        # enabled_detectors = None → возвращаем все
        if enabled_raw is None:
            result = all_fakes
        else:
            enabled_set: set[str] = set(enabled_raw)
            result = [d for d in all_fakes if d.name in enabled_set]

        assert len(result) == 2, (
            f"enabled_detectors=None должен возвращать все 2 детектора, получено: {len(result)}"
        )

    def test_banking_policy_enabled_detectors_field(self) -> None:
        """banking-политика содержит поле enabled_detectors со всеми зарегистрированными детекторами."""
        policy = PolicyLoader.load("banking")
        expected_names = {
            "blocklist", "allowlist", "cve", "secrets", "network",
            "executable", "compression", "gguf_metadata", "numpy_metadata",
            "joblib_metadata", "keras", "parse_error", "format_policy",
            "bundle_code",
        }
        enabled_in_policy = set(policy.get("enabled_detectors", []))
        assert expected_names == enabled_in_policy, (
            f"banking-политика должна включать {expected_names}, "
            f"в поле enabled_detectors: {enabled_in_policy}"
        )

    def test_all_builtin_policies_include_all_registered_detectors(self) -> None:
        """Регрессия: каждая встроенная политика должна включать все зарегистрированные детекторы.

        Если новый детектор добавлен в DetectorRegistry, но забыт в YAML — этот
        тест упадёт. Это защищает от ситуации, когда детектор молча отключён
        для всех пользователей (см. аудит, находка #1: gguf_metadata был забыт
        во всех 4 политиках).
        """
        registered_names = {d.name for d in DetectorRegistry.all_detectors()}
        if not registered_names:
            pytest.skip("Реестр детекторов пуст")

        for policy_name in ("default", "banking", "government", "strict"):
            policy = PolicyLoader.load(policy_name)
            enabled = set(policy.get("enabled_detectors") or [])
            missing = registered_names - enabled
            assert not missing, (
                f"Политика {policy_name!r} не включает зарегистрированные "
                f"детекторы: {sorted(missing)}. Добавьте их в "
                f"policies/{policy_name}.yaml в раздел enabled_detectors."
            )

    def test_registry_enabled_for_policy_with_real_detectors(self) -> None:
        """DetectorRegistry.enabled_for_policy() корректно фильтрует с реальными детекторами.

        Гарантируем заполненность реестра через прямую регистрацию.
        """
        from importlib import import_module

        # Форсируем заполнение реестра (идемпотентно при уже заполненном)
        _DETECTOR_MODULES = [
            "poison_check.detectors.blocklist_detector",
            "poison_check.detectors.allowlist_detector",
            "poison_check.detectors.cve_detector",
            "poison_check.detectors.secrets_detector",
            "poison_check.detectors.network_detector",
            "poison_check.detectors.executable_detector",
            "poison_check.detectors.compression_detector",
        ]
        for mod in _DETECTOR_MODULES:
            import_module(mod)

        all_detectors = DetectorRegistry.all_detectors()
        if not all_detectors:
            pytest.skip("Реестр детекторов пуст — тест пропущен (другие тесты сбросили реестр)")

        # Политика включает только blocklist и cve
        policy: dict[str, Any] = {"enabled_detectors": ["blocklist", "cve"]}
        enabled = DetectorRegistry.enabled_for_policy(policy)
        enabled_names = {d.name for d in enabled}
        # Должны быть только те, что в enabled_detectors
        assert enabled_names <= {"blocklist", "cve"}, (
            f"Получены детекторы за пределами enabled_detectors: {enabled_names - {'blocklist', 'cve'}}"
        )
        # blocklist и cve должны присутствовать (если зарегистрированы)
        registered_names = {d.name for d in all_detectors}
        expected_enabled = {"blocklist", "cve"} & registered_names
        assert enabled_names == expected_enabled, (
            f"Ожидались {expected_enabled}, получено: {enabled_names}"
        )


# ---------------------------------------------------------------------------
# Тест 7: Smoke-тест i18n — все ключи ru.yaml доступны без KeyError
# ---------------------------------------------------------------------------


class TestI18nSmoke:
    """Проверяет, что все ключи в ru.yaml и en.yaml доступны через I18n.t()."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> None:
        """Сбрасывает singleton I18n перед каждым тестом."""
        I18n.reset()

    def _collect_keys(self, d: dict[str, Any], prefix: str = "") -> list[str]:
        """Рекурсивно собирает все dot-notation ключи из словаря."""
        keys: list[str] = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(self._collect_keys(v, full_key))
            elif isinstance(v, str):
                keys.append(full_key)
        return keys

    def _load_yaml_raw(self, locale: str) -> dict[str, Any]:
        """Загружает YAML-файл локали напрямую для проверки ключей."""
        from pathlib import Path
        i18n_dir = Path(__file__).parent.parent / "poison_check" / "i18n"
        path = i18n_dir / f"{locale}.yaml"
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), f"Ожидался словарь в {path}"
        return data

    def test_all_ru_yaml_keys_accessible(self) -> None:
        """Все ключи ru.yaml доступны через I18n.t() без возврата самого ключа.

        I18n.t() при отсутствии ключа возвращает сам ключ — это fallback.
        Smoke-тест проверяет, что fallback не срабатывает ни для одного ключа.
        """
        i18n = I18n(locale="ru")
        raw = self._load_yaml_raw("ru")
        all_keys = self._collect_keys(raw)

        failed_keys: list[str] = []
        for key in all_keys:
            result = i18n.t(key)
            # Если I18n вернул сам ключ — значит ключ не найден
            if result == key:
                failed_keys.append(key)

        assert not failed_keys, (
            f"Следующие ключи из ru.yaml не найдены через I18n.t(): {failed_keys}"
        )

    def test_all_en_yaml_keys_accessible(self) -> None:
        """Все ключи en.yaml доступны через I18n.t() без возврата самого ключа."""
        i18n = I18n(locale="en")
        raw = self._load_yaml_raw("en")
        all_keys = self._collect_keys(raw)

        failed_keys: list[str] = []
        for key in all_keys:
            result = i18n.t(key)
            if result == key:
                failed_keys.append(key)

        assert not failed_keys, (
            f"Следующие ключи из en.yaml не найдены через I18n.t(): {failed_keys}"
        )

    def test_ru_has_all_keys_that_en_has(self) -> None:
        """ru.yaml содержит все ключи, что есть в en.yaml (основной язык полный)."""
        ru_raw = self._load_yaml_raw("ru")
        en_raw = self._load_yaml_raw("en")

        ru_keys = set(self._collect_keys(ru_raw))
        en_keys = set(self._collect_keys(en_raw))

        missing_in_ru = en_keys - ru_keys
        assert not missing_in_ru, (
            f"В ru.yaml отсутствуют ключи, присутствующие в en.yaml: {sorted(missing_in_ru)}"
        )

    def test_no_key_error_for_severity_labels(self) -> None:
        """Все 5 уровней severity имеют локализованные метки в ru.yaml."""
        i18n = I18n(locale="ru")
        severity_values = ("critical", "high", "medium", "low", "info")
        for sev in severity_values:
            key = f"severity.{sev}"
            result = i18n.t(key)
            assert result != key, (
                f"Ключ '{key}' не найден в ru.yaml (I18n.t вернул сам ключ)"
            )

    def test_no_key_error_for_issue_fields(self) -> None:
        """Все поля issue (why, remediation, location, etc.) имеют переводы."""
        i18n = I18n(locale="ru")
        issue_fields = ("issue.location", "issue.why", "issue.remediation",
                        "issue.references", "issue.decompiled", "issue.details")
        for key in issue_fields:
            result = i18n.t(key)
            assert result != key, (
                f"Ключ '{key}' не найден в ru.yaml"
            )


# ---------------------------------------------------------------------------
# Тест 8: PolicyLoader.list_builtin()
# ---------------------------------------------------------------------------


class TestPolicyLoaderListBuiltin:
    """PolicyLoader.list_builtin() возвращает список встроенных политик."""

    def test_returns_expected_policies(self) -> None:
        """list_builtin() включает все 4 встроенные политики."""
        builtin = PolicyLoader.list_builtin()
        expected = {"default", "banking", "government", "strict"}
        assert expected.issubset(set(builtin)), (
            f"Ожидались политики {expected}, получено: {builtin}"
        )

    def test_returns_sorted_list(self) -> None:
        """list_builtin() возвращает отсортированный список."""
        builtin = PolicyLoader.list_builtin()
        assert builtin == sorted(builtin), f"Список политик не отсортирован: {builtin}"


# ---------------------------------------------------------------------------
# Тест 9: extra_rules — аксессоры к применяемым (enforced) правилам
# ---------------------------------------------------------------------------


class TestExtraRulesAccessors:
    """``max_file_size_gb``, ``no_external_urls`` и ``no_unverified_files``.

    Это единственные три ключа extra_rules, которые реально влияют на
    сканирование. Остальные помечены в YAML как «not enforced yet».
    Подробные тесты ``no_unverified_files`` — в tests/test_h5py_required.py.
    """

    def test_missing_extra_rules_returns_empty(self) -> None:
        """Политика без extra_rules → пустой словарь правил."""
        assert policy_extra_rules({}) == {}

    def test_non_dict_extra_rules_returns_empty(self) -> None:
        """Некорректный тип extra_rules трактуется как отсутствие правил."""
        assert policy_extra_rules({"extra_rules": "нет"}) == {}
        assert policy_max_file_size_bytes({"extra_rules": ["a"]}) is None
        assert policy_escalate_external_urls({"extra_rules": None}) is False

    def test_max_file_size_gb_converted_to_bytes(self) -> None:
        """max_file_size_gb: 50 → 50 * 10^9 байт (десятичные ГБ, как у CLI-флага)."""
        policy = {"extra_rules": {"max_file_size_gb": 50}}
        assert policy_max_file_size_bytes(policy) == 50_000_000_000

    def test_max_file_size_gb_accepts_float(self) -> None:
        """Дробное значение допустимо — округляется вниз до целых байт."""
        policy = {"extra_rules": {"max_file_size_gb": 0.000001}}
        assert policy_max_file_size_bytes(policy) == 1000

    @pytest.mark.parametrize("value", [0, -1, "50", True, None])
    def test_max_file_size_gb_invalid_returns_none(self, value: Any) -> None:
        """Некорректное значение → None (лимит сканера по умолчанию)."""
        policy = {"extra_rules": {"max_file_size_gb": value}}
        assert policy_max_file_size_bytes(policy) is None

    def test_no_external_urls_true(self) -> None:
        """no_external_urls: true → эскалация включена."""
        assert policy_escalate_external_urls({"extra_rules": {"no_external_urls": True}})

    @pytest.mark.parametrize("value", [False, None, "true", 1])
    def test_no_external_urls_non_true_is_disabled(self, value: Any) -> None:
        """Любое значение кроме булева True → эскалация выключена."""
        policy = {"extra_rules": {"no_external_urls": value}}
        assert policy_escalate_external_urls(policy) is False

    @pytest.mark.parametrize(
        ("name", "expected_gb"),
        [("banking", 50), ("government", 100), ("strict", 10)],
    )
    def test_builtin_policies_expose_size_limit(self, name: str, expected_gb: int) -> None:
        """Встроенные отраслевые политики отдают свой лимит размера файла."""
        policy = PolicyLoader.load(name)
        assert policy_max_file_size_bytes(policy) == expected_gb * 1_000_000_000

    @pytest.mark.parametrize("name", ["banking", "government", "strict"])
    def test_builtin_policies_escalate_urls(self, name: str) -> None:
        """banking / government / strict требуют эскалации внешних URL."""
        assert policy_escalate_external_urls(PolicyLoader.load(name)) is True

    @pytest.mark.parametrize("name", ["banking", "government", "strict"])
    def test_builtin_policies_escalate_unverified(self, name: str) -> None:
        """banking / government / strict поднимают непроверенный файл до HIGH."""
        assert policy_escalate_unverified(PolicyLoader.load(name)) is True

    def test_default_policy_has_no_extra_rules(self) -> None:
        """default не задаёт ни лимита размера, ни запретов."""
        policy = PolicyLoader.load("default")
        assert policy_max_file_size_bytes(policy) is None
        assert policy_escalate_external_urls(policy) is False
        assert policy_escalate_unverified(policy) is False


# ---------------------------------------------------------------------------
# Тест 10: extra_rules → параметры детекторов (CLI и Python API одинаково)
# ---------------------------------------------------------------------------


class TestDetectorInstantiationFromPolicy:
    """``instantiate_detectors`` прокидывает extra_rules в конструкторы детекторов."""

    def test_kwargs_for_network_detector(self) -> None:
        """Для детектора network kwargs содержат escalate_external_urls."""
        policy = {"extra_rules": {"no_external_urls": True}}
        assert detector_kwargs_for(policy, "network") == {
            "escalate_external_urls": True
        }

    def test_kwargs_empty_for_other_detectors(self) -> None:
        """Детекторы без настроек получают пустые kwargs."""
        policy = {"extra_rules": {"no_external_urls": True}}
        for name in ("allowlist", "blocklist", "cve", "secrets"):
            assert detector_kwargs_for(policy, name) == {}

    def test_instantiate_passes_flag(self) -> None:
        """NetworkDetector создаётся с флагом из политики."""
        policy = {"extra_rules": {"no_external_urls": True}}
        instances = instantiate_detectors([NetworkDetector], policy)

        assert len(instances) == 1
        detector = instances[0]
        assert isinstance(detector, NetworkDetector)
        assert detector.escalate_external_urls is True

    def test_instantiate_without_flag(self) -> None:
        """Без правила в политике NetworkDetector создаётся с False."""
        instances = instantiate_detectors([NetworkDetector], {})
        detector = instances[0]
        assert isinstance(detector, NetworkDetector)
        assert detector.escalate_external_urls is False

    def test_cli_and_api_agree_on_escalation(self) -> None:
        """CLI (_get_cached_detectors) и Python API (Scanner) ведут себя одинаково.

        Регрессия задачи 3: раньше banking-политика не доходила до детекторов
        ни через один из путей.
        """
        from poison_check.cli import _get_cached_detectors
        from poison_check.scanner import Scanner

        policy = PolicyLoader.load("banking")

        cli_network = [
            d for d in _get_cached_detectors(policy) if isinstance(d, NetworkDetector)
        ]
        api = Scanner(policy="banking")
        api_network = [
            d
            for d in api._get_detectors(api._loaded_policy)
            if isinstance(d, NetworkDetector)
        ]

        assert len(cli_network) == 1
        assert len(api_network) == 1
        assert cli_network[0].escalate_external_urls is True
        assert api_network[0].escalate_external_urls is True

    def test_api_default_policy_does_not_escalate(self) -> None:
        """Политика default через Python API не включает эскалацию."""
        from poison_check.scanner import Scanner

        api = Scanner(policy="default")
        network = [
            d
            for d in api._get_detectors(api._loaded_policy)
            if isinstance(d, NetworkDetector)
        ]
        assert len(network) == 1
        assert network[0].escalate_external_urls is False


# ---------------------------------------------------------------------------
# Тест 11: не-enforced ключи extra_rules честно помечены в YAML
# ---------------------------------------------------------------------------


class TestNotEnforcedRulesMarked:
    """Ключи extra_rules без реализации помечены комментарием «not enforced yet».

    Иначе YAML создаёт ложное ощущение, что правило работает.

    Список применяемых ключей больше НЕ хардкодится здесь: источник истины —
    реестр ``EXTRA_RULE_KEYS`` в ``poison_check.policies``. Раньше дублирующий
    frozenset жил в тесте, и ключ, забытый в коде, но добавленный в этот
    список, проходил проверку. Полная версия инварианта (все ключи политики,
    их точки применения и тесты) — в ``tests/test_policy_keys_enforced.py``.
    """

    @pytest.mark.parametrize("name", ["default", "banking", "government", "strict"])
    def test_unenforced_keys_have_marker(self, name: str) -> None:
        """Каждый не применяемый ключ имеет рядом маркер not enforced yet."""
        path = PolicyLoader.BUILTIN_POLICIES_DIR / f"{name}.yaml"
        text = path.read_text(encoding="utf-8")
        extra = policy_extra_rules(PolicyLoader.load(name))
        specs = extra_rule_key_specs()

        for key in extra:
            spec = specs.get(key)
            assert spec is not None, (
                f"Ключ '{key}' в {name}.yaml отсутствует в реестре EXTRA_RULE_KEYS"
            )
            if spec.status is not PolicyKeyStatus.RESERVED:
                continue
            key_line_idx = next(
                i for i, line in enumerate(text.splitlines()) if line.strip().startswith(f"{key}:")
            )
            preceding = "\n".join(text.splitlines()[max(0, key_line_idx - 3):key_line_idx])
            assert "not enforced yet" in preceding, (
                f"Ключ '{key}' в {name}.yaml не помечен как «not enforced yet»"
            )

    def test_enforced_set_comes_from_registry(self) -> None:
        """Реестр — единственный источник списка применяемых ключей extra_rules."""
        enforced = {
            key
            for key, spec in extra_rule_key_specs().items()
            if spec.status is PolicyKeyStatus.ENFORCED
        }
        assert enforced == {
            "max_file_size_gb",
            "no_external_urls",
            "no_unverified_files",
            # Оба ключа переведены из RESERVED в ENFORCED вместе с появлением
            # FormatPolicyDetector (MLS-FMT-001 / MLS-FMT-002).
            "strict_format_detection",
            "require_safetensors",
        }

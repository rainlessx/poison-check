"""Тесты перевода h5py в обязательные зависимости.

Контекст дефекта: в окружении без h5py файлы .h5/.hdf5 получали MLS-KERAS-003
уровня INFO вместо MLS-KERAS-001 (Lambda-RCE, CVE-2025-1550). Вредоносная
HDF5-модель проходила CI-гейт «fail при HIGH+» как чистая — обход сканера через
свойство окружения, а не через содержимое файла.

Проверяем четыре контура защиты:
1. Метаданные пакета — h5py в [project].dependencies, extra [keras] пуст.
2. doctor — отсутствие h5py это ошибка (exit 1) с указанием последствия.
3. MLS-KERAS-003 — MEDIUM (HIGH в строгих политиках), семантика «сломанная
   установка», а не «опциональная библиотека».
4. Инварианты — сканер не падает и не теряет файл без h5py; с h5py вредоносный
   .hdf5 даёт MLS-KERAS-001 CRITICAL без MLS-KERAS-003.

Отсутствие h5py моделируется записью ``sys.modules["h5py"] = None``: и
``import h5py``, и ``importlib.util.find_spec("h5py")`` ведут себя так же, как
при непоставленной библиотеке, поэтому тесты работают в любом окружении.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from poison_check.cli import app
from poison_check.core.result import Confidence, MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.keras_detector import KerasThreatDetector
from poison_check.policies import (
    PolicyLoader,
    detector_kwargs_for,
    policy_escalate_unverified,
)
from poison_check.scanner import Scanner

runner = CliRunner()

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_CTX = MLContext(framework="tensorflow", confidence=0.0, detected_patterns=[])
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


def _hide_h5py(monkeypatch: pytest.MonkeyPatch) -> None:
    """Делает h5py неимпортируемым на время теста (сломанная установка)."""
    monkeypatch.setitem(sys.modules, "h5py", None)


def _unverified_raw() -> RawScanData:
    """RawScanData, который KerasScanner оставляет при неимпортируемом h5py."""
    return RawScanData(
        file_path=Path("model.h5"),
        file_hash={},
        file_size=0,
        scanner_name="keras",
        metadata={"keras_format": "h5", "h5py_available": "false"},
    )


def _lambda_model_config(layer_name: str = "evil") -> dict[str, object]:
    """Собирает model_config со слоем Lambda вручную (без keras.save)."""
    return {
        "class_name": "Sequential",
        "config": {
            "name": "seq",
            "layers": [
                {
                    "class_name": "Lambda",
                    "config": {
                        "name": layer_name,
                        "function": {
                            "class_name": "__lambda__",
                            "config": {"code": "PAYLOAD"},
                        },
                    },
                }
            ],
        },
    }


def _write_lambda_hdf5(path: Path) -> None:
    """Пишет .hdf5 с Lambda-слоем в root-атрибуте model_config."""
    import h5py  # noqa: PLC0415

    with h5py.File(path, "w") as hf:
        hf.attrs["keras_version"] = "2.15.0"
        hf.attrs["backend"] = "tensorflow"
        hf.attrs["model_config"] = json.dumps(_lambda_model_config())


def _normalized(text: str) -> str:
    """Схлопывает переносы rich-вывода, чтобы искать фразы целиком."""
    return " ".join(text.split())


def _parse_deps_fallback(text: str, section: str = "dependencies") -> list[str]:
    """Достаёт список зависимостей из pyproject без tomllib (Python 3.10).

    Ищет ``<section> = [ ... ]`` и собирает строки в кавычках, игнорируя
    комментарии. Достаточно для нашего плоского pyproject.toml.
    """
    match = re.search(rf"^{re.escape(section)} = \[(.*?)^\]", text, re.S | re.M)
    if match is None:
        return []
    body = "\n".join(
        line.split("#", 1)[0] for line in match.group(1).splitlines()
    )
    return re.findall(r'"([^"]+)"', body)


def _project_dependencies() -> list[str]:
    """Читает [project].dependencies из pyproject.toml программно."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib  # noqa: PLC0415 — на 3.10 отсутствует
    except ModuleNotFoundError:
        return _parse_deps_fallback(text)
    data = tomllib.loads(text)
    deps: list[str] = data["project"]["dependencies"]
    return deps


def _optional_dependencies() -> dict[str, list[str]]:
    """Читает [project.optional-dependencies] из pyproject.toml программно."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib  # noqa: PLC0415
    except ModuleNotFoundError:
        return {"keras": _parse_deps_fallback(text, "keras")}
    data = tomllib.loads(text)
    optional: dict[str, list[str]] = data["project"]["optional-dependencies"]
    return optional


# ---------------------------------------------------------------------------
# 1. Метаданные пакета
# ---------------------------------------------------------------------------


class TestPackageMetadata:
    """h5py — обязательная зависимость, а не extras."""

    def test_h5py_in_required_dependencies(self) -> None:
        """h5py присутствует в [project].dependencies."""
        deps = _project_dependencies()
        assert any(
            d.replace(" ", "").startswith("h5py") for d in deps
        ), f"h5py не найден в обязательных зависимостях: {deps}"

    def test_h5py_has_lower_bound_without_upper_pin(self) -> None:
        """Задан нижний bound (колёса для 3.10-3.12) и нет верхнего пина."""
        spec = next(
            d for d in _project_dependencies() if d.replace(" ", "").startswith("h5py")
        )
        assert ">=" in spec, f"У h5py нет нижней границы версии: {spec!r}"
        assert "<" not in spec, f"Верхний пин без доказанной несовместимости: {spec!r}"

    def test_keras_extra_is_noop_alias(self) -> None:
        """Extra [keras] сохранён пустым — pip install poison-check[keras] не ломается."""
        optional = _optional_dependencies()
        assert "keras" in optional, "Extra [keras] удалён — старые requirements сломаются"
        assert optional["keras"] == [], (
            "Extra [keras] должен быть пустым алиасом: h5py живёт в обязательных "
            f"зависимостях, а не здесь — получено {optional['keras']}"
        )

    def test_h5py_not_only_in_extra(self) -> None:
        """h5py не «спрятан» в каком-либо extra вместо обязательных зависимостей."""
        for name, deps in _optional_dependencies().items():
            assert not any("h5py" in d for d in deps), (
                f"h5py объявлен в extra [{name}] — он обязателен и должен быть "
                "только в [project].dependencies"
            )

    def test_fallback_parser_matches_tomllib(self) -> None:
        """Резервный парсер (путь Python 3.10) даёт тот же список, что tomllib."""
        text = _PYPROJECT.read_text(encoding="utf-8")
        assert _parse_deps_fallback(text) == _project_dependencies()


# ---------------------------------------------------------------------------
# 2. doctor
# ---------------------------------------------------------------------------


class TestDoctorH5py:
    """h5py проверяется в секции ОБЯЗАТЕЛЬНЫХ зависимостей doctor."""

    def test_doctor_ok_when_h5py_present(self) -> None:
        """При установленном h5py проверка проходит, exit code 0."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, result.stdout
        assert "h5py" in result.stdout

    def test_doctor_fails_without_h5py(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Отсутствие обязательного h5py → ошибка (exit 1), а не предупреждение."""
        _hide_h5py(monkeypatch)
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1, result.stdout
        out = _normalized(result.stdout)
        assert "ОТСУТСТВУЕТ (обязательно)" in out
        assert "Обнаружены проблемы" in out

    def test_doctor_names_concrete_consequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """В тексте указано, что именно перестаёт проверяться, и команда установки."""
        _hide_h5py(monkeypatch)
        out = _normalized(runner.invoke(app, ["doctor"]).stdout)
        assert ".h5/.hdf5" in out
        assert "Lambda-RCE" in out
        assert "CVE-2025-1550" in out
        assert "pip install h5py" in out

    def test_doctor_ok_without_optional_pdf_deps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Опциональные зависимости на статус doctor не влияют (контраст с h5py)."""
        monkeypatch.setitem(sys.modules, "weasyprint", None)
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, result.stdout


# ---------------------------------------------------------------------------
# 3. MLS-KERAS-003: severity и семантика
# ---------------------------------------------------------------------------


class TestUnverifiedIssueSeverity:
    """Правило описывает аварию окружения, а не штатную опциональность."""

    def test_severity_is_at_least_medium(self) -> None:
        """Непроверенный файл — не INFO: минимум MEDIUM."""
        issue = KerasThreatDetector().analyze(_unverified_raw(), _CTX)[0]
        assert issue.code == "MLS-KERAS-003"
        assert issue.severity >= Severity.MEDIUM
        assert issue.severity is Severity.MEDIUM
        assert issue.confidence is Confidence.CERTAIN

    def test_message_says_file_not_verified(self) -> None:
        """Сообщение говорит о непроверенном файле, а не о «нет библиотеки»."""
        issue = KerasThreatDetector().analyze(_unverified_raw(), _CTX)[0]
        assert "НЕ ПРОВЕРЕН" in issue.message
        assert "h5py" in issue.message

    def test_why_describes_broken_installation(self) -> None:
        """why: обязательная зависимость + непроверенный файл, не «опционально»."""
        issue = KerasThreatDetector().analyze(_unverified_raw(), _CTX)[0]
        assert issue.why is not None
        assert "обязательные зависимости" in issue.why
        assert "непроверенным" in issue.why
        assert "опционал" not in issue.why.lower()

    def test_remediation_is_reinstall_not_optional_install(self) -> None:
        """remediation: переустановите пакет, файл остался непроверенным."""
        issue = KerasThreatDetector().analyze(_unverified_raw(), _CTX)[0]
        assert issue.remediation is not None
        assert "reinstall" in issue.remediation
        assert "poison-check doctor" in issue.remediation
        assert "poison-check[keras]" not in issue.remediation

    def test_details_flag_file_unverified(self) -> None:
        """details несут машиночитаемый признак «файл не проверен»."""
        issue = KerasThreatDetector().analyze(_unverified_raw(), _CTX)[0]
        assert issue.details["dependency"] == "h5py"
        assert issue.details["file_verified"] is False


class TestUnverifiedPolicyEscalation:
    """Строгие политики поднимают непроверенный файл до HIGH."""

    def test_escalated_to_high(self) -> None:
        """escalate_unverified=True → HIGH вместо MEDIUM."""
        detector = KerasThreatDetector(escalate_unverified=True)
        issue = detector.analyze(_unverified_raw(), _CTX)[0]
        assert issue.severity is Severity.HIGH

    def test_default_policy_does_not_escalate(self) -> None:
        """default: правила no_unverified_files нет → MEDIUM."""
        policy = PolicyLoader.load("default")
        assert policy_escalate_unverified(policy) is False
        assert detector_kwargs_for(policy, "keras") == {"escalate_unverified": False}

    @pytest.mark.parametrize("name", ["banking", "government", "strict"])
    def test_strict_policies_escalate(self, name: str) -> None:
        """banking / government / strict требуют эскалации непроверенного файла."""
        policy = PolicyLoader.load(name)
        assert policy_escalate_unverified(policy) is True
        assert detector_kwargs_for(policy, "keras") == {"escalate_unverified": True}

    @pytest.mark.parametrize("value", [False, None, "true", 1])
    def test_non_true_value_is_disabled(self, value: object) -> None:
        """Только булево True включает эскалацию."""
        policy = {"extra_rules": {"no_unverified_files": value}}
        assert policy_escalate_unverified(policy) is False

    def test_escalation_does_not_touch_lambda_severity(self) -> None:
        """Флаг не влияет на MLS-KERAS-001 — он и так CRITICAL."""
        raw = RawScanData(
            file_path=Path("model.keras"),
            file_hash={},
            file_size=0,
            scanner_name="keras",
            metadata={"lambda_layer_count": "1"},
        )
        issue = KerasThreatDetector(escalate_unverified=True).analyze(raw, _CTX)[0]
        assert issue.code == "MLS-KERAS-001"
        assert issue.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# 4. Инварианты: сигнал не теряется / детект работает
# ---------------------------------------------------------------------------


class TestGracefulDegradationInvariant:
    """При сломанной установке сканер не падает, а файл не исчезает из отчёта."""

    def test_scan_does_not_raise_and_keeps_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Без h5py: нет исключения, файл в отчёте, MLS-KERAS-003 эмитится."""
        _hide_h5py(monkeypatch)
        f = tmp_path / "model.h5"
        f.write_bytes(_HDF5_MAGIC + b"\x00" * 64)

        result = Scanner().scan(f)

        assert f in result.results_per_file, "Файл потерян из отчёта"
        fr = result.results_per_file[f]
        assert fr.error is None
        assert fr.scanner_name == "keras"
        issue = next(i for i in fr.issues if i.code == "MLS-KERAS-003")
        assert issue.severity is Severity.MEDIUM

    def test_worst_severity_is_not_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Непроверенный файл не выглядит чистым в агрегатах отчёта."""
        _hide_h5py(monkeypatch)
        f = tmp_path / "model.hdf5"
        f.write_bytes(_HDF5_MAGIC + b"\x00" * 64)

        result = Scanner().scan(f)

        assert result.worst_severity >= Severity.MEDIUM

    def test_strict_policy_escalates_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """banking: непроверенный файл приходит как HIGH и валит гейт fail_on=high."""
        _hide_h5py(monkeypatch)
        f = tmp_path / "model.h5"
        f.write_bytes(_HDF5_MAGIC + b"\x00" * 64)

        fr = Scanner(policy="banking").scan(f).results_per_file[f]

        issue = next(i for i in fr.issues if i.code == "MLS-KERAS-003")
        assert issue.severity is Severity.HIGH


class TestDetectionRegression:
    """С установленным h5py вредоносный .hdf5 детектится как раньше."""

    def test_malicious_hdf5_gives_critical_lambda(self, tmp_path: Path) -> None:
        """Lambda в .hdf5 → MLS-KERAS-001 CRITICAL, MLS-KERAS-003 отсутствует."""
        f = tmp_path / "malicious.hdf5"
        _write_lambda_hdf5(f)

        fr = Scanner().scan(f).results_per_file[f]

        codes = {i.code for i in fr.issues}
        assert "MLS-KERAS-001" in codes
        assert "MLS-KERAS-003" not in codes, "Файл разобран — правила о непроверке быть не должно"
        lam = next(i for i in fr.issues if i.code == "MLS-KERAS-001")
        assert lam.severity is Severity.CRITICAL
        assert any(r.id == "CVE-2025-1550" for r in lam.references)

    def test_same_file_is_silently_missed_without_h5py(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тот же файл без h5py: Lambda не найден — но файл помечен непроверенным.

        Это и есть исходный дефект. Гарантия после правки — не детект (его без
        h5py быть не может), а видимость: MLS-KERAS-003 не ниже MEDIUM.
        """
        f = tmp_path / "malicious.hdf5"
        _write_lambda_hdf5(f)
        _hide_h5py(monkeypatch)

        fr = Scanner().scan(f).results_per_file[f]

        codes = {i.code for i in fr.issues}
        assert "MLS-KERAS-001" not in codes
        assert "MLS-KERAS-003" in codes
        assert max(i.severity for i in fr.issues) >= Severity.MEDIUM

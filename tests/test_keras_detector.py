"""Тесты для KerasThreatDetector.

Детектор эмитит MLS-KERAS-* по фактам, которые KerasScanner оставляет в metadata
(``lambda_layer_count`` / ``custom_object_count`` / ``h5py_available``).

Юнит-тесты работают на синтетических RawScanData, end-to-end — через Scanner на
собранных вручную .keras-архивах (без keras.save).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import Confidence, MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.keras_detector import KerasThreatDetector
from poison_check.policies import PolicyLoader
from poison_check.scanner import Scanner

_CTX = MLContext(framework="tensorflow", confidence=0.0, detected_patterns=[])


def _raw(metadata: dict[str, str] | None, scanner_name: str = "keras") -> RawScanData:
    """Минимальный RawScanData для юнит-тестов детектора."""
    return RawScanData(
        file_path=Path("model.keras"),
        file_hash={},
        file_size=0,
        scanner_name=scanner_name,
        metadata=metadata,
    )


def _codes(issues: list) -> set[str]:
    return {i.code for i in issues}


# ---------------------------------------------------------------------------
# Контракт и регистрация
# ---------------------------------------------------------------------------


class TestKerasDetectorContract:
    def test_name(self) -> None:
        assert KerasThreatDetector.name == "keras"

    def test_severity_range(self) -> None:
        """Нижняя граница — MEDIUM: INFO детектор больше не эмитит.

        MLS-KERAS-003 (файл не проверен из-за отсутствия обязательного h5py)
        поднят с INFO до MEDIUM — см. tests/test_h5py_required.py.
        """
        assert KerasThreatDetector.severity_range == (Severity.MEDIUM, Severity.CRITICAL)

    def test_registered(self) -> None:
        assert DetectorRegistry.get("keras") is KerasThreatDetector

    def test_enabled_in_all_builtin_policies(self) -> None:
        for policy_name in ("default", "banking", "government", "strict"):
            policy = PolicyLoader.load(policy_name)
            enabled = set(policy.get("enabled_detectors") or [])
            assert "keras" in enabled, f"Политика {policy_name!r} не включает keras"


# ---------------------------------------------------------------------------
# Юнит-логика analyze()
# ---------------------------------------------------------------------------


class TestKerasDetectorAnalyze:
    def test_ignores_other_scanners(self) -> None:
        """RawScanData от другого сканера игнорируется."""
        raw = _raw({"lambda_layer_count": "3"}, scanner_name="pickle")
        assert KerasThreatDetector().analyze(raw, _CTX) == []

    def test_no_metadata_no_issues(self) -> None:
        assert KerasThreatDetector().analyze(_raw(None), _CTX) == []

    def test_clean_metadata_no_issues(self) -> None:
        raw = _raw(
            {
                "keras_format": "keras_zip",
                "lambda_layer_count": "0",
                "custom_object_count": "0",
            }
        )
        assert KerasThreatDetector().analyze(raw, _CTX) == []

    def test_lambda_emits_critical(self) -> None:
        raw = _raw(
            {
                "keras_format": "keras_zip",
                "lambda_layer_count": "2",
                "lambda_layers": "a, b",
                "custom_object_count": "0",
            }
        )
        issues = KerasThreatDetector().analyze(raw, _CTX)
        assert "MLS-KERAS-001" in _codes(issues)
        lam = next(i for i in issues if i.code == "MLS-KERAS-001")
        assert lam.severity is Severity.CRITICAL
        assert lam.confidence is Confidence.HIGH
        assert lam.details["lambda_layer_count"] == 2
        # ссылка на CVE-2025-1550 присутствует
        assert any(r.id == "CVE-2025-1550" for r in lam.references)

    def test_custom_emits_high(self) -> None:
        raw = _raw(
            {
                "keras_format": "keras_zip",
                "lambda_layer_count": "0",
                "custom_object_count": "1",
                "custom_objects": "pkg>Evil",
            }
        )
        issues = KerasThreatDetector().analyze(raw, _CTX)
        assert "MLS-KERAS-002" in _codes(issues)
        custom = next(i for i in issues if i.code == "MLS-KERAS-002")
        assert custom.severity is Severity.HIGH
        assert "pkg>Evil" in custom.message

    def test_h5py_missing_emits_medium(self) -> None:
        """h5py обязателен → непроверенный файл это MEDIUM, а не INFO."""
        raw = _raw({"keras_format": "h5", "h5py_available": "false"})
        issues = KerasThreatDetector().analyze(raw, _CTX)
        assert "MLS-KERAS-003" in _codes(issues)
        unverified = next(i for i in issues if i.code == "MLS-KERAS-003")
        assert unverified.severity is Severity.MEDIUM
        assert "h5py" in unverified.message

    def test_lambda_and_custom_together(self) -> None:
        raw = _raw(
            {
                "keras_format": "keras_zip",
                "lambda_layer_count": "1",
                "custom_object_count": "1",
                "custom_objects": "pkg>X",
            }
        )
        codes = _codes(KerasThreatDetector().analyze(raw, _CTX))
        assert {"MLS-KERAS-001", "MLS-KERAS-002"} <= codes

    def test_texts_are_russian(self) -> None:
        raw = _raw({"lambda_layer_count": "1"})
        issue = KerasThreatDetector().analyze(raw, _CTX)[0]
        assert issue.why is not None and "код" in issue.why
        assert issue.remediation is not None
        assert "owasp-ml:ml03" in issue.compliance_tags

    def test_bad_count_value_ignored(self) -> None:
        """Нечисловой счётчик не роняет детектор."""
        raw = _raw({"lambda_layer_count": "not-a-number"})
        assert KerasThreatDetector().analyze(raw, _CTX) == []


# ---------------------------------------------------------------------------
# End-to-end через Scanner
# ---------------------------------------------------------------------------


def _make_keras_zip(config: dict[str, object]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("config.json", json.dumps(config))
        zf.writestr("metadata.json", json.dumps({"keras_version": "3.5.0"}))
    return buf.getvalue()


_CLEAN = {
    "module": "keras",
    "class_name": "Sequential",
    "registered_name": None,
    "config": {
        "name": "seq",
        "layers": [
            {
                "class_name": "Dense",
                "registered_name": None,
                "config": {"name": "dense", "units": 8, "activation": "relu"},
            }
        ],
    },
}

_LAMBDA = {
    "module": "keras",
    "class_name": "Sequential",
    "registered_name": None,
    "config": {
        "name": "seq",
        "layers": [
            {
                "class_name": "Lambda",
                "registered_name": None,
                "config": {
                    "name": "evil",
                    "function": {"class_name": "__lambda__", "config": {"code": "PAYLOAD"}},
                },
            }
        ],
    },
}


class TestKerasEndToEnd:
    def test_clean_keras_no_issues(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.keras"
        f.write_bytes(_make_keras_zip(_CLEAN))
        result = Scanner().scan(f)
        fr = result.results_per_file[f]
        assert fr.error is None
        assert fr.scanner_name == "keras"
        keras_codes = [i.code for i in fr.issues if i.code.startswith("MLS-KERAS")]
        assert keras_codes == [], f"Ожидалось 0 keras-issue, получено: {keras_codes}"

    def test_lambda_keras_gives_critical(self, tmp_path: Path) -> None:
        f = tmp_path / "lam.keras"
        f.write_bytes(_make_keras_zip(_LAMBDA))
        result = Scanner().scan(f)
        fr = result.results_per_file[f]
        codes = {i.code for i in fr.issues}
        assert "MLS-KERAS-001" in codes
        worst = max(i.severity for i in fr.issues)
        assert worst is Severity.CRITICAL

    def test_lambda_via_scan_bytes(self, tmp_path: Path) -> None:
        fr = Scanner().scan_bytes(_make_keras_zip(_LAMBDA), filename="model.keras")
        assert "MLS-KERAS-001" in {i.code for i in fr.issues}

    def test_corrupted_keras_reports_error(self, tmp_path: Path) -> None:
        f = tmp_path / "corrupt.keras"
        f.write_bytes(b"PK\x03\x04" + b"\xff" * 40)
        result = Scanner().scan(f)
        fr = result.results_per_file[f]
        assert fr.error is not None

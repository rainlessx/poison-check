"""Тесты для NumpyMetadataDetector.

Детектор эмитит MLS-NPY-001 по фактам, которые NumpyScanner оставляет
в metadata (``object_dtype_detected`` / ``pickle_payload_detected``).
До выделения детектора сканер конструировал Issue сам, но RawScanData
не имеет поля issues — находка терялась и пользователь её не видел.

Все .npy-fixture строятся вручную по бинарной спецификации формата,
вредоносные pickle-payload — opcode-конструированием (не pickle.dumps).
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import Confidence, MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.numpy_metadata_detector import (
    NumpyMetadataDetector,
    _iter_numpy_entries,
    _make_object_dtype_issue,
)
from poison_check.policies import PolicyLoader
from poison_check.scanner import Scanner

_ISSUE_CODE = "MLS-NPY-001"


# ---------------------------------------------------------------------------
# Вспомогательные builder-функции (.npy собирается вручную)
# ---------------------------------------------------------------------------


def _make_npy_v1(descr: str, shape: tuple[int, ...], data: bytes) -> bytes:
    """Строит .npy v1.0 из dtype-дескриптора, shape и бинарных данных."""
    header_dict = (
        f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"
    )
    header_no_newline = header_dict.encode("latin-1")
    pad_needed = 64 - ((10 + len(header_no_newline) + 1) % 64)
    if pad_needed == 64:
        pad_needed = 0
    header_bytes = header_no_newline + b" " * pad_needed + b"\n"

    return (
        b"\x93NUMPY"
        + bytes([1, 0])
        + struct.pack("<H", len(header_bytes))
        + header_bytes
        + data
    )


def _make_float32_npy(shape: tuple[int, ...] = (4,)) -> bytes:
    """Валидный .npy с float32-массивом (нулевые данные)."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    return _make_npy_v1("<f4", shape, b"\x00" * (n_elements * 4))


def _make_object_npy_no_pickle() -> bytes:
    """Object-dtype массив без pickle-потока в данных."""
    return _make_npy_v1("|O", (3,), b"some arbitrary bytes without pickle signature")


def _make_os_system_pickle() -> bytes:
    """Вредоносный pickle-payload os.system — ручная opcode-конструкция."""
    return (
        b"\x80\x02"               # PROTO 2
        + b"cos\nsystem\n"        # GLOBAL os.system
        + b"("                    # MARK
        + b"t"                    # TUPLE → ()
        + b"R"                    # REDUCE
        + b"."                    # STOP
    )


def _make_object_npy_with_pickle() -> bytes:
    """Object-dtype массив с вредоносным pickle-payload в данных."""
    return _make_npy_v1("|O", (1,), b"\x00" * 8 + _make_os_system_pickle())


def _make_npz(*members: tuple[str, bytes]) -> bytes:
    """Строит .npz (ZIP) из пар (имя_без_расширения, npy_байты)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, content in members:
            zf.writestr(f"{name}.npy", content)
    return buf.getvalue()


def _raw(
    metadata: dict[str, str] | None,
    scanner_name: str = "numpy",
    file_path: str = "obj.npy",
    nested: list[RawScanData] | None = None,
) -> RawScanData:
    """Собирает минимальный RawScanData для unit-тестов детектора."""
    return RawScanData(
        file_path=Path(file_path),
        file_hash={},
        file_size=0,
        scanner_name=scanner_name,
        metadata=metadata,
        nested_files=nested,
    )


_CTX = MLContext(framework="numpy", confidence=1.0, detected_patterns=[])


# ---------------------------------------------------------------------------
# Тест 1: контракт детектора
# ---------------------------------------------------------------------------


class TestNumpyMetadataDetectorContract:
    """Атрибуты детектора и его регистрация."""

    def test_detector_name(self) -> None:
        """Имя детектора совпадает с именем в политиках."""
        assert NumpyMetadataDetector.name == "numpy_metadata"

    def test_severity_range(self) -> None:
        """severity_range — только MEDIUM."""
        assert NumpyMetadataDetector.severity_range == (
            Severity.MEDIUM,
            Severity.MEDIUM,
        )

    def test_registered_in_registry(self) -> None:
        """Детектор зарегистрирован в DetectorRegistry."""
        assert DetectorRegistry.get("numpy_metadata") is NumpyMetadataDetector

    def test_enabled_in_all_builtin_policies(self) -> None:
        """Детектор включён во всех встроенных политиках."""
        for policy_name in ("default", "banking", "government", "strict"):
            policy = PolicyLoader.load(policy_name)
            enabled = set(policy.get("enabled_detectors") or [])
            assert "numpy_metadata" in enabled, (
                f"Политика {policy_name!r} не включает numpy_metadata"
            )


# ---------------------------------------------------------------------------
# Тест 2: unit-логика analyze()
# ---------------------------------------------------------------------------


class TestNumpyMetadataDetectorAnalyze:
    """Детектор реагирует только на факты numpy-сканера."""

    def test_ignores_other_scanners(self) -> None:
        """RawScanData от другого сканера игнорируется."""
        raw = _raw({"object_dtype_detected": "true"}, scanner_name="pickle")
        assert NumpyMetadataDetector().analyze(raw, _CTX) == []

    def test_no_metadata_no_issues(self) -> None:
        """metadata=None → пустой список."""
        assert NumpyMetadataDetector().analyze(_raw(None), _CTX) == []

    def test_non_object_dtype_no_issues(self) -> None:
        """Числовой dtype → находок нет."""
        raw = _raw({"dtype": "f4", "shape": "[4]", "npy_version": "1.0"})
        assert NumpyMetadataDetector().analyze(raw, _CTX) == []

    def test_object_dtype_emits_single_issue(self) -> None:
        """object_dtype_detected=true → ровно один MLS-NPY-001 MEDIUM."""
        raw = _raw(
            {
                "object_dtype_detected": "true",
                "dtype": "O",
                "shape": "[3]",
                "npy_version": "1.0",
            }
        )
        issues = NumpyMetadataDetector().analyze(raw, _CTX)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.code == _ISSUE_CODE
        assert issue.severity is Severity.MEDIUM
        assert issue.confidence is Confidence.HIGH

    def test_issue_details_from_metadata(self) -> None:
        """dtype/shape/npy_version переносятся в details."""
        raw = _raw(
            {
                "object_dtype_detected": "true",
                "dtype": "O",
                "shape": "[3]",
                "npy_version": "2.0",
            }
        )
        details = NumpyMetadataDetector().analyze(raw, _CTX)[0].details
        assert details["dtype"] == "O"
        assert details["shape"] == "[3]"
        assert details["npy_version"] == "2.0"
        assert details["pickle_payload_detected"] is False

    def test_issue_texts_are_russian(self) -> None:
        """why и remediation заполнены на русском."""
        raw = _raw({"object_dtype_detected": "true", "dtype": "O"})
        issue = NumpyMetadataDetector().analyze(raw, _CTX)[0]
        assert issue.why is not None
        assert "pickle" in issue.why
        assert issue.remediation is not None
        assert "safetensors" in issue.remediation
        assert "object" in issue.message

    def test_pickle_payload_reflected(self) -> None:
        """pickle_payload_detected=true отражается в details и сообщении."""
        raw = _raw(
            {
                "object_dtype_detected": "true",
                "dtype": "O",
                "pickle_payload_detected": "true",
            }
        )
        issue = NumpyMetadataDetector().analyze(raw, _CTX)[0]
        assert issue.details["pickle_payload_detected"] is True
        assert "pickle-поток" in issue.message

    def test_location_is_file_path(self) -> None:
        """location — путь файла (для .npz это путь члена архива)."""
        raw = _raw(
            {"object_dtype_detected": "true", "dtype": "O"},
            file_path="model.npz/labels.npy",
        )
        assert NumpyMetadataDetector().analyze(raw, _CTX)[0].location == (
            "model.npz/labels.npy"
        )

    def test_compliance_tags_present(self) -> None:
        """Issue помечен compliance-тегами ОWASP/ФСТЭК/ГОСТ."""
        raw = _raw({"object_dtype_detected": "true", "dtype": "O"})
        tags = NumpyMetadataDetector().analyze(raw, _CTX)[0].compliance_tags
        assert "owasp-ml:ml03" in tags
        assert "fstec:ubi-067" in tags


# ---------------------------------------------------------------------------
# Тест 3: обход вложенных членов .npz
# ---------------------------------------------------------------------------


class TestNumpyMetadataDetectorNested:
    """Члены .npz — вложенные RawScanData, их тоже надо проверять."""

    def test_nested_numpy_member_detected(self) -> None:
        """object-dtype в члене .npz даёт issue с location члена."""
        member = _raw(
            {"object_dtype_detected": "true", "dtype": "O"},
            file_path="model.npz/labels.npy",
        )
        container = _raw({"npz_format": "zip"}, file_path="model.npz", nested=[member])
        issues = NumpyMetadataDetector().analyze(container, _CTX)
        assert len(issues) == 1
        assert issues[0].location == "model.npz/labels.npy"

    def test_two_object_members_two_issues(self) -> None:
        """Два object-массива в архиве → два issue с разными location."""
        members = [
            _raw({"object_dtype_detected": "true", "dtype": "O"}, file_path=f"m.npz/{n}.npy")
            for n in ("a", "b")
        ]
        container = _raw({"npz_format": "zip"}, file_path="m.npz", nested=members)
        issues = NumpyMetadataDetector().analyze(container, _CTX)
        assert len(issues) == 2
        assert {i.location for i in issues} == {"m.npz/a.npy", "m.npz/b.npy"}

    def test_nested_pickle_not_traversed(self) -> None:
        """Вложенный pickle-результат не обрабатывается (это чужой слой)."""
        pickle_inner = _raw(
            {"object_dtype_detected": "true"},
            scanner_name="pickle",
            file_path="obj.npy",
        )
        raw = _raw(
            {"object_dtype_detected": "true", "dtype": "O"},
            file_path="obj.npy",
            nested=[pickle_inner],
        )
        issues = NumpyMetadataDetector().analyze(raw, _CTX)
        assert len(issues) == 1, "Дубль issue из вложенного pickle-результата"

    def test_iter_entries_respects_depth_limit(self) -> None:
        """_iter_numpy_entries не уходит в бесконечную рекурсию.

        Собираем самоссылающийся RawScanData — обход обязан завершиться.
        """
        raw = _raw({"object_dtype_detected": "true", "dtype": "O"})
        raw.nested_files = [raw]
        entries = list(_iter_numpy_entries(raw))
        assert 1 <= len(entries) <= 16

    def test_make_issue_returns_none_without_flag(self) -> None:
        """_make_object_dtype_issue возвращает None без флага."""
        assert _make_object_dtype_issue(_raw({"dtype": "f4"})) is None


# ---------------------------------------------------------------------------
# Тест 4: end-to-end через Scanner (главная регрессия задачи)
# ---------------------------------------------------------------------------


class TestObjectDtypeEndToEnd:
    """Раньше object-dtype .npy давал пустой список issues — это регрессия."""

    def test_object_npy_gives_exactly_one_issue(self, tmp_path: Path) -> None:
        """object-dtype .npy → ровно один MLS-NPY-001 MEDIUM."""
        f = tmp_path / "obj.npy"
        f.write_bytes(_make_object_npy_no_pickle())

        result = Scanner().scan(f)
        issues = result.results_per_file[f].issues
        npy_issues = [i for i in issues if i.code == _ISSUE_CODE]
        assert len(npy_issues) == 1, (
            f"Ожидался ровно один {_ISSUE_CODE}, получено: "
            f"{[(i.code, i.severity.value) for i in issues]}"
        )
        assert npy_issues[0].severity is Severity.MEDIUM

    def test_object_npy_via_scan_bytes(self, tmp_path: Path) -> None:
        """Тот же результат через Python API scan_bytes()."""
        file_result = Scanner().scan_bytes(
            _make_object_npy_no_pickle(), filename="obj.npy"
        )
        codes = [i.code for i in file_result.issues]
        assert codes.count(_ISSUE_CODE) == 1

    def test_clean_float_npy_no_issue(self, tmp_path: Path) -> None:
        """Чистый float32 .npy не даёт MLS-NPY-001 (нет ложных срабатываний)."""
        f = tmp_path / "weights.npy"
        f.write_bytes(_make_float32_npy((3, 4)))

        result = Scanner().scan(f)
        codes = [i.code for i in result.results_per_file[f].issues]
        assert _ISSUE_CODE not in codes

    def test_object_npy_with_os_system_gives_critical_too(
        self, tmp_path: Path
    ) -> None:
        """object-dtype + вложенный os.system → MLS-NPY-001 и CRITICAL-находка."""
        f = tmp_path / "malicious.npy"
        f.write_bytes(_make_object_npy_with_pickle())

        result = Scanner().scan(f)
        issues = result.results_per_file[f].issues
        codes = [i.code for i in issues]
        assert codes.count(_ISSUE_CODE) == 1, (
            f"Ожидался {_ISSUE_CODE}, получено: {codes}"
        )
        criticals = [i for i in issues if i.severity is Severity.CRITICAL]
        assert criticals, (
            "Ожидалась CRITICAL-находка по os.system из blocklist, получено: "
            f"{[(i.code, i.severity.value) for i in issues]}"
        )

    def test_object_dtype_inside_npz_detected(self, tmp_path: Path) -> None:
        """object-dtype внутри .npz тоже даёт MLS-NPY-001."""
        f = tmp_path / "model.npz"
        f.write_bytes(
            _make_npz(
                ("weights", _make_float32_npy((4,))),
                ("labels", _make_object_npy_no_pickle()),
            )
        )

        result = Scanner().scan(f)
        issues = result.results_per_file[f].issues
        npy_issues = [i for i in issues if i.code == _ISSUE_CODE]
        assert len(npy_issues) == 1, (
            f"Ожидался {_ISSUE_CODE} по члену labels.npy, получено: "
            f"{[(i.code, i.location) for i in issues]}"
        )
        assert "labels.npy" in npy_issues[0].location

    def test_malicious_npz_gives_both_findings(self, tmp_path: Path) -> None:
        """.npz с object-массивом и os.system → MLS-NPY-001 + CRITICAL."""
        f = tmp_path / "mixed.npz"
        f.write_bytes(
            _make_npz(
                ("weights", _make_float32_npy((4,))),
                ("labels", _make_object_npy_with_pickle()),
            )
        )

        result = Scanner().scan(f)
        issues = result.results_per_file[f].issues
        assert any(i.code == _ISSUE_CODE for i in issues)
        assert any(i.severity is Severity.CRITICAL for i in issues)

    def test_clean_npz_no_object_issue(self, tmp_path: Path) -> None:
        """Чистый .npz не даёт MLS-NPY-001."""
        f = tmp_path / "clean.npz"
        f.write_bytes(
            _make_npz(("a", _make_float32_npy()), ("b", _make_float32_npy((2, 2))))
        )

        result = Scanner().scan(f)
        codes = [i.code for i in result.results_per_file[f].issues]
        assert _ISSUE_CODE not in codes

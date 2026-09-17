"""Аномалии заголовка GGUF: симметрия magic ↔ версия.

Раньше была асимметрия: неверная ВЕРСИЯ → структурированный MLS-GGUF-005 (детектор
по metadata-факту gguf_version), а неверный MAGIC → только свободный error (и
generic MLS-PARSE-001), хотя подделка magic — более сильный сигнал искажения.

Фикс: сканер фиксирует факт «magic != GGUF» в metadata (gguf_header_valid=false +
сырые magic/hex/expected), а GGUFMetadataDetector эмитит MLS-GGUF-006 (MEDIUM) —
тем же путём «заголовок → факт в RawScanData → Issue детектора», что и версия.
Сканер не бросает исключение и не эмитит Issue.

Фикстуры собраны ВРУЧНУЮ из байт (struct.pack заголовка), tmp_path.
"""

from __future__ import annotations

import struct
from pathlib import Path

from poison_check.core.result import MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.gguf_metadata_detector import (
    GGUFMetadataDetector,
    _make_bad_magic_issue,
)
from poison_check.scanner import Scanner
from poison_check.scanners.gguf_scanner import GGUFScanner

_CODE_BAD_MAGIC = "MLS-GGUF-006"
_CODE_BAD_VERSION = "MLS-GGUF-005"
_CODE_PARSE = "MLS-PARSE-001"

_GGUF = b"GGUF"
_CTX = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])


def _gguf_header(magic: bytes, version: int, tensor_count: int = 0, kv_count: int = 0) -> bytes:
    """Собирает минимальный GGUF-заголовок из байт (без записи тензоров)."""
    return (
        magic
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", kv_count)
    )


def fixture_gguf_bad_magic() -> bytes:
    """Заголовок с b'GGXF' вместо b'GGUF' + валидные version/counts."""
    return _gguf_header(b"GGXF", version=3)


def fixture_gguf_good_magic_bad_version() -> bytes:
    """Корректный magic, но неизвестная версия — контроль симметрии (MLS-GGUF-005)."""
    return _gguf_header(_GGUF, version=999)


def fixture_gguf_valid_minimal() -> bytes:
    """Корректный минимальный GGUF-заголовок (v3, 0 тензоров, 0 KV)."""
    return _gguf_header(_GGUF, version=3)


def _codes(issues: list) -> list[str]:
    return [i.code for i in issues]


# ---------------------------------------------------------------------------
# Bad magic → MLS-GGUF-006 (структурированный Issue, не raise)
# ---------------------------------------------------------------------------


class TestBadMagicIssue:
    """Неверный magic → MLS-GGUF-006 (MEDIUM) с сырыми байтами в message."""

    def test_bad_magic_emits_mls_gguf_006_medium(self, tmp_path: Path) -> None:
        """bad_magic → Issue MLS-GGUF-006, severity СРЕДНИЙ."""
        f = tmp_path / "bad_magic.gguf"
        f.write_bytes(fixture_gguf_bad_magic())

        fr = Scanner().scan(f).results_per_file[f]

        assert _CODE_BAD_MAGIC in _codes(fr.issues), _codes(fr.issues)
        issue = next(i for i in fr.issues if i.code == _CODE_BAD_MAGIC)
        assert issue.severity is Severity.MEDIUM

    def test_message_contains_raw_magic_and_expected(self, tmp_path: Path) -> None:
        """message содержит СЫРЫЕ прочитанные magic-байты (repr+hex) и ожидаемое."""
        f = tmp_path / "bad_magic.gguf"
        f.write_bytes(fixture_gguf_bad_magic())

        fr = Scanner().scan(f).results_per_file[f]
        issue = next(i for i in fr.issues if i.code == _CODE_BAD_MAGIC)

        # Сырые байты b'GGXF' и hex 47475846, ожидаемое b'GGUF'.
        assert "GGXF" in issue.message
        assert "47475846" in issue.message  # hex сырых байт (forensic)
        assert "GGUF" in issue.message
        # details тоже несут сырьё (forensic-инвариант, ничего не нормализовано).
        assert issue.details["bad_magic"] == "b'GGXF'"
        assert issue.details["bad_magic_hex"] == "47475846"
        assert issue.details["expected_magic"] == "b'GGUF'"

    def test_severity_not_below_bad_version(self, tmp_path: Path) -> None:
        """Severity MLS-GGUF-006 не ниже, чем у MLS-GGUF-005 (подделка сильнее)."""
        bad_magic = tmp_path / "bad_magic.gguf"
        bad_magic.write_bytes(fixture_gguf_bad_magic())
        bad_ver = tmp_path / "bad_version.gguf"
        bad_ver.write_bytes(fixture_gguf_good_magic_bad_version())

        m006 = next(
            i for i in Scanner().scan(bad_magic).results_per_file[bad_magic].issues
            if i.code == _CODE_BAD_MAGIC
        )
        m005 = next(
            i for i in Scanner().scan(bad_ver).results_per_file[bad_ver].issues
            if i.code == _CODE_BAD_VERSION
        )
        assert m006.severity >= m005.severity

    def test_no_duplicate_generic_parse_issue(self, tmp_path: Path) -> None:
        """MLS-GGUF-006 подавляет общий MLS-PARSE-001 (нет дубль-шума)."""
        f = tmp_path / "bad_magic.gguf"
        f.write_bytes(fixture_gguf_bad_magic())
        codes = _codes(Scanner().scan(f).results_per_file[f].issues)
        assert _CODE_PARSE not in codes, codes


# ---------------------------------------------------------------------------
# Сканер: симметрия и отсутствие raise
# ---------------------------------------------------------------------------


class TestScannerNoRaiseSymmetry:
    """Обе аномалии заголовка → факт в RawScanData, без исключений."""

    def test_scanner_bad_magic_no_raise_records_fact(self, tmp_path: Path) -> None:
        """Сканер на bad_magic НЕ бросает исключение — факт в metadata + error."""
        f = tmp_path / "bad_magic.gguf"
        f.write_bytes(fixture_gguf_bad_magic())

        raw = GGUFScanner().scan(f)  # не должно бросить

        assert isinstance(raw, RawScanData)
        assert raw.error is not None  # файл некорректен → error сохранён
        assert raw.metadata is not None
        assert raw.metadata.get("gguf_header_valid") == "false"
        assert raw.metadata.get("gguf_bad_magic") == "b'GGXF'"
        assert raw.metadata.get("gguf_expected_magic") == "b'GGUF'"

    def test_pipeline_bad_magic_valid_report_with_issue(self, tmp_path: Path) -> None:
        """Полный пайплайн на bad_magic → валидный отчёт с ≥1 Issue (не исключение)."""
        f = tmp_path / "bad_magic.gguf"
        f.write_bytes(fixture_gguf_bad_magic())

        result = Scanner().scan(f)  # не должно бросить
        fr = result.results_per_file[f]
        assert fr.issues, "по bad_magic файлу нет ни одной Issue — файл потерян"

    def test_both_anomalies_go_through_detector_path(self, tmp_path: Path) -> None:
        """Симметрия: и magic, и версия дают GGUF-Issue из детектора по metadata-факту."""
        bad_magic = tmp_path / "bad_magic.gguf"
        bad_magic.write_bytes(fixture_gguf_bad_magic())
        bad_ver = tmp_path / "bad_version.gguf"
        bad_ver.write_bytes(fixture_gguf_good_magic_bad_version())

        for f, expected_code in ((bad_magic, _CODE_BAD_MAGIC), (bad_ver, _CODE_BAD_VERSION)):
            raw = GGUFScanner().scan(f)
            # Сканер фиксирует факт в metadata (не эмитит Issue сам).
            assert raw.metadata is not None
            issues = GGUFMetadataDetector().analyze(raw, _CTX)
            assert expected_code in _codes(issues), (f, _codes(issues))


# ---------------------------------------------------------------------------
# Регресс: bad_version и valid не сломаны
# ---------------------------------------------------------------------------


class TestRegression:
    """Существующее поведение сохранено."""

    def test_bad_version_still_mls_gguf_005(self, tmp_path: Path) -> None:
        """Регресс: неизвестная версия по-прежнему → MLS-GGUF-005 (не сломано фиксом)."""
        f = tmp_path / "bad_version.gguf"
        f.write_bytes(fixture_gguf_good_magic_bad_version())

        codes = _codes(Scanner().scan(f).results_per_file[f].issues)
        assert _CODE_BAD_VERSION in codes, codes
        assert _CODE_BAD_MAGIC not in codes  # magic корректен — нет ложного 006

    def test_valid_minimal_no_issues(self, tmp_path: Path) -> None:
        """Корректный минимальный GGUF → 0 Issue (нет ложного MLS-GGUF-006)."""
        f = tmp_path / "valid.gguf"
        f.write_bytes(fixture_gguf_valid_minimal())

        fr = Scanner().scan(f).results_per_file[f]
        assert fr.issues == [], _codes(fr.issues)
        assert fr.error is None

    def test_file_not_lost_bad_magic_present_in_report(self, tmp_path: Path) -> None:
        """Инвариант «файл не теряется»: bad_magic присутствует в отчёте с находкой."""
        f = tmp_path / "bad_magic.gguf"
        f.write_bytes(fixture_gguf_bad_magic())

        result = Scanner().scan(f)
        assert f in result.results_per_file
        assert result.results_per_file[f].issues


# ---------------------------------------------------------------------------
# Unit: builder MLS-GGUF-006
# ---------------------------------------------------------------------------


def test_make_bad_magic_issue_unit() -> None:
    """_make_bad_magic_issue: MEDIUM/HIGH, сырые байты в message и details."""
    metadata = {
        "gguf_header_valid": "false",
        "gguf_bad_magic": "b'GGXF'",
        "gguf_bad_magic_hex": "47475846",
        "gguf_expected_magic": "b'GGUF'",
    }
    issue = _make_bad_magic_issue(metadata, "m.gguf")
    assert issue.code == _CODE_BAD_MAGIC
    assert issue.severity is Severity.MEDIUM
    assert "GGXF" in issue.message and "47475846" in issue.message
    assert issue.why is not None and issue.remediation is not None

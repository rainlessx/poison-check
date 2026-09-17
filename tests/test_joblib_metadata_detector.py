"""Тесты для JoblibMetadataDetector.

Детектор эмитит MLS-JOBLIB-001/002/003 по фактам, которые JoblibScanner
оставляет в metadata:
- ``joblib_missing_codec`` == "lz4"/"zstd" → INFO (отсутствует опциональная либа);
- ``joblib_decompression_bomb`` == "true" → MEDIUM (превышен MAX_DECOMP).

До выделения детектора сканер конструировал Issue сам, но RawScanData не имеет
поля issues — находки терялись (заметка project_joblib_info_issues_dropped).
Это тот же класс бага, что уже починен для NumpyScanner.

Вредоносные фикстуры для e2e строятся из сжатых байт (bytes-конструирование,
tmp_path); лимит декомпрессии понижается monkeypatch'ем MAX_DECOMP — реальные
2 ГБ не аллоцируются.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
import zlib
from pathlib import Path

import poison_check.scanners.joblib_scanner as joblib_scanner_module
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import Confidence, MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.joblib_metadata_detector import (
    JoblibMetadataDetector,
    _make_bomb_issue,
    _make_missing_codec_issue,
)
from poison_check.policies import PolicyLoader
from poison_check.scanner import Scanner

_CODE_LZ4 = "MLS-JOBLIB-001"
_CODE_ZSTD = "MLS-JOBLIB-002"
_CODE_BOMB = "MLS-JOBLIB-003"

_CTX = MLContext(framework="sklearn", confidence=0.9, detected_patterns=[])


def _raw(
    metadata: dict[str, str] | None,
    *,
    scanner_name: str = "joblib",
    error: str | None = None,
    file_path: str = "model.joblib",
) -> RawScanData:
    """Собирает минимальный RawScanData для unit-тестов детектора."""
    return RawScanData(
        file_path=Path(file_path),
        file_hash={},
        file_size=0,
        scanner_name=scanner_name,
        metadata=metadata,
        error=error,
    )


# ---------------------------------------------------------------------------
# Тест 1: контракт детектора и регистрация
# ---------------------------------------------------------------------------


class TestJoblibMetadataDetectorContract:
    """Атрибуты детектора, регистрация и включение в политики."""

    def test_detector_name(self) -> None:
        """Имя детектора совпадает с именем в политиках."""
        assert JoblibMetadataDetector.name == "joblib_metadata"

    def test_severity_range(self) -> None:
        """severity_range — от INFO (отсутствие кодека) до HIGH (ratio-bomb)."""
        assert JoblibMetadataDetector.severity_range == (
            Severity.INFO,
            Severity.HIGH,
        )

    def test_registered_in_registry(self) -> None:
        """Детектор зарегистрирован в DetectorRegistry."""
        assert DetectorRegistry.get("joblib_metadata") is JoblibMetadataDetector

    def test_enabled_in_all_builtin_policies(self) -> None:
        """Детектор включён во всех встроенных политиках."""
        for policy_name in ("default", "banking", "government", "strict"):
            policy = PolicyLoader.load(policy_name)
            enabled = set(policy.get("enabled_detectors") or [])
            assert "joblib_metadata" in enabled, (
                f"Политика {policy_name!r} не включает joblib_metadata"
            )


# ---------------------------------------------------------------------------
# Тест 2: unit-логика analyze()
# ---------------------------------------------------------------------------


class TestJoblibMetadataDetectorAnalyze:
    """Детектор реагирует только на факты joblib-сканера."""

    def test_ignores_other_scanners(self) -> None:
        """RawScanData от другого сканера игнорируется даже с фактом bomb."""
        raw = _raw({"joblib_decompression_bomb": "true"}, scanner_name="pickle")
        assert JoblibMetadataDetector().analyze(raw, _CTX) == []

    def test_no_metadata_no_issues(self) -> None:
        """metadata=None → пустой список."""
        assert JoblibMetadataDetector().analyze(_raw(None), _CTX) == []

    def test_irrelevant_metadata_no_issues(self) -> None:
        """metadata без релевантных фактов → находок нет."""
        raw = _raw({"joblib_compression": "zlib"})
        assert JoblibMetadataDetector().analyze(raw, _CTX) == []

    def test_missing_lz4_emits_info(self) -> None:
        """joblib_missing_codec=lz4 → ровно один MLS-JOBLIB-001 INFO CERTAIN."""
        raw = _raw({"joblib_missing_codec": "lz4"})
        issues = JoblibMetadataDetector().analyze(raw, _CTX)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.code == _CODE_LZ4
        assert issue.severity is Severity.INFO
        assert issue.confidence is Confidence.CERTAIN
        assert "lz4" in issue.message.lower()

    def test_missing_zstd_emits_info(self) -> None:
        """joblib_missing_codec=zstd → MLS-JOBLIB-002 INFO."""
        raw = _raw({"joblib_missing_codec": "zstd"})
        issues = JoblibMetadataDetector().analyze(raw, _CTX)
        assert len(issues) == 1
        assert issues[0].code == _CODE_ZSTD
        assert issues[0].severity is Severity.INFO
        assert "zstd" in issues[0].message.lower()

    def test_unknown_codec_value_ignored(self) -> None:
        """Незнакомое значение joblib_missing_codec → находок нет (defensive)."""
        raw = _raw({"joblib_missing_codec": "brotli"})
        assert JoblibMetadataDetector().analyze(raw, _CTX) == []

    def test_bomb_emits_medium(self) -> None:
        """joblib_decompression_bomb=true → MLS-JOBLIB-003 MEDIUM HIGH."""
        raw = _raw(
            {
                "joblib_decompression_bomb": "true",
                "joblib_bomb_method": "gzip",
                "joblib_bomb_limit_bytes": "2000000000",
            }
        )
        issues = JoblibMetadataDetector().analyze(raw, _CTX)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.code == _CODE_BOMB
        assert issue.severity is Severity.MEDIUM
        assert issue.confidence is Confidence.HIGH
        assert issue.details["compression"] == "gzip"
        assert issue.details["limit_bytes"] == 2_000_000_000

    def test_bomb_emitted_even_when_error_set(self) -> None:
        """КРИТИЧНО: bomb ставит И error, И факт — детектор не должен молчать.

        Регрессия задачи 8: ранний ``if raw_data.error: return []`` снова
        потерял бы MLS-JOBLIB-003, так как при bomb error всегда задан.
        """
        raw = _raw(
            {
                "joblib_decompression_bomb": "true",
                "joblib_bomb_method": "zlib",
                "joblib_bomb_limit_bytes": "1024",
            },
            error="Декомпрессия zlib превысила лимит 0 ГБ",
        )
        codes = [i.code for i in JoblibMetadataDetector().analyze(raw, _CTX)]
        assert codes == [_CODE_BOMB]

    def test_location_is_file_path(self) -> None:
        """location Issue — путь к файлу, а не пустая строка."""
        raw = _raw({"joblib_missing_codec": "lz4"}, file_path="/models/m.joblib")
        issue = JoblibMetadataDetector().analyze(raw, _CTX)[0]
        assert issue.location == "/models/m.joblib"

    def test_texts_are_russian(self) -> None:
        """Сообщения детектора на русском (why/remediation заполнены)."""
        raw = _raw(
            {
                "joblib_decompression_bomb": "true",
                "joblib_bomb_method": "bz2",
                "joblib_bomb_limit_bytes": "2000000000",
            }
        )
        issue = JoblibMetadataDetector().analyze(raw, _CTX)[0]
        assert issue.why is not None and "bomb" in issue.why.lower()
        assert issue.remediation is not None
        # Кириллица присутствует
        assert any("а" <= ch.lower() <= "я" for ch in issue.message)


# ---------------------------------------------------------------------------
# Тест 3: helper-функции модульного уровня
# ---------------------------------------------------------------------------


class TestJoblibMetadataHelpers:
    """Прямое тестирование builder-функций (без RawScanData)."""

    def test_make_missing_codec_unknown_returns_none(self) -> None:
        """Незнакомый кодек → None."""
        assert _make_missing_codec_issue("snappy", "m.joblib") is None

    def test_make_bomb_issue_without_limit_fact(self) -> None:
        """Отсутствие joblib_bomb_limit_bytes не ломает построение Issue."""
        issue = _make_bomb_issue({"joblib_bomb_method": "gzip"}, "m.joblib")
        assert issue.code == _CODE_BOMB
        assert issue.details["compression"] == "gzip"
        # limit_bytes отсутствует, но details валиден
        assert "limit_bytes" not in issue.details

    def test_make_bomb_issue_non_numeric_limit_ignored(self) -> None:
        """Нечисловое значение лимита не попадает в details и не бросает."""
        issue = _make_bomb_issue(
            {"joblib_bomb_method": "zlib", "joblib_bomb_limit_bytes": "NaN"},
            "m.joblib",
        )
        assert "limit_bytes" not in issue.details


# ---------------------------------------------------------------------------
# Тест 4: end-to-end через Scanner (полный пайплайн)
# ---------------------------------------------------------------------------


class TestJoblibMetadataEndToEnd:
    """Факт из metadata доходит до file_result.issues через Scanner."""

    def test_zlib_bomb_surfaces_mls_joblib_003(self, tmp_path: Path) -> None:
        """zlib-бомба (маленький вход, раскрытие > MAX_DECOMP) → MLS-JOBLIB-003.

        Понижаем MAX_DECOMP monkeypatch'ем до 1 КБ — 2 ГБ не аллоцируем.
        До задачи 8 bomb давала только error, MLS-JOBLIB-003 в issues отсутствовал.
        """
        bomb_file = tmp_path / "bomb.joblib"
        bomb_file.write_bytes(zlib.compress(b"\x00" * 4096))

        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024):
            result = Scanner().scan(bomb_file)

        file_result = result.results_per_file[bomb_file]
        codes = [i.code for i in file_result.issues]
        assert _CODE_BOMB in codes, (
            f"MLS-JOBLIB-003 должен присутствовать в issues, получено: {codes}"
        )
        # bomb сохраняет error (exit 2)
        assert file_result.error is not None

    def test_missing_lz4_surfaces_info_issue(self, tmp_path: Path) -> None:
        """lz4-файл при отсутствии lz4 → MLS-JOBLIB-001 INFO в issues.

        lz4 в окружении установлена, поэтому её отсутствие эмулируем через
        подмену sys.modules — импорт lz4.frame внутри сканера падает ImportError.
        """
        lz4_file = tmp_path / "model.joblib"
        lz4_file.write_bytes(b"\x04\x22\x4d\x18" + b"\x00" * 32)  # LZ4 frame magic

        with mock.patch.dict(sys.modules, {"lz4": None, "lz4.frame": None}):
            result = Scanner().scan(lz4_file)

        file_result = result.results_per_file[lz4_file]
        codes = [i.code for i in file_result.issues]
        assert _CODE_LZ4 in codes, (
            f"MLS-JOBLIB-001 должен присутствовать в issues, получено: {codes}"
        )
        # Отсутствие кодека — не ошибка сканирования (error=None)
        assert file_result.error is None

    def test_clean_zlib_joblib_no_joblib_issue(self, tmp_path: Path) -> None:
        """Валидный несжатый/сжатый joblib без bomb → нет MLS-JOBLIB-* issues."""
        # Минимальный безопасный pickle (пустой список) под zlib
        safe_pickle = b"\x80\x04\x95\x03\x00\x00\x00\x00\x00\x00\x00]\x94."
        clean = tmp_path / "clean.joblib"
        clean.write_bytes(zlib.compress(safe_pickle))

        result = Scanner().scan(clean)
        codes = [i.code for i in result.results_per_file[clean].issues]
        assert not any(c.startswith("MLS-JOBLIB-") for c in codes), (
            f"Чистый joblib не должен давать MLS-JOBLIB-* находок: {codes}"
        )

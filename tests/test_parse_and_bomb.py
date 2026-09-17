"""Тесты инварианта «файл никогда не теряется» + decompression bomb.

Закрывает баг: joblib-файл, тело которого не разбирается как pickle (в т.ч.
decompression bomb — короткий zlib-поток, разжимающийся во много раз), давал в
консоль строки-ошибки и НИ ОДНОЙ Issue — выпадал из severity-таблицы.

Два независимых дефекта и их фиксы проверяются здесь:
  A) Общий сбой разбора → всегда ≥1 Issue уровня LOW/MEDIUM (MLS-PARSE-001,
     ParseErrorDetector). Ноль Issue по упавшему файлу запрещён.
  B) Ratio-bomb распознаётся сканером ДО разбора pickle, по коэффициенту
     распаковки, с ограничением чтения (потоковый инвариант) → MLS-BOMB-001
     (HIGH, JoblibMetadataDetector).

Фикстуры собираются вручную из байт (bytes-конструирование, tmp_path); НЕ через
pickle.dumps злонамеренного объекта. Лимиты декомпрессии при необходимости
понижаются monkeypatch'ем — реальные объёмы не аллоцируются.
"""

from __future__ import annotations

import tracemalloc
import unittest.mock as mock
import zlib
from pathlib import Path

import poison_check.scanners.joblib_scanner as joblib_scanner_module
from poison_check.core.result import MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.joblib_metadata_detector import JoblibMetadataDetector
from poison_check.detectors.parse_error_detector import (
    ParseErrorDetector,
    _has_other_signal,
)
from poison_check.scanner import Scanner
from poison_check.scanners.joblib_scanner import JoblibScanner

# Минимальный безопасный pickle — пустой список ]. (PROTO 4, FRAME, EMPTY_LIST, STOP)
_SAFE_PICKLE = b"\x80\x04\x95\x03\x00\x00\x00\x00\x00\x00\x00]\x94."

# Пустой ML-контекст: находки parse/bomb от фреймворка не зависят.
_CTX = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])

_CODE_PARSE = "MLS-PARSE-001"
_CODE_BOMB = "MLS-BOMB-001"


def _codes(issues: list) -> list[str]:
    return [i.code for i in issues]


# ---------------------------------------------------------------------------
# fixture-builders (ручная сборка байт)
# ---------------------------------------------------------------------------


def _write_unparseable_joblib(path: Path) -> None:
    """joblib-контейнер (валидный zlib), тело которого genops не разбирает.

    Внутри: PROTO 4 + FRAME-опкод (0x95), у которого нет 8-байтной длины —
    genops останавливается сразу, глобалов не извлекает. Идёт по joblib-ветке
    (zlib-magic), а не опознаётся как сырой pickle.
    """
    path.write_bytes(zlib.compress(b"\x80\x04\x95\xff\xff"))


def _write_zlib_bomb(path: Path, decompressed_size: int) -> int:
    """Пишет короткий zlib-поток из нулей, разжимающийся в decompressed_size.

    Возвращает фактический сжатый размер файла (для проверки коэффициента).
    """
    comp = zlib.compress(b"\x00" * decompressed_size)
    path.write_bytes(comp)
    return len(comp)


def _write_safe_joblib(path: Path) -> None:
    """Валидный безопасный joblib (пустой список под zlib)."""
    path.write_bytes(zlib.compress(_SAFE_PICKLE))


# ---------------------------------------------------------------------------
# A) Упавший при разборе файл → ровно ≥1 Issue MLS-PARSE-001 (LOW/MEDIUM)
# ---------------------------------------------------------------------------


class TestUnparseableFileBecomesIssue:
    """Инвариант: сбой разбора никогда не даёт ноль Issue."""

    def test_unparseable_joblib_yields_mls_parse_001(self, tmp_path: Path) -> None:
        """Непарсящийся joblib → ≥1 Issue MLS-PARSE-001, severity LOW/MEDIUM."""
        f = tmp_path / "unparseable.joblib"
        _write_unparseable_joblib(f)

        file_result = Scanner().scan(f).results_per_file[f]

        assert file_result.issues, "упавший файл дал НОЛЬ Issue — инвариант нарушен"
        assert _CODE_PARSE in _codes(file_result.issues), _codes(file_result.issues)
        parse_issue = next(i for i in file_result.issues if i.code == _CODE_PARSE)
        assert parse_issue.severity in (Severity.LOW, Severity.MEDIUM)
        # Причина сбоя доходит как поле данных (не хардкод в Output).
        assert "error" in parse_issue.details
        # error по файлу сохранён (exit 2 отработает штатно).
        assert file_result.error is not None

    def test_parse_error_issue_not_zero_via_bytes(self) -> None:
        """scan_bytes непарсящегося joblib тоже даёт MLS-PARSE-001 (не ноль)."""
        data = zlib.compress(b"\x80\x04\x95\xff\xff")
        fr = Scanner().scan_bytes(data, filename="broken.joblib")
        assert _CODE_PARSE in _codes(fr.issues)

    def test_detector_suppressed_when_globals_extracted(self) -> None:
        """MLS-PARSE-001 подавляется, если сканер извлёк globals (нет дубля).

        При наличии globals файл не потерян — сработает allowlist/blocklist,
        общий parse-issue был бы шумом.
        """
        raw = RawScanData(
            file_path=Path("x.pkl"),
            file_hash={},
            file_size=1,
            scanner_name="pickle",
            globals={("os", "system")},
            error="at position 5, opcode ... unknown",
        )
        assert ParseErrorDetector().analyze(raw, _CTX) == []
        assert _has_other_signal(raw) is True

    def test_detector_suppressed_on_bomb_fact(self) -> None:
        """MLS-PARSE-001 подавляется при факте bomb (эмитится MLS-BOMB-001)."""
        raw = RawScanData(
            file_path=Path("b.joblib"),
            file_hash={},
            file_size=1,
            scanner_name="joblib",
            metadata={"joblib_decompression_bomb": "true", "joblib_bomb_kind": "ratio"},
            error="Decompression bomb (zlib): ...",
        )
        assert ParseErrorDetector().analyze(raw, _CTX) == []

    def test_detector_silent_without_error(self) -> None:
        """Без error детектор молчит (не ложное срабатывание на чистом файле)."""
        raw = RawScanData(
            file_path=Path("ok.pkl"),
            file_hash={},
            file_size=1,
            scanner_name="pickle",
        )
        assert ParseErrorDetector().analyze(raw, _CTX) == []

    def test_pure_parse_error_emits_single_low_issue(self) -> None:
        """«Чистый» сбой разбора (нет globals/фактов) → ровно один LOW Issue."""
        raw = RawScanData(
            file_path=Path("t.joblib"),
            file_hash={},
            file_size=1,
            scanner_name="joblib",
            error="not enough data in stream to read uint8",
        )
        issues = ParseErrorDetector().analyze(raw, _CTX)
        assert len(issues) == 1
        assert issues[0].code == _CODE_PARSE
        assert issues[0].severity is Severity.LOW


# ---------------------------------------------------------------------------
# B) Decompression bomb → MLS-BOMB-001 HIGH + ограниченное чтение
# ---------------------------------------------------------------------------


class TestDecompressionBomb:
    """Ratio-bomb распознаётся до разбора pickle, чтение ограничено лимитом."""

    def test_zlib_bomb_yields_mls_bomb_001_high(self, tmp_path: Path) -> None:
        """zlib-бомба (короткий вход, огромное раскрытие) → MLS-BOMB-001 HIGH."""
        f = tmp_path / "bomb.joblib"
        # 16 МБ нулей → ~16 КБ zlib (коэффициент ~1000, выше порога 500× и
        # выше пола 8 МБ). Дефолтные константы — без monkeypatch.
        comp_size = _write_zlib_bomb(f, 16 * 1024 * 1024)
        assert comp_size < 1_000_000  # действительно короткий вход

        file_result = Scanner().scan(f).results_per_file[f]

        assert _CODE_BOMB in _codes(file_result.issues), _codes(file_result.issues)
        bomb_issue = next(i for i in file_result.issues if i.code == _CODE_BOMB)
        assert bomb_issue.severity is Severity.HIGH
        # Общий MLS-PARSE-001 по bomb не дублируется.
        assert _CODE_PARSE not in _codes(file_result.issues)
        assert file_result.error is not None

    def test_bomb_read_is_bounded_not_full_alloc(self, tmp_path: Path) -> None:
        """Сканер НЕ аллоцирует полный разжатый объём — чтение ограничено лимитом.

        Проверяем двумя независимыми способами:
          1) факт joblib_bomb_decompressed_bytes (прочитано на момент останова)
             строго меньше полного разжатого объёма;
          2) пиковая аллокация tracemalloc за время scan строго меньше полного
             разжатого объёма.
        """
        full_size = 32 * 1024 * 1024  # 32 МБ полного раскрытия
        f = tmp_path / "bomb_big.joblib"
        _write_zlib_bomb(f, full_size)

        tracemalloc.start()
        raw_data = JoblibScanner().scan(f)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        meta = raw_data.metadata or {}
        assert meta.get("joblib_decompression_bomb") == "true"
        assert meta.get("joblib_bomb_kind") == "ratio"

        read_bytes = int(meta["joblib_bomb_decompressed_bytes"])
        assert read_bytes < full_size, (
            f"прочитано {read_bytes} из {full_size} — чтение не было ограничено"
        )
        assert peak < full_size, (
            f"пиковая аллокация {peak} ≈ полный объём {full_size} — bomb "
            "аллоцирован целиком (потоковый инвариант нарушен)"
        )

    def test_high_ratio_bomb_read_dramatically_bounded(self, tmp_path: Path) -> None:
        """Высокоэнтропийный кодек (bz2) с огромным ratio → чтение « полного объёма.

        bz2 сжимает нули в сотни раз сильнее zlib, поэтому лимит (при пониженном
        поле 256 КБ) на порядки меньше полного раскрытия. Прочитанный объём —
        меньше десятой доли полного, что явно демонстрирует ранний выход
        (потоковый инвариант), а не пост-фактум обнаружение.
        """
        import bz2

        full_size = 8 * 1024 * 1024
        comp = bz2.compress(b"\x00" * full_size)
        f = tmp_path / "bomb.joblib"
        f.write_bytes(comp)

        with mock.patch.object(joblib_scanner_module, "BOMB_OUTPUT_FLOOR", 256 * 1024):
            raw_data = JoblibScanner().scan(f)

        meta = raw_data.metadata or {}
        assert meta.get("joblib_bomb_kind") == "ratio"
        assert meta.get("joblib_bomb_method") == "bz2"
        read_bytes = int(meta["joblib_bomb_decompressed_bytes"])
        assert read_bytes < full_size // 10, (
            f"прочитано {read_bytes} из {full_size} — bomb почти аллоцирован"
        )

    def test_bomb_detector_unit_high(self) -> None:
        """Unit: факт kind=ratio → JoblibMetadataDetector даёт MLS-BOMB-001 HIGH."""
        raw = RawScanData(
            file_path=Path("m.joblib"),
            file_hash={},
            file_size=1,
            scanner_name="joblib",
            metadata={
                "joblib_decompression_bomb": "true",
                "joblib_bomb_kind": "ratio",
                "joblib_bomb_method": "zlib",
                "joblib_bomb_ratio": "1024",
                "joblib_bomb_compressed_bytes": "1024",
                "joblib_bomb_decompressed_bytes": "8388608",
            },
            error="Decompression bomb (zlib): ...",
        )
        issues = JoblibMetadataDetector().analyze(raw, _CTX)
        assert len(issues) == 1
        assert issues[0].code == _CODE_BOMB
        assert issues[0].severity is Severity.HIGH
        assert issues[0].details["ratio"] == "1024"


# ---------------------------------------------------------------------------
# Регресс: чистые файлы не дают ложных MLS-PARSE / MLS-BOMB
# ---------------------------------------------------------------------------


class TestCleanFilesNoFalsePositive:
    """Валидные безопасные файлы по-прежнему дают 0 находок."""

    def test_existing_safe_joblib_fixture_clean(self) -> None:
        """Существующая безопасная фикстура simple_list.joblib → 0 issues."""
        p = Path(__file__).parent / "fixtures" / "safe" / "simple_list.joblib"
        fr = Scanner().scan(p).results_per_file[p]
        assert fr.issues == [], _codes(fr.issues)
        assert fr.error is None

    def test_existing_safe_gzip_joblib_fixture_clean(self) -> None:
        """Существующая безопасная gzip-фикстура → нет MLS-PARSE/BOMB."""
        p = Path(__file__).parent / "fixtures" / "safe" / "simple_list_gzip.joblib"
        fr = Scanner().scan(p).results_per_file[p]
        codes = _codes(fr.issues)
        assert not any(c.startswith("MLS-PARSE") or c.startswith("MLS-BOMB") for c in codes)
        assert fr.error is None

    def test_inline_safe_joblib_clean(self, tmp_path: Path) -> None:
        """Собранный вручную безопасный joblib → 0 issues (нет false positive)."""
        f = tmp_path / "clean.joblib"
        _write_safe_joblib(f)
        fr = Scanner().scan(f).results_per_file[f]
        assert fr.issues == [], _codes(fr.issues)

    def test_safe_pickle_no_parse_issue(self, tmp_path: Path) -> None:
        """Валидный .pkl → нет MLS-PARSE-001 (error не выставлен)."""
        f = tmp_path / "clean.pkl"
        f.write_bytes(_SAFE_PICKLE)
        fr = Scanner().scan(f).results_per_file[f]
        assert _CODE_PARSE not in _codes(fr.issues)
        assert fr.error is None


# ---------------------------------------------------------------------------
# Инвариант: набор битых входов → число отчётных записей == числу файлов
# ---------------------------------------------------------------------------


class TestFileNeverLost:
    """Каждый битый файл учтён в отчёте и даёт хотя бы одну находку."""

    def test_broken_inputs_each_produce_a_finding(self, tmp_path: Path) -> None:
        """Директория битых joblib/pkl: записей == файлов, у каждого ≥1 Issue."""
        # Битые файлы разного рода, все идущие через сканеры (joblib/pickle):
        _write_unparseable_joblib(tmp_path / "a_truncated.joblib")
        _write_zlib_bomb(tmp_path / "b_bomb.joblib", 16 * 1024 * 1024)
        # Валидный zlib-контейнер с мусором внутри (не pickle):
        (tmp_path / "c_garbage.joblib").write_bytes(zlib.compress(b"not a pickle stream"))
        # Усечённый pickle .pkl (валидный старт, обрыв опкода):
        (tmp_path / "d_truncated.pkl").write_bytes(b"\x80\x04\x95\x10\x00\x00\x00")

        files = sorted(tmp_path.iterdir())
        result = Scanner().scan(tmp_path)

        # Число отчётных записей == числу файлов.
        assert len(result.results_per_file) == len(files)

        # Ни один файл не потерян: у каждого ≥1 Issue.
        for path, fr in result.results_per_file.items():
            assert fr.issues, f"файл {path.name} потерян — ноль Issue"

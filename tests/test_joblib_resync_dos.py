"""Регрессионный тест DoS в joblib resync (аудит #6 — квадратичная сложность).

Старая реализация ``_resync_pickle_frames`` шла по КАЖДОМУ байту
декомпрессированного потока и на каждом PROTO-маркере (``\\x80`` + версия 2-5)
вызывала ``scan_bytes(data[i:])`` — копия всего хвоста плюс полный повторный
парсинг. При ранней parse error (атакующий обеспечивает её битым первым фреймом)
и потоке из тысяч ``\\x80\\x02`` это давало десятки млн вызовов на копиях до
сотен МБ → зависание/OOM.

Новая реализация ищет маркеры одним проходом (``bytes.find``), парсит окно
фиксированного размера и ограничивает число попыток и суммарный объём. Тесты
проверяют, что число вызовов ``scan_bytes`` ограничено, а обработка завершается
за разумное время.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.joblib_scanner import (
    _MAX_RESYNC_ATTEMPTS,
    JoblibScanner,
)
from poison_check.scanners.pickle_scanner import PickleScanner


def _make_resync_bomb(marker_count: int) -> bytes:
    """Собирает «resync-бомбу»: ранняя parse error + flood из ``\\x80\\x02``.

    ``\\x80\\x02`` — PROTO 2, затем ``\\x05`` — неизвестный opcode: genops падает
    на позиции 2. После этого идёт ``marker_count`` повторов ``\\x80\\x02``, каждый
    из которых в старой реализации триггерил отдельный ``scan_bytes(data[i:])``.
    """
    return b"\x80\x02\x05" + b"\x80\x02" * marker_count


class TestResyncBombBounded:
    """resync-бомба обрабатывается за ограниченное число вызовов scan_bytes."""

    def test_scan_bytes_call_count_is_bounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тысячи ``\\x80\\x02`` + ранняя ошибка → число scan_bytes ограничено."""
        calls = {"n": 0}
        original = PickleScanner.scan_bytes

        def counting(self: PickleScanner, data: bytes, source_path: Path) -> RawScanData:
            calls["n"] += 1
            return original(self, data, source_path)

        monkeypatch.setattr(PickleScanner, "scan_bytes", counting)

        # 50 000 маркеров: в старой O(n²) реализации это 50k вызовов на копиях.
        path = tmp_path / "resync_bomb.joblib"
        path.write_bytes(_make_resync_bomb(50_000))

        start = time.monotonic()
        raw = JoblibScanner().scan(path)
        elapsed = time.monotonic() - start

        # Один основной scan_bytes + не более _MAX_RESYNC_ATTEMPTS попыток resync.
        assert calls["n"] <= 1 + _MAX_RESYNC_ATTEMPTS, (
            f"Слишком много вызовов scan_bytes: {calls['n']} "
            f"(ожидалось ≤ {1 + _MAX_RESYNC_ATTEMPTS}) — возможна квадратичность."
        )
        # Сканер не должен бросать исключение и обязан завершиться быстро.
        assert raw.scanner_name == "joblib"
        assert elapsed < 10.0, f"resync-бомба обрабатывалась слишком долго: {elapsed:.1f}с"

    def test_resync_helper_call_count_directly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Прямой вызов _resync_pickle_frames не превышает лимит попыток."""
        calls = {"n": 0}
        original = PickleScanner.scan_bytes

        def counting(self: PickleScanner, data: bytes, source_path: Path) -> RawScanData:
            calls["n"] += 1
            return original(self, data, source_path)

        monkeypatch.setattr(PickleScanner, "scan_bytes", counting)

        data = _make_resync_bomb(50_000)
        # error_pos=2 (unknown opcode \x05), как в реальном inner.error.
        JoblibScanner._resync_pickle_frames(data, 2, tmp_path / "x.joblib")

        assert calls["n"] <= _MAX_RESYNC_ATTEMPTS, (
            f"resync сделал {calls['n']} вызовов scan_bytes "
            f"(лимит {_MAX_RESYNC_ATTEMPTS})."
        )


class TestResyncStillDetects:
    """Ограничение попыток не ломает детекцию реального parse-stop payload."""

    def test_malicious_global_in_early_frame_detected(
        self, tmp_path: Path
    ) -> None:
        """os.system во втором фрейме (сразу после ошибки) по-прежнему находится."""
        import struct

        first_frame = b"\x80\x02(K\x01"  # PROTO 2, MARK, SHORT_BININT(1), без STOP
        raw_interrupt = bytes([0x05, 0x00, 0x01, 0xFF]) * 20
        cmd = b"echo pwned"
        second_frame = (
            b"\x80\x02"
            b"cos\nsystem\n"
            b"X" + struct.pack("<I", len(cmd)) + cmd
            + b"q\x00\x85Rq\x01."
        )

        data = first_frame + raw_interrupt + second_frame
        # Ошибка на позиции 5 (opcode \x05); resync ищет фреймы после неё.
        extra = JoblibScanner._resync_pickle_frames(data, 5, tmp_path / "x.joblib")

        assert ("os", "system") in extra, (
            f"Вредоносный global не найден при resync: {extra}"
        )

    def test_bomb_does_not_hide_trailing_malicious_frame(
        self, tmp_path: Path
    ) -> None:
        """Небольшой flood + вредоносный фрейм в пределах лимита попыток → найден."""
        import struct

        cmd = b"id"
        mal_frame = (
            b"\x80\x02"
            b"cos\nsystem\n"
            b"X" + struct.pack("<I", len(cmd)) + cmd
            + b"q\x00\x85Rq\x01."
        )
        # Небольшой flood (меньше лимита попыток), затем вредоносный фрейм.
        data = b"\x80\x02\x05" + b"\x80\x02" * 10 + mal_frame
        extra = JoblibScanner._resync_pickle_frames(data, 2, tmp_path / "x.joblib")

        assert ("os", "system") in extra, (
            f"Вредоносный фрейм после flood не найден: {extra}"
        )

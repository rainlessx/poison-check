"""Тесты опционального 7z-разбора в ContainerExtractor.

Мотивация — nullifAI (ReversingLabs 2025): вредоносные модели прячут pickle в
7z-контейнере, чтобы обойти сканеры, знающие только zip/tar. py7zr — ОПЦИОНАЛЬНАЯ
зависимость: при её отсутствии extract_7z_members отдаёт graceful ContainerError
(а не ImportError), чтобы вызывающий код мог деградировать в Issue INFO.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from poison_check.core.container import ContainerError, ContainerExtractor

_PY7ZR_AVAILABLE: bool = importlib.util.find_spec("py7zr") is not None

_7Z_MAGIC = b"7z\xbc\xaf\x27\x1c"


# ---------------------------------------------------------------------------
# Детекция magic bytes (не требует py7zr)
# ---------------------------------------------------------------------------


class TestIs7z:
    def test_detects_7z_magic(self, tmp_path: Path) -> None:
        f = tmp_path / "a.7z"
        f.write_bytes(_7Z_MAGIC + b"\x00" * 16)
        assert ContainerExtractor.is_7z(f) is True

    def test_rejects_zip(self, tmp_path: Path) -> None:
        f = tmp_path / "a.zip"
        f.write_bytes(b"PK\x03\x04" + b"\x00" * 16)
        assert ContainerExtractor.is_7z(f) is False

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        assert ContainerExtractor.is_7z(tmp_path / "nope.7z") is False


# ---------------------------------------------------------------------------
# Graceful degradation при отсутствии py7zr
# ---------------------------------------------------------------------------


class TestExtract7zDegradation:
    @pytest.mark.skipif(_PY7ZR_AVAILABLE, reason="py7zr установлен — тест для его отсутствия")
    def test_missing_py7zr_raises_container_error(self, tmp_path: Path) -> None:
        """Без py7zr — ContainerError (не ImportError), сообщение с подсказкой."""
        f = tmp_path / "a.7z"
        f.write_bytes(_7Z_MAGIC + b"\x00" * 16)
        with pytest.raises(ContainerError) as exc_info:
            list(ContainerExtractor.extract_7z_members(f))
        assert "py7zr" in str(exc_info.value)

    @pytest.mark.skipif(not _PY7ZR_AVAILABLE, reason="требуется py7zr")
    def test_roundtrip_with_py7zr(self, tmp_path: Path) -> None:
        """С py7zr извлекаются члены архива (round-trip)."""
        import py7zr  # noqa: PLC0415

        f = tmp_path / "a.7z"
        with py7zr.SevenZipFile(f, "w") as archive:
            archive.writestr(b"\x80\x04hello", "data.pkl")

        members = dict(ContainerExtractor.extract_7z_members(f))
        assert "data.pkl" in members
        assert members["data.pkl"] == b"\x80\x04hello"

    @pytest.mark.skipif(not _PY7ZR_AVAILABLE, reason="требуется py7zr")
    def test_corrupted_7z_raises_container_error(self, tmp_path: Path) -> None:
        """7z-magic + мусор → ContainerError, не необработанное исключение."""
        f = tmp_path / "bad.7z"
        f.write_bytes(_7Z_MAGIC + b"\xff" * 64)
        with pytest.raises(ContainerError):
            list(ContainerExtractor.extract_7z_members(f))

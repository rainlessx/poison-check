"""Тесты для poison_check/core/format_detector.py."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from poison_check.core.format_detector import FileFormat, FormatDetector, FormatSignature

# ---------------------------------------------------------------------------
# Минимальные тестовые байты для каждого формата
# ---------------------------------------------------------------------------

ZIP_MAGIC = b"\x50\x4b\x03\x04" + b"\x00" * 26  # ZIP Local File Header
NUMPY_NPY_MAGIC = b"\x93NUMPY" + b"\x00" * 506
GGUF_MAGIC = b"GGUF" + b"\x00" * 508
PICKLE2_MAGIC = b"\x80\x02" + b"\x00" * 510
PICKLE3_MAGIC = b"\x80\x03" + b"\x00" * 510
PICKLE4_MAGIC = b"\x80\x04" + b"\x00" * 510
PICKLE5_MAGIC = b"\x80\x05" + b"\x00" * 510

# TAR: magic bytes "ustar" находятся со смещением 257 байт
TAR_MAGIC = b"\x00" * 257 + b"ustar" + b"\x00" * 250

# SafeTensors: 8 байт uint64 LE (размер заголовка) + '{' + контент
_ST_HEADER = b'{"__metadata__": {}}'
_ST_HEADER_LEN = len(_ST_HEADER)
SAFETENSORS_MAGIC = struct.pack("<Q", _ST_HEADER_LEN) + _ST_HEADER + b"\x00" * 484

# Байты, не совпадающие ни с одним форматом
UNKNOWN_MAGIC = b"\xff\xfe\xfd\xfc" + b"\x00" * 508


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def write_tmp(tmp_path: Path, name: str, data: bytes) -> Path:
    """Записывает байты во временный файл и возвращает путь к нему."""
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Тесты FormatSignature
# ---------------------------------------------------------------------------


class TestFormatSignature:
    def test_defaults(self) -> None:
        sig = FormatSignature(format=FileFormat.PICKLE, magic=b"\x80\x02")
        assert sig.offset == 0
        assert sig.description == ""

    def test_custom_values(self) -> None:
        sig = FormatSignature(
            format=FileFormat.TAR_ARCHIVE,
            magic=b"ustar",
            offset=257,
            description="TAR",
        )
        assert sig.offset == 257
        assert sig.description == "TAR"


# ---------------------------------------------------------------------------
# Тесты SIGNATURES
# ---------------------------------------------------------------------------


class TestSignatures:
    def test_signatures_is_list(self) -> None:
        assert isinstance(FormatDetector.SIGNATURES, list)

    def test_signatures_not_empty(self) -> None:
        assert len(FormatDetector.SIGNATURES) > 0

    def test_all_signatures_have_format(self) -> None:
        for sig in FormatDetector.SIGNATURES:
            assert isinstance(sig.format, FileFormat)
            assert len(sig.magic) > 0

    def test_tar_signature_offset(self) -> None:
        tar_sigs = [s for s in FormatDetector.SIGNATURES if s.format == FileFormat.TAR_ARCHIVE]
        assert tar_sigs, "Сигнатура TAR должна присутствовать"
        assert tar_sigs[0].offset == 257

    def test_pickle_protocols_covered(self) -> None:
        pickle_magics = {
            s.magic
            for s in FormatDetector.SIGNATURES
            if s.format == FileFormat.PICKLE
        }
        for expected in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
            assert expected in pickle_magics


# ---------------------------------------------------------------------------
# Тесты detect() — ZIP-контейнер
# ---------------------------------------------------------------------------


class TestDetectZipVariants:
    def test_zip_with_pt_extension_is_pytorch(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "model.pt", ZIP_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PYTORCH

    def test_zip_with_pth_extension_is_pytorch(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "weights.pth", ZIP_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PYTORCH

    def test_zip_with_bin_extension_is_pytorch(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "model.bin", ZIP_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PYTORCH

    def test_zip_with_ckpt_extension_is_pytorch(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "checkpoint.ckpt", ZIP_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PYTORCH

    def test_zip_with_zip_extension_is_zip_archive(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "archive.zip", ZIP_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.ZIP_ARCHIVE

    def test_zip_with_npz_extension_is_numpy_npz(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "arrays.npz", ZIP_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.NUMPY_NPZ

    def test_zip_with_unknown_extension_is_zip_archive(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "data.model", ZIP_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.ZIP_ARCHIVE


# ---------------------------------------------------------------------------
# Тесты detect() — остальные форматы
# ---------------------------------------------------------------------------


class TestDetectFormats:
    def test_numpy_npy(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "arr.npy", NUMPY_NPY_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.NUMPY_NPY

    def test_gguf(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "model.gguf", GGUF_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.GGUF

    def test_pickle_protocol2(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "data.pkl", PICKLE2_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PICKLE

    def test_pickle_protocol3(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "data.pkl", PICKLE3_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PICKLE

    def test_pickle_protocol4(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "data.pkl", PICKLE4_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PICKLE

    def test_pickle_protocol5(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "data.pkl", PICKLE5_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PICKLE

    def test_tar_archive(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "model.tar", TAR_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.TAR_ARCHIVE

    def test_safetensors(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "model.safetensors", SAFETENSORS_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.SAFETENSORS

    def test_unknown_file(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "data.bin", UNKNOWN_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.UNKNOWN

    def test_nonexistent_file_is_unknown(self, tmp_path: Path) -> None:
        p = tmp_path / "does_not_exist.pkl"
        assert FormatDetector.detect(p) == FileFormat.UNKNOWN


# ---------------------------------------------------------------------------
# Тесты detect() — Joblib
# ---------------------------------------------------------------------------


class TestDetectJoblib:
    def test_joblib_extension_with_pickle_magic(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "model.joblib", PICKLE2_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.JOBLIB

    def test_joblib_extension_any_pickle_protocol(self, tmp_path: Path) -> None:
        for magic in (PICKLE2_MAGIC, PICKLE3_MAGIC, PICKLE4_MAGIC, PICKLE5_MAGIC):
            p = write_tmp(tmp_path, "model.joblib", magic)
            assert FormatDetector.detect(p) == FileFormat.JOBLIB

    def test_pickle_extension_is_not_joblib(self, tmp_path: Path) -> None:
        p = write_tmp(tmp_path, "model.pkl", PICKLE2_MAGIC)
        assert FormatDetector.detect(p) == FileFormat.PICKLE


# ---------------------------------------------------------------------------
# Тесты detect_bytes() — идентичность с detect()
# ---------------------------------------------------------------------------


class TestDetectBytes:
    def test_numpy_npy_bytes(self) -> None:
        assert FormatDetector.detect_bytes(NUMPY_NPY_MAGIC, "arr.npy") == FileFormat.NUMPY_NPY

    def test_gguf_bytes(self) -> None:
        assert FormatDetector.detect_bytes(GGUF_MAGIC, "model.gguf") == FileFormat.GGUF

    def test_pickle_bytes(self) -> None:
        assert FormatDetector.detect_bytes(PICKLE2_MAGIC, "data.pkl") == FileFormat.PICKLE

    def test_safetensors_bytes(self) -> None:
        assert (
            FormatDetector.detect_bytes(SAFETENSORS_MAGIC, "model.safetensors")
            == FileFormat.SAFETENSORS
        )

    def test_tar_bytes(self) -> None:
        assert FormatDetector.detect_bytes(TAR_MAGIC, "archive.tar") == FileFormat.TAR_ARCHIVE

    def test_unknown_bytes(self) -> None:
        assert FormatDetector.detect_bytes(UNKNOWN_MAGIC) == FileFormat.UNKNOWN

    def test_detect_bytes_matches_detect_for_all_formats(self, tmp_path: Path) -> None:
        """detect_bytes с тем же файлом и именем даёт тот же результат, что detect."""
        cases = [
            ("arr.npy", NUMPY_NPY_MAGIC),
            ("model.gguf", GGUF_MAGIC),
            ("data.pkl", PICKLE2_MAGIC),
            ("model.joblib", PICKLE3_MAGIC),
            ("model.pt", ZIP_MAGIC),
            ("archive.zip", ZIP_MAGIC),
            ("arrays.npz", ZIP_MAGIC),
            ("model.safetensors", SAFETENSORS_MAGIC),
            ("archive.tar", TAR_MAGIC),
            ("data.bin", UNKNOWN_MAGIC),
        ]
        for name, data in cases:
            p = write_tmp(tmp_path, name, data)
            via_path = FormatDetector.detect(p)
            via_bytes = FormatDetector.detect_bytes(data, name)
            assert via_path == via_bytes, (
                f"Расхождение для {name!r}: detect={via_path}, detect_bytes={via_bytes}"
            )

    def test_detect_bytes_without_filename_zip_is_zip(self) -> None:
        """Без имени файла ZIP возвращается как ZIP_ARCHIVE, а не PYTORCH."""
        assert FormatDetector.detect_bytes(ZIP_MAGIC) == FileFormat.ZIP_ARCHIVE

    def test_detect_bytes_without_filename_pickle_is_pickle(self) -> None:
        assert FormatDetector.detect_bytes(PICKLE2_MAGIC) == FileFormat.PICKLE

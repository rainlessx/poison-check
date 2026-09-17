"""Определение формата ML-файлов по magic bytes."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileFormat(Enum):
    """Поддерживаемые форматы ML-файлов и архивов."""

    PICKLE = "pickle"
    PYTORCH = "pytorch"
    NUMPY_NPY = "numpy_npy"
    NUMPY_NPZ = "numpy_npz"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    JOBLIB = "joblib"
    ONNX = "onnx"
    KERAS_H5 = "keras_h5"
    ZIP_ARCHIVE = "zip_archive"
    TAR_ARCHIVE = "tar_archive"
    UNKNOWN = "unknown"


@dataclass
class FormatSignature:
    """Magic bytes сигнатура одного формата файла."""

    format: FileFormat
    magic: bytes
    offset: int = 0
    description: str = ""


@dataclass(frozen=True)
class FormatEvidence:
    """Формат, определённый ТОЛЬКО по содержимому, и признак его определения.

    Отличие от :meth:`FormatDetector.detect`: там расширение файла участвует в
    разрешении неоднозначностей (ZIP → PyTorch/npz, pickle → joblib). Здесь
    расширение не учитывается вовсе — это нужно там, где расширению доверять
    нельзя (сверка «расширение ↔ фактический формат», MLS-FMT-001).

    :ivar format: Формат, определённый по байтам, или ``UNKNOWN``.
    :ivar basis: Сырой признак, по которому формат определён (magic-байты и их
        смещение либо структурный признак). Пишется в отчёт как есть —
        forensic-инвариант, ничего не нормализуется.
    """

    format: FileFormat
    basis: str


# ---------------------------------------------------------------------------
# Вспомогательные константы
# ---------------------------------------------------------------------------

_PICKLE_SECOND_BYTES: frozenset[bytes] = frozenset(
    [b"\x02", b"\x03", b"\x04", b"\x05"]
)

_PYTORCH_EXTENSIONS: frozenset[str] = frozenset([".pt", ".pth", ".bin", ".ckpt"])

_SAFETENSORS_MAX_HEADER: int = 100 * 1024 * 1024  # 100 МБ — разумный лимит

#: Сигнатура HDF5-контейнера (Keras .h5). Совпадает с ``keras_scanner._HDF5_MAGIC``;
#: продублирована здесь потому, что core не имеет права импортировать scanners.
_HDF5_MAGIC: bytes = b"\x89HDF\r\n\x1a\n"


class FormatDetector:
    """Определяет формат файла по magic bytes без загрузки его в память.

    Читает не более _HEADER_SIZE байт с начала файла.
    Никогда не вызывает pickle.load() / torch.load() / joblib.load().
    """

    _HEADER_SIZE: int = 512

    SIGNATURES: list[FormatSignature] = [
        FormatSignature(
            format=FileFormat.GGUF,
            magic=b"GGUF",
            offset=0,
            description="GGUF (llama.cpp / Ollama)",
        ),
        FormatSignature(
            format=FileFormat.NUMPY_NPY,
            magic=b"\x93NUMPY",
            offset=0,
            description="NumPy .npy массив",
        ),
        FormatSignature(
            format=FileFormat.ZIP_ARCHIVE,
            magic=b"\x50\x4b\x03\x04",
            offset=0,
            description="ZIP-архив (PyTorch .pt/.pth, NumPy .npz, generic ZIP)",
        ),
        FormatSignature(
            format=FileFormat.PICKLE,
            magic=b"\x80\x02",
            offset=0,
            description="Pickle протокол 2",
        ),
        FormatSignature(
            format=FileFormat.PICKLE,
            magic=b"\x80\x03",
            offset=0,
            description="Pickle протокол 3",
        ),
        FormatSignature(
            format=FileFormat.PICKLE,
            magic=b"\x80\x04",
            offset=0,
            description="Pickle протокол 4",
        ),
        FormatSignature(
            format=FileFormat.PICKLE,
            magic=b"\x80\x05",
            offset=0,
            description="Pickle протокол 5",
        ),
        FormatSignature(
            format=FileFormat.KERAS_H5,
            magic=b"\x89HDF\r\n\x1a\n",
            offset=0,
            description="HDF5-контейнер (Keras .h5 / TensorFlow)",
        ),
        FormatSignature(
            format=FileFormat.TAR_ARCHIVE,
            magic=b"ustar",
            offset=257,
            description="TAR-архив (POSIX ustar заголовок)",
        ),
    ]

    @classmethod
    def detect(cls, path: Path) -> FileFormat:
        """Определяет формат файла по magic bytes.

        Читает только первые _HEADER_SIZE байт — не загружает файл целиком.
        Возвращает UNKNOWN при ошибке чтения.
        """
        try:
            with path.open("rb") as f:
                header = f.read(cls._HEADER_SIZE)
        except OSError:
            return FileFormat.UNKNOWN
        return cls._detect(header, path.suffix.lower())

    @classmethod
    def detect_bytes(cls, data: bytes, filename: str = "") -> FileFormat:
        """Определяет формат по байтам (для вложенных файлов внутри архивов).

        filename — необязательное имя файла для определения расширения.
        """
        header = data[: cls._HEADER_SIZE]
        ext = Path(filename).suffix.lower() if filename else ""
        return cls._detect(header, ext)

    @classmethod
    def detect_content(cls, path: Path) -> FormatEvidence:
        """Определяет формат ТОЛЬКО по содержимому файла, игнорируя расширение.

        Нужен там, где расширению доверять нельзя: сверка «расширение ↔
        фактический формат» (``strict_format_detection``, MLS-FMT-001) и
        классификация формата как code-bearing (``require_safetensors``,
        MLS-FMT-002). Логика определения не дублируется — :meth:`_detect`
        построен поверх той же функции :meth:`_detect_content`.

        Читает только первые ``_HEADER_SIZE`` байт. При ошибке чтения
        возвращает ``UNKNOWN`` с пустым признаком (файл не потерян: его
        обрабатывает штатный путь ошибок сканера).

        :param path: Путь к файлу.
        :return: Формат по содержимому и сырой признак, по которому он определён.
        """
        try:
            with path.open("rb") as f:
                header = f.read(cls._HEADER_SIZE)
        except OSError:
            return FormatEvidence(format=FileFormat.UNKNOWN, basis="")
        return cls._detect_content(header)

    @classmethod
    def detect_content_bytes(cls, data: bytes) -> FormatEvidence:
        """Определяет формат по байтам без обращения к диску и без расширения.

        :param data: Начало файла (достаточно ``_HEADER_SIZE`` байт).
        :return: Формат по содержимому и сырой признак его определения.
        """
        return cls._detect_content(data[: cls._HEADER_SIZE])

    @classmethod
    def _detect_content(cls, header: bytes) -> FormatEvidence:
        """Определяет формат по magic bytes / структуре заголовка.

        Единственное место, где содержимое сопоставляется с сигнатурами.
        Расширение здесь не участвует: неоднозначности (ZIP → PyTorch/npz,
        pickle → joblib) разрешает :meth:`_detect` поверх этого результата.
        """
        # TAR: magic bytes расположены со смещением 257 байт
        if len(header) >= 262 and header[257:262] == b"ustar":
            return FormatEvidence(
                format=FileFormat.TAR_ARCHIVE,
                basis="magic-байты b'ustar' по смещению 257",
            )

        # GGUF
        if header[:4] == b"GGUF":
            return FormatEvidence(
                format=FileFormat.GGUF,
                basis="magic-байты b'GGUF' по смещению 0",
            )

        # NumPy .npy
        if header[:6] == b"\x93NUMPY":
            return FormatEvidence(
                format=FileFormat.NUMPY_NPY,
                basis=r"magic-байты b'\x93NUMPY' по смещению 0",
            )

        # HDF5 — контейнер Keras .h5 / TensorFlow
        if header[: len(_HDF5_MAGIC)] == _HDF5_MAGIC:
            return FormatEvidence(
                format=FileFormat.KERAS_H5,
                basis=r"magic-байты b'\x89HDF\r\n\x1a\n' по смещению 0",
            )

        # ZIP-контейнер: PyTorch, NumPy .npz, Keras .keras или generic ZIP
        if header[:4] == b"\x50\x4b\x03\x04":
            return FormatEvidence(
                format=FileFormat.ZIP_ARCHIVE,
                basis=r"magic-байты b'PK\x03\x04' по смещению 0",
            )

        # Pickle протоколы 2–5 (опкод PROTO + номер протокола).
        if len(header) >= 2 and header[:1] == b"\x80" and header[1:2] in _PICKLE_SECOND_BYTES:
            return FormatEvidence(
                format=FileFormat.PICKLE,
                basis=(
                    f"опкод PROTO (0x80) + номер протокола {header[1]} "
                    f"по смещению 0"
                ),
            )

        # SafeTensors: первые 8 байт — uint64 LE (длина JSON-заголовка), затем '{'
        # Проверяется после Pickle, чтобы избежать коллизий на малых значениях uint64.
        if len(header) >= 9:
            (header_len,) = struct.unpack_from("<Q", header[:8])
            if 0 < header_len < _SAFETENSORS_MAX_HEADER and header[8:9] == b"{":
                return FormatEvidence(
                    format=FileFormat.SAFETENSORS,
                    basis=(
                        f"uint64 LE длина JSON-заголовка ({header_len}) "
                        f"и символ '{{' по смещению 8"
                    ),
                )

        return FormatEvidence(format=FileFormat.UNKNOWN, basis="")

    @classmethod
    def _detect(cls, header: bytes, ext: str) -> FileFormat:
        """Определяет формат по байтам, разрешая неоднозначности расширением.

        Контейнерные и pickle-подобные форматы неотличимы по magic bytes
        (``.pt``/``.npz``/``.keras`` — все ZIP; joblib без сжатия — обычный
        pickle), поэтому здесь и только здесь к результату
        :meth:`_detect_content` применяется расширение.
        """
        detected = cls._detect_content(header).format

        # ZIP-контейнер: PyTorch, NumPy .npz или generic ZIP
        if detected is FileFormat.ZIP_ARCHIVE:
            if ext in _PYTORCH_EXTENSIONS:
                return FileFormat.PYTORCH
            if ext == ".npz":
                return FileFormat.NUMPY_NPZ
            return FileFormat.ZIP_ARCHIVE

        # Joblib сохраняет pickle внутри, поэтому magic bytes совпадают —
        # отличаем только по расширению файла.
        if detected is FileFormat.PICKLE and ext == ".joblib":
            return FileFormat.JOBLIB

        # Joblib по расширению — запасной вариант, если magic bytes не совпали
        # (сжатые joblib-файлы начинаются с заголовка кодека zlib/lz4/zstd).
        if detected is FileFormat.UNKNOWN and ext == ".joblib":
            return FileFormat.JOBLIB

        return detected

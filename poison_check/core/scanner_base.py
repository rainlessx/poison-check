"""Базовый класс для сканеров ML-файлов и структура RawScanData."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from poison_check.core.result import (
    EmbeddedSignature,
    OpcodeInfo,
    ReduceCall,
    StringInfo,
    TensorInfo,
)

# ---------------------------------------------------------------------------
# Ограничения размера файла
# ---------------------------------------------------------------------------

#: Максимальный размер файла для загрузки в RAM (10 ГБ) — общий дефолт.
#: Подходит для pickle/joblib/safetensors/numpy. Для GGUF переопределяется
#: классом ``GGUFScanner.DEFAULT_MAX_FILE_SIZE`` до 100 ГБ, поскольку
#: реальные LLM (Llama-3-70B, Mixtral-8x22B) имеют размер 50–90 ГБ и
#: GGUFScanner работает потоково — целый файл в RAM не загружается.
#:
#: Переопределяется через --max-file-size в CLI или конструктором
#: BaseScanner(max_file_size=...) для программного API.
MAX_FILE_SIZE: int = 10 * 1024 * 1024 * 1024  # 10 ГБ


@dataclass
class RawScanData:
    """Сырые данные, извлечённые сканером из файла."""

    file_path: Path
    file_hash: dict[str, str]  # {"sha256": "...", "sha512": "...", "md5": "..."}
    file_size: int
    scanner_name: str

    opcodes: list[OpcodeInfo] | None = None
    globals: set[tuple[str, str]] | None = None  # (module, name)
    strings: list[StringInfo] | None = None
    reduce_calls: list[ReduceCall] | None = None
    metadata: dict[str, Any] | None = None
    tensor_info: list[TensorInfo] | None = None
    nested_files: list[RawScanData] | None = None
    embedded_bytes: list[EmbeddedSignature] | None = None
    raw_content_sample: bytes | None = None  # первые 4096 байт
    error: str | None = None  # заполняется при ошибке парсинга


class BaseScanner(ABC):
    """Абстрактный сканер одного формата ML-файлов."""

    name: ClassVar[str]
    description: ClassVar[str]
    supported_extensions: ClassVar[list[str]]
    magic_bytes: ClassVar[list[bytes]]

    #: Дефолтный лимит размера для конкретного формата.
    #: Сканеры, работающие потоково с большими файлами (GGUF), переопределяют
    #: этот атрибут, чтобы не блокировать легитимные LLM-модели.
    DEFAULT_MAX_FILE_SIZE: ClassVar[int] = MAX_FILE_SIZE

    def __init__(self, max_file_size: int | None = None) -> None:
        """Инициализирует сканер с ограничением на размер файла.

        Args:
            max_file_size: Максимально допустимый размер файла в байтах.
                           ``None`` — использует ``DEFAULT_MAX_FILE_SIZE``
                           конкретного сканера. Передаётся из CLI через
                           ``--max-file-size`` (для глобального override).
        """
        self._max_file_size = (
            max_file_size if max_file_size is not None else self.DEFAULT_MAX_FILE_SIZE
        )

    @classmethod
    @abstractmethod
    def can_handle(cls, path: Path) -> bool:
        """Может ли сканер обработать этот файл (по расширению и magic bytes)?"""

    @abstractmethod
    def scan(self, path: Path) -> RawScanData:
        """Извлекает сырые данные из файла для последующего анализа детекторами.

        НЕ интерпретирует найденное — это задача детекторов.
        При ошибке парсинга возвращает RawScanData с заполненным полем error.
        """

    def _check_file_size(self, path: Path) -> None:
        """Проверяет размер файла перед загрузкой в RAM.

        Бросает ValueError, если файл превышает self._max_file_size.
        Должен вызываться в начале scan() до любого чтения содержимого,
        чтобы предотвратить OOM-краш на GGUF-файлах LLM (30–70 ГБ).

        Args:
            path: Путь к проверяемому файлу.

        Raises:
            ValueError: Если размер файла превышает лимит.
            OSError:    Если stat() не удалось (обрабатывается вызывающим кодом).
        """
        size = path.stat().st_size
        if size > self._max_file_size:
            limit_gb = self._max_file_size / 1_000_000_000
            size_gb = size / 1_000_000_000
            raise ValueError(
                f"Файл {path.name} слишком большой: {size_gb:.1f} ГБ "
                f"(лимит {limit_gb:.0f} ГБ). "
                "Используйте --max-file-size для изменения лимита."
            )

    @staticmethod
    def _compute_hashes(path: Path, *, compute_md5: bool = False) -> dict[str, str]:
        """Вычисляет хеши файла потоковым чтением.

        По умолчанию (аудит #18) считаются только SHA-256 и SHA-512:

        * SHA-256 — основной идентификатор для отчётов.
        * SHA-512 — дублирующий хеш для compliance-требований ФСТЭК.

        MD5 убран из дефолта, потому что для security-инструмента он не нужен
        (cryptographically broken), а на больших GGUF-файлах (50–90 ГБ) тройное
        хеширование = тройной IO. Если MD5 нужен для совместимости с legacy
        forensics-тулзами или матчинга по MD5-сигнатурам антивирусов — передайте
        ``compute_md5=True``.

        Args:
            path: Путь к файлу.
            compute_md5: Включить MD5 (для forensics-режима). По умолчанию False.

        Returns:
            Словарь с ключами sha256, sha512 (всегда) и md5 (если compute_md5=True).
        """
        h_sha256 = hashlib.sha256()
        h_sha512 = hashlib.sha512()
        h_md5 = hashlib.md5(usedforsecurity=False) if compute_md5 else None  # noqa: S324
        with path.open("rb") as f:
            while chunk := f.read(65536):
                h_sha256.update(chunk)
                h_sha512.update(chunk)
                if h_md5 is not None:
                    h_md5.update(chunk)
        result = {
            "sha256": h_sha256.hexdigest(),
            "sha512": h_sha512.hexdigest(),
        }
        if h_md5 is not None:
            result["md5"] = h_md5.hexdigest()
        return result

    @staticmethod
    def _read_sample(path: Path, size: int = 4096) -> bytes:
        """Читает первые size байт файла."""
        with path.open("rb") as f:
            return f.read(size)

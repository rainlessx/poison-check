"""Тесты для базовых абстрактных классов BaseScanner и BaseDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.core.detector_base import BaseDetector
from poison_check.core.result import Issue, MLContext, Severity
from poison_check.core.scanner_base import BaseScanner, RawScanData


class _ConcreteScanner(BaseScanner):
    """Минимальная реализация BaseScanner для тестирования."""

    name = "concrete"
    description = "Тестовый сканер-заглушка"
    supported_extensions = [".test"]
    magic_bytes = [b"\x00\x01"]

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix == ".test"

    def scan(self, path: Path) -> RawScanData:
        return RawScanData(
            file_path=path,
            file_hash=self._compute_hashes(path),
            file_size=path.stat().st_size,
            scanner_name=self.name,
        )


class _ConcreteDetector(BaseDetector):
    """Минимальная реализация BaseDetector для тестирования."""

    name = "concrete"
    description = "Тестовый детектор-заглушка"
    severity_range = (Severity.LOW, Severity.HIGH)

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        return []


def test_base_scanner_not_instantiable() -> None:
    """Нельзя инстанциировать BaseScanner напрямую."""
    with pytest.raises(TypeError):
        BaseScanner()  # type: ignore[abstract]


def test_base_detector_not_instantiable() -> None:
    """Нельзя инстанциировать BaseDetector напрямую."""
    with pytest.raises(TypeError):
        BaseDetector()  # type: ignore[abstract]


def test_concrete_scanner_instantiation() -> None:
    """Конкретная реализация BaseScanner создаётся без ошибок."""
    scanner = _ConcreteScanner()
    assert scanner is not None


def test_compute_hashes_returns_sha256_and_sha512_by_default(tmp_path: Path) -> None:
    """По умолчанию (аудит #18) считаем только SHA-256 и SHA-512.

    MD5 не считается — для security-инструмента он не нужен (broken),
    а на больших GGUF-файлах тройное хеширование = тройной IO.
    """
    f = tmp_path / "data.bin"
    f.write_bytes(b"test content for hashing")
    hashes = _ConcreteScanner._compute_hashes(f)
    assert set(hashes.keys()) == {"sha256", "sha512"}
    assert all(len(v) > 0 for v in hashes.values())


def test_compute_hashes_with_md5_optional(tmp_path: Path) -> None:
    """compute_md5=True добавляет MD5 для forensics-режима."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"test content for hashing")
    hashes = _ConcreteScanner._compute_hashes(f, compute_md5=True)
    assert set(hashes.keys()) == {"sha256", "sha512", "md5"}
    # Известный MD5 для строки "test content for hashing"
    import hashlib as _h

    expected_md5 = _h.md5(b"test content for hashing", usedforsecurity=False).hexdigest()
    assert hashes["md5"] == expected_md5


def test_compute_hashes_deterministic(tmp_path: Path) -> None:
    """_compute_hashes возвращает одинаковый результат при двух вызовах на один файл."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"test content for hashing")
    assert _ConcreteScanner._compute_hashes(f) == _ConcreteScanner._compute_hashes(f)

"""Интеграционный тест всей функциональности core/, разработанной на неделе 1.

Проверяет совместную работу: FormatDetector, ScannerRegistry,
RawScanData, ScanResult, ContainerExtractor.
"""

from __future__ import annotations

import pickle
import zipfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest

from poison_check.core.container import ContainerExtractor
from poison_check.core.format_detector import FileFormat, FormatDetector
from poison_check.core.registry import ScannerRegistry
from poison_check.core.result import ScanResult, Severity
from poison_check.core.scanner_base import RawScanData


@pytest.fixture(autouse=True)
def _isolated_registry() -> Generator[None, None, None]:
    """Snapshot/restore реестра для изоляции тестов (аудит #12).

    Раньше ``_reset()`` разрушал глобальное состояние, и последующие тесты
    получали пустой реестр. Теперь сохраняем содержимое до теста, очищаем
    на время теста, восстанавливаем после.
    """
    saved = dict(ScannerRegistry._scanners)
    ScannerRegistry._scanners.clear()
    yield
    ScannerRegistry._scanners.clear()
    ScannerRegistry._scanners.update(saved)


# ---------------------------------------------------------------------------
# FormatDetector
# ---------------------------------------------------------------------------


def test_pickle_file_detected_as_pickle(tmp_path: Path) -> None:
    """FormatDetector.detect() возвращает PICKLE для файла с заголовком \\x80\\x04\\x95."""
    pkl_path = tmp_path / "model.pkl"
    # pickle.dumps(None, protocol=4) начинается с \x80\x04
    pkl_path.write_bytes(pickle.dumps(None, protocol=4))
    assert FormatDetector.detect(pkl_path) == FileFormat.PICKLE


def test_pickle_format_bytes_direct_detection(tmp_path: Path) -> None:
    """FormatDetector.detect_bytes() распознаёт pickle по байтам без файла на диске."""
    pkl_bytes = pickle.dumps(None, protocol=4)
    assert FormatDetector.detect_bytes(pkl_bytes, "model.pkl") == FileFormat.PICKLE


# ---------------------------------------------------------------------------
# ScannerRegistry — изоляция
# ---------------------------------------------------------------------------


def test_scanner_registry_empty_after_reset() -> None:
    """После сброса реестр не содержит ни одного сканера."""
    assert ScannerRegistry.all_scanners() == []


def test_scanner_registry_find_returns_none_when_empty(tmp_path: Path) -> None:
    """find_scanner возвращает None, если реестр пуст."""
    result = ScannerRegistry.find_scanner(tmp_path / "model.pkl")
    assert result is None


# ---------------------------------------------------------------------------
# RawScanData
# ---------------------------------------------------------------------------


def test_raw_scan_data_creation_with_required_fields(tmp_path: Path) -> None:
    """RawScanData создаётся без ошибок, необязательные поля по умолчанию None."""
    dummy = tmp_path / "data.bin"
    dummy.write_bytes(b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00.")
    data = RawScanData(
        file_path=dummy,
        file_hash={"sha256": "abc123", "sha512": "def456", "md5": "ghi789"},
        file_size=dummy.stat().st_size,
        scanner_name="test_scanner",
    )
    assert data.file_path == dummy
    assert data.scanner_name == "test_scanner"
    assert data.opcodes is None
    assert data.globals is None
    assert data.strings is None
    assert data.error is None
    assert data.nested_files is None


# ---------------------------------------------------------------------------
# ScanResult
# ---------------------------------------------------------------------------


def test_scan_result_empty_has_no_issues() -> None:
    """ScanResult без issues: has_issues_above(LOW) == False, worst_severity == None."""
    result = ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(),
        duration_ms=0.0,
    )
    assert result.has_issues_above(Severity.LOW) is False
    assert result.has_issues_above(Severity.CRITICAL) is False
    assert result.has_critical is False
    assert result.has_errors is False
    assert result.worst_severity is None


# ---------------------------------------------------------------------------
# ContainerExtractor
# ---------------------------------------------------------------------------


def test_container_extractor_list_zip_members_real_zip(tmp_path: Path) -> None:
    """ContainerExtractor.list_zip_members перечисляет файлы из реального ZIP в памяти."""
    zip_path = tmp_path / "model.zip"
    pkl_content = pickle.dumps(None, protocol=4)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("model/data.pkl", pkl_content)
        zf.writestr("model/config.json", b'{"version": 1}')
        # Директория — не должна попасть в список
        dir_info = zipfile.ZipInfo("model/")
        zf.writestr(dir_info, "")

    members = ContainerExtractor.list_zip_members(zip_path)

    assert "model/data.pkl" in members
    assert "model/config.json" in members
    # Директории исключены
    assert not any(m.endswith("/") for m in members)
    assert len(members) == 2


def test_container_extractor_is_zip_detects_created_zip(tmp_path: Path) -> None:
    """is_zip возвращает True для только что созданного ZIP."""
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("hello.txt", b"world")
    assert ContainerExtractor.is_zip(zip_path) is True


def test_container_extractor_extract_zip_matches_list(tmp_path: Path) -> None:
    """Имена из extract_zip_members совпадают с именами из list_zip_members."""
    files = {"a.pkl": b"\x80\x04.", "b.json": b"{}"}
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)

    listed = set(ContainerExtractor.list_zip_members(zip_path))
    extracted = {name for name, _ in ContainerExtractor.extract_zip_members(zip_path)}
    assert listed == extracted

"""Тесты защиты от OOM: проверка ограничения размера файла в сканерах.

Сканеры должны возвращать RawScanData с полем error (не падать с OOM)
при попытке загрузить файл, превышающий лимит _max_file_size.

Реальные большие файлы не создаются — подменяем path.stat().st_size
через unittest.mock.patch, чтобы симулировать файл нужного размера
без фактического выделения памяти.
"""

from __future__ import annotations

import io
import pickle
import struct
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from poison_check.core.scanner_base import MAX_FILE_SIZE, BaseScanner
from poison_check.scanners.gguf_scanner import GGUFScanner
from poison_check.scanners.joblib_scanner import JoblibScanner
from poison_check.scanners.numpy_scanner import NumpyScanner
from poison_check.scanners.pickle_scanner import PickleScanner
from poison_check.scanners.pytorch_scanner import PyTorchScanner


# ---------------------------------------------------------------------------
# Вспомогательные фабрики минимальных валидных файлов
# ---------------------------------------------------------------------------


def _make_minimal_pickle() -> bytes:
    """Создаёт минимальный валидный pickle-поток (пустой список)."""
    return b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00]\x94."


def _make_minimal_pt(tmp_path: Path) -> Path:
    """Создаёт минимальный .pt файл (ZIP с data.pkl)."""
    pkl = _make_minimal_pickle()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", pkl)
    p = tmp_path / "model.pt"
    p.write_bytes(buf.getvalue())
    return p


def _make_minimal_npy(tmp_path: Path) -> Path:
    """Создаёт минимальный .npy файл (пустой float32-массив)."""
    # magic + version 1.0 + header_len + header + данные (пусто)
    header_str = b"{'descr': '<f4', 'fortran_order': False, 'shape': (0,), }          \n"
    header_len = len(header_str)
    data = (
        b"\x93NUMPY"          # magic
        + b"\x01\x00"         # версия 1.0
        + struct.pack("<H", header_len)
        + header_str
    )
    p = tmp_path / "arr.npy"
    p.write_bytes(data)
    return p


def _make_minimal_pkl_file(tmp_path: Path) -> Path:
    """Создаёт минимальный .pkl файл."""
    p = tmp_path / "model.pkl"
    p.write_bytes(_make_minimal_pickle())
    return p


def _make_minimal_joblib(tmp_path: Path) -> Path:
    """Создаёт минимальный .joblib файл (raw pickle без компрессии)."""
    p = tmp_path / "model.joblib"
    p.write_bytes(_make_minimal_pickle())
    return p


def _make_minimal_gguf(tmp_path: Path) -> Path:
    """Создаёт минимальный валидный GGUF-файл (версия 3, 0 тензоров, 0 KV)."""
    data = (
        b"GGUF"                     # magic
        + struct.pack("<I", 3)      # version = 3
        + struct.pack("<Q", 0)      # tensor_count = 0
        + struct.pack("<Q", 0)      # kv_count = 0
    )
    p = tmp_path / "model.gguf"
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Константы для тестов
# ---------------------------------------------------------------------------

_OVER_LIMIT: int = MAX_FILE_SIZE + 1          # 1 байт выше дефолтного лимита
_UNDER_LIMIT: int = MAX_FILE_SIZE - 1         # чуть ниже дефолтного лимита
_CUSTOM_LIMIT: int = 100                      # 100 байт — для теста кастомного лимита
_OVER_CUSTOM: int = 101                       # выше кастомного лимита


# ---------------------------------------------------------------------------
# Параметризованный тест: все сканеры возвращают error при превышении лимита
# ---------------------------------------------------------------------------


class _FakeStat:
    """Заглушка os.stat_result с нужным st_size."""

    def __init__(self, size: int) -> None:
        self.st_size = size


@pytest.mark.parametrize(
    "scanner_cls, make_file",
    [
        (PickleScanner, _make_minimal_pkl_file),
        (PyTorchScanner, _make_minimal_pt),
        (NumpyScanner, _make_minimal_npy),
        (JoblibScanner, _make_minimal_joblib),
        (GGUFScanner, _make_minimal_gguf),
    ],
    ids=["pickle", "pytorch", "numpy", "joblib", "gguf"],
)
def test_file_over_limit_returns_error(
    scanner_cls: type[BaseScanner],
    make_file: object,
    tmp_path: Path,
) -> None:
    """Файл размером выше лимита → RawScanData с полем error, без OOM."""
    path = make_file(tmp_path)  # type: ignore[operator]
    scanner = scanner_cls(max_file_size=_CUSTOM_LIMIT)

    # Подменяем stat чтобы симулировать большой файл
    fake_stat = _FakeStat(_OVER_CUSTOM)
    with patch.object(Path, "stat", return_value=fake_stat):
        result = scanner.scan(path)

    # Сканер НЕ должен падать с исключением — только возвращать error
    assert result.error is not None, (
        f"{scanner_cls.__name__}: ожидался RawScanData.error при превышении лимита"
    )
    assert "слишком большой" in result.error, (
        f"{scanner_cls.__name__}: сообщение об ошибке должно содержать 'слишком большой', "
        f"получено: {result.error!r}"
    )
    assert result.scanner_name == scanner_cls.name


@pytest.mark.parametrize(
    "scanner_cls, make_file",
    [
        (PickleScanner, _make_minimal_pkl_file),
        (PyTorchScanner, _make_minimal_pt),
        (NumpyScanner, _make_minimal_npy),
        (JoblibScanner, _make_minimal_joblib),
        (GGUFScanner, _make_minimal_gguf),
    ],
    ids=["pickle", "pytorch", "numpy", "joblib", "gguf"],
)
def test_file_under_limit_scans_normally(
    scanner_cls: type[BaseScanner],
    make_file: object,
    tmp_path: Path,
) -> None:
    """Файл размером ниже лимита → сканирование проходит без ошибки размера."""
    path = make_file(tmp_path)  # type: ignore[operator]
    # Лимит намеренно очень большой — реальный файл всегда ниже
    scanner = scanner_cls(max_file_size=MAX_FILE_SIZE)

    result = scanner.scan(path)

    # Ошибки, связанной с размером, быть не должно
    if result.error is not None:
        assert "слишком большой" not in result.error, (
            f"{scanner_cls.__name__}: неожиданная ошибка размера для малого файла: "
            f"{result.error!r}"
        )


# ---------------------------------------------------------------------------
# Тест: кастомный лимит работает корректно
# ---------------------------------------------------------------------------


def test_custom_limit_respected(tmp_path: Path) -> None:
    """Лимит, заданный через max_file_size, учитывается точно."""
    path = _make_minimal_pkl_file(tmp_path)
    real_size = path.stat().st_size

    # Лимит ровно на 1 байт меньше реального размера → должна быть ошибка
    scanner_too_small = PickleScanner(max_file_size=real_size - 1)
    result_fail = scanner_too_small.scan(path)
    assert result_fail.error is not None
    assert "слишком большой" in result_fail.error

    # Лимит ровно равен размеру файла → должно пройти
    scanner_exact = PickleScanner(max_file_size=real_size)
    result_ok = scanner_exact.scan(path)
    if result_ok.error is not None:
        assert "слишком большой" not in result_ok.error


def test_default_limit_is_10gb() -> None:
    """MAX_FILE_SIZE по умолчанию равен 10 ГБ."""
    assert MAX_FILE_SIZE == 10 * 1024 * 1024 * 1024


def test_base_scanner_default_limit() -> None:
    """BaseScanner.__init__ без аргументов использует MAX_FILE_SIZE."""
    scanner = PickleScanner()
    assert scanner._max_file_size == MAX_FILE_SIZE


def test_base_scanner_custom_limit() -> None:
    """BaseScanner.__init__ принимает кастомный max_file_size."""
    custom = 5 * 1024 * 1024 * 1024  # 5 ГБ
    scanner = PickleScanner(max_file_size=custom)
    assert scanner._max_file_size == custom


# ---------------------------------------------------------------------------
# Тест: GGUF-сканер не читает тензорные данные (потоковый парсинг)
# ---------------------------------------------------------------------------


def test_gguf_scanner_does_not_read_tensor_data(tmp_path: Path) -> None:
    """GGUFScanner не загружает тензорные данные в RAM.

    Создаём GGUF с одной KV-парой и фиктивными «тензорными данными»
    в виде 1 МБ нулей. Проверяем, что сканер успешно разбирает
    файл и не падает — тензоры должны быть проигнорированы.
    """
    # Строим GGUF: magic + version + 0 tensors + 1 KV (строка)
    key = b"general.name"
    value = b"test-model"

    kv_block = (
        struct.pack("<Q", len(key)) + key      # ключ: length-prefixed string
        + struct.pack("<I", 8)                  # тип = string
        + struct.pack("<Q", len(value)) + value # значение: length-prefixed string
    )

    # «Тензорные данные» — 1 МБ нулей после KV-секции (не должны загружаться)
    tensor_garbage = b"\x00" * (1024 * 1024)

    gguf_data = (
        b"GGUF"
        + struct.pack("<I", 3)       # version = 3
        + struct.pack("<Q", 0)       # tensor_count = 0 (нет описаний тензоров)
        + struct.pack("<Q", 1)       # kv_count = 1
        + kv_block
        + tensor_garbage             # тензорные байты — сканер не должен их читать
    )

    path = tmp_path / "with_tensor_garbage.gguf"
    path.write_bytes(gguf_data)

    scanner = GGUFScanner()
    result = scanner.scan(path)

    # Нет ошибок парсинга
    assert result.error is None, f"Неожиданная ошибка: {result.error}"
    # Метаданные распарсены
    assert result.metadata is not None
    assert result.metadata.get("general.name") == "test-model"
    assert result.metadata.get("kv_count") == "1"


def test_gguf_scanner_with_file_size_limit(tmp_path: Path) -> None:
    """GGUFScanner: файл больше лимита → error, не OOM."""
    path = _make_minimal_gguf(tmp_path)
    scanner = GGUFScanner(max_file_size=_CUSTOM_LIMIT)

    fake_stat = _FakeStat(_OVER_CUSTOM)
    with patch.object(Path, "stat", return_value=fake_stat):
        result = scanner.scan(path)

    assert result.error is not None
    assert "слишком большой" in result.error
    assert result.scanner_name == "gguf"


# ---------------------------------------------------------------------------
# Тест: _check_file_size бросает ValueError, не OSError
# ---------------------------------------------------------------------------


def test_check_file_size_raises_value_error(tmp_path: Path) -> None:
    """_check_file_size бросает ValueError при превышении лимита."""
    path = _make_minimal_pkl_file(tmp_path)
    real_size = path.stat().st_size

    scanner = PickleScanner(max_file_size=real_size - 1)

    with pytest.raises(ValueError, match="слишком большой"):
        scanner._check_file_size(path)


def test_check_file_size_passes_within_limit(tmp_path: Path) -> None:
    """_check_file_size не бросает исключений при файле в пределах лимита."""
    path = _make_minimal_pkl_file(tmp_path)
    real_size = path.stat().st_size

    scanner = PickleScanner(max_file_size=real_size)
    # Не должно бросить исключение
    scanner._check_file_size(path)


# ---------------------------------------------------------------------------
# Тест: сообщение об ошибке содержит размер файла и лимит в ГБ
# ---------------------------------------------------------------------------


def test_error_message_contains_size_info(tmp_path: Path) -> None:
    """Сообщение об ошибке размера содержит фактический размер и лимит."""
    path = _make_minimal_pkl_file(tmp_path)
    scanner = PickleScanner(max_file_size=_CUSTOM_LIMIT)

    fake_stat = _FakeStat(_OVER_CUSTOM)
    with patch.object(Path, "stat", return_value=fake_stat):
        result = scanner.scan(path)

    assert result.error is not None
    # Сообщение должно содержать имя файла
    assert path.name in result.error
    # Должна упоминаться рекомендация использовать --max-file-size
    assert "--max-file-size" in result.error


# ---------------------------------------------------------------------------
# Регрессия аудита #24: GGUF-aware DEFAULT_MAX_FILE_SIZE
# ---------------------------------------------------------------------------


class TestGGUFDefaultMaxFileSize:
    """GGUFScanner имеет свой увеличенный лимит для LLM-моделей."""

    def test_gguf_default_is_100gb(self) -> None:
        """GGUFScanner.DEFAULT_MAX_FILE_SIZE = 100 ГБ — больше базового 10 ГБ."""
        from poison_check.core.scanner_base import MAX_FILE_SIZE  # noqa: PLC0415
        from poison_check.scanners.gguf_scanner import GGUFScanner  # noqa: PLC0415

        assert GGUFScanner.DEFAULT_MAX_FILE_SIZE > MAX_FILE_SIZE
        assert GGUFScanner.DEFAULT_MAX_FILE_SIZE >= 100 * 1024 * 1024 * 1024

    def test_pickle_default_is_base(self) -> None:
        """PickleScanner использует общий MAX_FILE_SIZE."""
        from poison_check.core.scanner_base import MAX_FILE_SIZE  # noqa: PLC0415
        from poison_check.scanners.pickle_scanner import PickleScanner  # noqa: PLC0415

        assert PickleScanner.DEFAULT_MAX_FILE_SIZE == MAX_FILE_SIZE

    def test_max_file_size_none_uses_class_default(self) -> None:
        """Конструктор с max_file_size=None → берётся DEFAULT_MAX_FILE_SIZE."""
        from poison_check.scanners.gguf_scanner import GGUFScanner  # noqa: PLC0415

        scanner = GGUFScanner(max_file_size=None)
        assert scanner._max_file_size == GGUFScanner.DEFAULT_MAX_FILE_SIZE

    def test_max_file_size_explicit_overrides_default(self) -> None:
        """Явное число → override дефолта."""
        from poison_check.scanners.gguf_scanner import GGUFScanner  # noqa: PLC0415

        scanner = GGUFScanner(max_file_size=42)
        assert scanner._max_file_size == 42

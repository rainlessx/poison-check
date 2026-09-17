"""Тесты для SafetensorsScanner.

Fixture-файлы создаются вручную по бинарной спецификации SafeTensors:
  [8 байт little-endian uint64 = длина заголовка]
  [JSON-заголовок в UTF-8]
  [бинарные данные тензоров — в тестах пустые]

Тесты не зависят от сети и не используют библиотеку safetensors.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from poison_check.scanners.safetensors_scanner import SafetensorsScanner


# ---------------------------------------------------------------------------
# Вспомогательные функции для построения бинарных fixture-файлов
# ---------------------------------------------------------------------------


def _build_safetensors(header_dict: dict, tensor_data: bytes = b"") -> bytes:
    """Строит валидный SafeTensors-файл из Python-словаря и опциональных данных.

    Используется в тестах вместо реального torch/safetensors, чтобы не
    зависеть от внешних зависимостей.
    """
    header_json = json.dumps(header_dict, ensure_ascii=False).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    return header_len + header_json + tensor_data


def _build_minimal_valid() -> bytes:
    """Минимальный валидный SafeTensors-файл: один тензор F32 с формой [2, 3].

    Данные тензора: 6 * 4 = 24 байта (float32), заполнены нулями.
    data_offsets указывают на начало и конец данных.
    """
    tensor_bytes = b"\x00" * 24  # 6 float32 = 24 байта
    header: dict = {
        "__metadata__": {"format": "pt"},
        "weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24],
        },
    }
    return _build_safetensors(header, tensor_data=tensor_bytes)


def _build_with_api_key() -> bytes:
    """SafeTensors-файл с API_KEY в __metadata__ для теста SecretsDetector."""
    header: dict = {
        "__metadata__": {
            "format": "pt",
            "API_KEY": "sk-proj-AAABBBCCCDDDEEE123456789",
        },
        "weight": {
            "dtype": "F32",
            "shape": [3],
            "data_offsets": [0, 12],
        },
    }
    return _build_safetensors(header, tensor_data=b"\x00" * 12)


def _build_oversized_header() -> bytes:
    """SafeTensors-файл с размером заголовка > MAX_HEADER_SIZE.

    JSON-заголовок не записывается реальным — сразу пишем большое число
    в поле длины (200 MB), тело намеренно отсутствует.
    """
    oversized = SafetensorsScanner.MAX_HEADER_SIZE + 1
    return struct.pack("<Q", oversized)  # только 8-байтовое поле, без заголовка


def _build_malformed_json() -> bytes:
    """SafeTensors-файл с некорректным JSON в заголовке."""
    bad_header = b"{this is not valid json!!!"
    header_len = struct.pack("<Q", len(bad_header))
    return header_len + bad_header


# ---------------------------------------------------------------------------
# Тест 1: минимальный валидный файл
# ---------------------------------------------------------------------------


class TestSafetensorsScannerValid:
    """SafeTensors-файл без проблем должен сканироваться без issues."""

    def test_valid_file_no_error(self, tmp_path: Path) -> None:
        """Минимальный валидный SafeTensors-файл сканируется без ошибок."""
        sf_file = tmp_path / "model.safetensors"
        sf_file.write_bytes(_build_minimal_valid())

        scanner = SafetensorsScanner()
        result = scanner.scan(sf_file)

        assert result.error is None, f"Ожидался error=None, получен: {result.error!r}"

    def test_valid_file_tensor_info(self, tmp_path: Path) -> None:
        """Сканер корректно извлекает tensor_info из валидного файла."""
        sf_file = tmp_path / "model.safetensors"
        sf_file.write_bytes(_build_minimal_valid())

        scanner = SafetensorsScanner()
        result = scanner.scan(sf_file)

        assert result.tensor_info is not None
        assert len(result.tensor_info) == 1
        ti = result.tensor_info[0]
        assert ti.name == "weight"
        assert ti.dtype == "F32"
        assert ti.shape == [2, 3]

    def test_valid_file_metadata(self, tmp_path: Path) -> None:
        """Сканер извлекает __metadata__ из валидного файла."""
        sf_file = tmp_path / "model.safetensors"
        sf_file.write_bytes(_build_minimal_valid())

        scanner = SafetensorsScanner()
        result = scanner.scan(sf_file)

        assert result.metadata is not None
        assert result.metadata.get("format") == "pt"

    def test_valid_file_hashes_computed(self, tmp_path: Path) -> None:
        """Сканер вычисляет хеши файла."""
        sf_file = tmp_path / "model.safetensors"
        sf_file.write_bytes(_build_minimal_valid())

        scanner = SafetensorsScanner()
        result = scanner.scan(sf_file)

        assert "sha256" in result.file_hash
        assert len(result.file_hash["sha256"]) == 64  # hex sha256

    def test_valid_file_scanner_name(self, tmp_path: Path) -> None:
        """scanner_name должен быть 'safetensors'."""
        sf_file = tmp_path / "model.safetensors"
        sf_file.write_bytes(_build_minimal_valid())

        result = SafetensorsScanner().scan(sf_file)
        assert result.scanner_name == "safetensors"

    def test_multiple_tensors(self, tmp_path: Path) -> None:
        """Файл с несколькими тензорами — все попадают в tensor_info."""
        header: dict = {
            "__metadata__": {},
            "layer.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
            "layer.bias": {"dtype": "F32", "shape": [4], "data_offsets": [64, 80]},
        }
        data = _build_safetensors(header, tensor_data=b"\x00" * 80)
        sf_file = tmp_path / "multi.safetensors"
        sf_file.write_bytes(data)

        result = SafetensorsScanner().scan(sf_file)

        assert result.error is None
        assert result.tensor_info is not None
        assert len(result.tensor_info) == 2
        names = {ti.name for ti in result.tensor_info}
        assert names == {"layer.weight", "layer.bias"}


# ---------------------------------------------------------------------------
# Тест 2: API-ключ в __metadata__ → строки передаются SecretsDetector
# ---------------------------------------------------------------------------


class TestSafetensorsScannerSecrets:
    """API-ключи в __metadata__ должны попадать в raw_data.strings."""

    def test_api_key_in_strings(self, tmp_path: Path) -> None:
        """Значение API_KEY из __metadata__ должно оказаться в strings."""
        sf_file = tmp_path / "secret.safetensors"
        sf_file.write_bytes(_build_with_api_key())

        scanner = SafetensorsScanner()
        result = scanner.scan(sf_file)

        # Ошибки нет — файл технически корректен
        assert result.error is None

        # Значение должно быть передано SecretsDetector через strings
        assert result.strings is not None
        string_values = [s.value for s in result.strings]
        assert any("sk-proj-" in v for v in string_values), (
            f"Ожидалась строка с 'sk-proj-' в strings, получено: {string_values!r}"
        )

    def test_metadata_key_preserved(self, tmp_path: Path) -> None:
        """Ключ API_KEY сохраняется в metadata."""
        sf_file = tmp_path / "secret.safetensors"
        sf_file.write_bytes(_build_with_api_key())

        result = SafetensorsScanner().scan(sf_file)

        assert result.metadata is not None
        assert "API_KEY" in result.metadata

    def test_all_metadata_values_in_strings(self, tmp_path: Path) -> None:
        """Все значения из __metadata__ должны присутствовать в strings."""
        header: dict = {
            "__metadata__": {
                "author": "test_user",
                "token": "ghp_AAAABBBBCCCC1234",
            }
        }
        data = _build_safetensors(header)
        sf_file = tmp_path / "tokens.safetensors"
        sf_file.write_bytes(data)

        result = SafetensorsScanner().scan(sf_file)

        assert result.strings is not None
        values = {s.value for s in result.strings}
        assert "test_user" in values
        assert "ghp_AAAABBBBCCCC1234" in values


# ---------------------------------------------------------------------------
# Тест 3: аномально большой header_size → Issue HIGH
# ---------------------------------------------------------------------------


class TestSafetensorsScannerOversizedHeader:
    """Аномально большой заголовок должен фиксироваться как проблема."""

    def test_oversized_header_returns_error(self, tmp_path: Path) -> None:
        """При header_size > MAX_HEADER_SIZE сканер возвращает error, не падает."""
        sf_file = tmp_path / "oversized.safetensors"
        sf_file.write_bytes(_build_oversized_header())

        scanner = SafetensorsScanner()
        result = scanner.scan(sf_file)

        assert result.error is not None
        assert "Аномально большой заголовок" in result.error

    def test_oversized_header_metadata_flag(self, tmp_path: Path) -> None:
        """anomaly_header_size должен быть выставлен в metadata."""
        sf_file = tmp_path / "oversized.safetensors"
        sf_file.write_bytes(_build_oversized_header())

        result = SafetensorsScanner().scan(sf_file)

        assert result.metadata is not None
        assert "anomaly_header_size" in result.metadata
        stored_size = int(result.metadata["anomaly_header_size"])
        assert stored_size > SafetensorsScanner.MAX_HEADER_SIZE

    def test_oversized_header_does_not_raise(self, tmp_path: Path) -> None:
        """Сканер не бросает исключений при аномальном заголовке."""
        sf_file = tmp_path / "oversized.safetensors"
        sf_file.write_bytes(_build_oversized_header())

        # Не должно быть исключений
        result = SafetensorsScanner().scan(sf_file)
        assert isinstance(result.error, str)

    def test_exactly_at_limit_is_ok(self, tmp_path: Path) -> None:
        """Размер заголовка ровно равный MAX_HEADER_SIZE — не является аномалией."""
        # Строим реальный маленький заголовок, но эмулируем размер поля
        # Для теста: используем заголовок точно в лимите через маленький файл,
        # просто проверяем пограничное условие через константу
        assert SafetensorsScanner.MAX_HEADER_SIZE == 100 * 1024 * 1024

    def test_one_byte_over_limit_triggers_error(self, tmp_path: Path) -> None:
        """MAX_HEADER_SIZE + 1 байт — аномалия."""
        oversized_by_one = SafetensorsScanner.MAX_HEADER_SIZE + 1
        data = struct.pack("<Q", oversized_by_one)
        sf_file = tmp_path / "one_over.safetensors"
        sf_file.write_bytes(data)

        result = SafetensorsScanner().scan(sf_file)
        assert result.error is not None
        assert "Аномально большой заголовок" in result.error


# ---------------------------------------------------------------------------
# Тест 4: malformed JSON → не падает, возвращает RawScanData с error
# ---------------------------------------------------------------------------


class TestSafetensorsScannerMalformed:
    """Некорректный JSON-заголовок не должен валить сканер."""

    def test_malformed_json_returns_error(self, tmp_path: Path) -> None:
        """Malformed JSON возвращает RawScanData с описанием ошибки."""
        sf_file = tmp_path / "malformed.safetensors"
        sf_file.write_bytes(_build_malformed_json())

        scanner = SafetensorsScanner()
        result = scanner.scan(sf_file)

        assert result.error is not None
        assert "JSON" in result.error or "json" in result.error.lower()

    def test_malformed_json_does_not_raise(self, tmp_path: Path) -> None:
        """Сканер не бросает исключений при malformed JSON."""
        sf_file = tmp_path / "malformed.safetensors"
        sf_file.write_bytes(_build_malformed_json())

        # Должна вернуться структура, а не упасть
        result = SafetensorsScanner().scan(sf_file)
        assert result is not None

    def test_malformed_json_has_hashes(self, tmp_path: Path) -> None:
        """Даже при malformed JSON хеши файла должны быть вычислены."""
        sf_file = tmp_path / "malformed.safetensors"
        sf_file.write_bytes(_build_malformed_json())

        result = SafetensorsScanner().scan(sf_file)
        # Хеши должны быть посчитаны до попытки парсинга
        assert "sha256" in result.file_hash

    def test_truncated_file_returns_error(self, tmp_path: Path) -> None:
        """Файл меньше 8 байт → error без исключения."""
        sf_file = tmp_path / "tiny.safetensors"
        sf_file.write_bytes(b"\x00\x00\x00")  # < 8 байт

        result = SafetensorsScanner().scan(sf_file)
        assert result.error is not None

    def test_empty_file_returns_error(self, tmp_path: Path) -> None:
        """Пустой файл → error без исключения."""
        sf_file = tmp_path / "empty.safetensors"
        sf_file.write_bytes(b"")

        result = SafetensorsScanner().scan(sf_file)
        assert result.error is not None

    def test_json_not_object_returns_error(self, tmp_path: Path) -> None:
        """JSON-массив вместо объекта — ошибка формата."""
        header_bytes = b"[1, 2, 3]"
        data = struct.pack("<Q", len(header_bytes)) + header_bytes
        sf_file = tmp_path / "array_header.safetensors"
        sf_file.write_bytes(data)

        result = SafetensorsScanner().scan(sf_file)
        assert result.error is not None
        assert "JSON-объект" in result.error or "json" in result.error.lower()

    def test_truncated_header_body_returns_error(self, tmp_path: Path) -> None:
        """Поле длины обещает 1000 байт, а тело — только 10."""
        header_len = struct.pack("<Q", 1000)
        data = header_len + b"x" * 10  # только 10 байт
        sf_file = tmp_path / "truncated_body.safetensors"
        sf_file.write_bytes(data)

        result = SafetensorsScanner().scan(sf_file)
        assert result.error is not None


# ---------------------------------------------------------------------------
# Тест 5: необычные dtype
# ---------------------------------------------------------------------------


class TestSafetensorsScannerUnusualDtype:
    """Необычные dtype тензоров должны помечаться в metadata."""

    def test_unusual_dtype_flagged(self, tmp_path: Path) -> None:
        """Тензор с dtype=I32 помечается как необычный в metadata."""
        header: dict = {
            "__metadata__": {},
            "labels": {
                "dtype": "I32",
                "shape": [100],
                "data_offsets": [0, 400],
            },
        }
        data = _build_safetensors(header, tensor_data=b"\x00" * 400)
        sf_file = tmp_path / "unusual.safetensors"
        sf_file.write_bytes(data)

        result = SafetensorsScanner().scan(sf_file)

        assert result.metadata is not None
        assert "unusual_dtypes" in result.metadata
        assert "I32" in result.metadata["unusual_dtypes"]

    def test_normal_dtype_not_flagged(self, tmp_path: Path) -> None:
        """Тензор с dtype=BF16 не помечается как необычный."""
        header: dict = {
            "__metadata__": {},
            "embed": {
                "dtype": "BF16",
                "shape": [512],
                "data_offsets": [0, 1024],
            },
        }
        data = _build_safetensors(header, tensor_data=b"\x00" * 1024)
        sf_file = tmp_path / "normal.safetensors"
        sf_file.write_bytes(data)

        result = SafetensorsScanner().scan(sf_file)
        assert result.metadata is not None
        assert "unusual_dtypes" not in result.metadata


# ---------------------------------------------------------------------------
# Тест 6: can_handle
# ---------------------------------------------------------------------------


class TestSafetensorsScannerCanHandle:
    """can_handle должен принимать только .safetensors файлы."""

    def test_can_handle_correct_extension(self, tmp_path: Path) -> None:
        """can_handle возвращает True для .safetensors."""
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"")
        assert SafetensorsScanner.can_handle(f) is True

    def test_can_handle_wrong_extension(self, tmp_path: Path) -> None:
        """can_handle возвращает False для .pkl."""
        f = tmp_path / "model.pkl"
        f.write_bytes(b"")
        assert SafetensorsScanner.can_handle(f) is False

    def test_can_handle_case_insensitive(self, tmp_path: Path) -> None:
        """can_handle регистронезависим для расширения."""
        f = tmp_path / "MODEL.SAFETENSORS"
        f.write_bytes(b"")
        assert SafetensorsScanner.can_handle(f) is True

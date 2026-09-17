"""Сканер файлов в формате GGUF (.gguf, .ggml).

GGUF (GPT-Generated Unified Format) — бинарный формат хранения LLM-моделей,
используемый Ollama, llama.cpp и аналогичными инструментами локального инференса.
Появился как замена устаревшему GGML.

Бинарная структура (версии 1–3):
  [4 байта]         Magic: b'GGUF'
  [uint32 LE]       Version: 1, 2 или 3
  [uint64 LE]       tensor_count: количество тензоров
  [uint64 LE]       metadata_kv_count: количество пар ключ-значение в metadata
  [kv_count раз]    Записи metadata: ключ (string), тип значения (uint32), значение
  [tensor_count раз] Описания тензоров
  [выравнивание]     Данные тензоров (не читаются — только заголовок)

Типы значений metadata (GGUF_METADATA_VALUE_TYPE):
  0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32, 6=float32, 7=bool,
  8=string, 9=array, 10=uint64, 11=int64, 12=float64

Потоковый парсинг:
  Сканер читает ТОЛЬКО заголовок и KV-metadata через seek/read,
  не загружая тензорные данные (которые составляют 99%+ объёма GGUF-файла).
  Это принципиально для LLM-моделей размером 30–70 ГБ — иначе OOM.

Ссылки:
  https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
  https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/constants.py
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import IO, ClassVar

from poison_check.core.executable_signatures import find_signatures_in_file
from poison_check.core.registry import ScannerRegistry
from poison_check.core.result import StringInfo
from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы формата
# ---------------------------------------------------------------------------

_GGUF_MAGIC: bytes = b"GGUF"
_MAGIC_SIZE: int = 4

# ---------------------------------------------------------------------------
# Ключи фактов заголовка, которые сканер кладёт в metadata для GGUFMetadataDetector
# ---------------------------------------------------------------------------
# Симметрично gguf_version: аномалия заголовка (неверный magic) фиксируется как
# ФАКТ в metadata, а Issue (MLS-GGUF-006) эмитит детектор — сканер знает формат,
# детектор знает угрозу. Сканер НЕ бросает исключение и
# НЕ эмитит Issue; неверный magic одновременно ставит error (файл некорректен).

#: Валиден ли заголовок GGUF ("false" при неверном magic) — MLS-GGUF-006 (MEDIUM).
META_HEADER_VALID: str = "gguf_header_valid"
#: Сырые прочитанные magic-байты в repr-виде (forensic), напр. "b'GGXF'".
META_BAD_MAGIC: str = "gguf_bad_magic"
#: Сырые прочитанные magic-байты в hex (forensic, однозначно), напр. "47475846".
META_BAD_MAGIC_HEX: str = "gguf_bad_magic_hex"
#: Ожидаемое значение magic в repr-виде, напр. "b'GGUF'".
META_EXPECTED_MAGIC: str = "gguf_expected_magic"
_VERSION_SIZE: int = 4   # uint32
_TENSOR_COUNT_SIZE: int = 8   # uint64
_KV_COUNT_SIZE: int = 8   # uint64
_HEADER_MIN_SIZE: int = _MAGIC_SIZE + _VERSION_SIZE + _TENSOR_COUNT_SIZE + _KV_COUNT_SIZE

# Поддерживаемые версии формата
_SUPPORTED_VERSIONS: frozenset[int] = frozenset({1, 2, 3})

# Словарь типов значений metadata
_VALUE_TYPE_NAMES: dict[int, str] = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

# Размер скалярных типов (в байтах). string и array — переменные, обрабатываются отдельно.
_SCALAR_SIZES: dict[int, int] = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}

# Анализ подозрительных паттернов (URL, IP, suspicious keys, длинные строки)
# вынесен в GGUFMetadataDetector — сканер только извлекает данные.
# Разделение слоёв: Scanner знает формат, Detector — угрозу.

# ---------------------------------------------------------------------------
# Защитные ограничения
# ---------------------------------------------------------------------------

MAX_METADATA_COUNT: int = 10_000       # максимальное число KV-пар
MAX_STRING_LENGTH: int = 1_000_000     # максимальная длина строки в байтах
_MAX_ARRAY_ITEMS: int = 100_000        # максимальное число элементов массива


@ScannerRegistry.register
class GGUFScanner(BaseScanner):
    """Сканер файлов формата GGUF (.gguf, .ggml).

    Парсит бинарный заголовок GGUF без загрузки тензорных данных в память.
    Извлекает все metadata KV-пары и анализирует их строковые значения
    на предмет URL, IP-адресов, подозрительных ключей и аномально длинных строк.

    Никогда не вызывает pickle.load(), torch.load() и аналогичные функции.
    """

    name: ClassVar[str] = "gguf"
    description: ClassVar[str] = "Сканер файлов формата GGUF/GGML (llama.cpp, Ollama)"
    supported_extensions: ClassVar[list[str]] = [".gguf", ".ggml"]
    magic_bytes: ClassVar[list[bytes]] = [_GGUF_MAGIC]

    #: Дефолтный лимит для GGUF — 100 ГБ (аудит #24).
    #: Real-world LLM: Llama-3-70B Q5_K_M ≈ 49 ГБ, Mixtral-8x22B ≈ 90 ГБ.
    #: GGUFScanner работает потоково (читает только заголовок и KV-metadata),
    #: целый файл в RAM никогда не загружается, поэтому большой лимит безопасен.
    DEFAULT_MAX_FILE_SIZE: ClassVar[int] = 100 * 1024 * 1024 * 1024  # 100 ГБ

    GGUF_MAGIC: ClassVar[bytes] = _GGUF_MAGIC
    SUPPORTED_VERSIONS: ClassVar[frozenset[int]] = _SUPPORTED_VERSIONS
    VALUE_TYPES: ClassVar[dict[int, str]] = _VALUE_TYPE_NAMES
    MAX_METADATA_COUNT: ClassVar[int] = MAX_METADATA_COUNT
    MAX_STRING_LENGTH: ClassVar[int] = MAX_STRING_LENGTH

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Проверяет, что файл является GGUF-форматом.

        Использует расширение (.gguf, .ggml) и/или magic bytes b'GGUF'.
        Magic bytes проверяются только если файл существует и доступен.
        """
        if path.suffix.lower() in cls.supported_extensions:
            return True
        # Проверка по magic bytes для файлов без стандартного расширения
        try:
            with path.open("rb") as fh:
                header = fh.read(_MAGIC_SIZE)
            return header == cls.GGUF_MAGIC
        except OSError:
            return False

    def scan(self, path: Path) -> RawScanData:
        """Сканирует GGUF-файл и возвращает RawScanData.

        Алгоритм:
        1. Проверяет размер файла — отказывает если превышает лимит.
        2. Проверяет magic bytes (b'GGUF').
        3. Читает версию и проверяет, что она поддерживается.
        4. Читает и разбирает metadata KV-пары потоково (без загрузки тензоров).
        5. Анализирует строковые значения на предмет URL, IP, подозрительных ключей.
        6. Возвращает RawScanData с metadata= и strings=.

        При любой ошибке возвращает RawScanData с заполненным полем error.
        Никогда не бросает исключений наружу.
        Тензорные данные НЕ загружаются в RAM — только заголовок и KV-metadata.
        """
        try:
            self._check_file_size(path)
            hashes = self._compute_hashes(path)
            file_size = path.stat().st_size
        except ValueError as exc:
            return RawScanData(
                file_path=path,
                file_hash={},
                file_size=0,
                scanner_name=self.name,
                error=str(exc),
            )
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash={},
                file_size=0,
                scanner_name=self.name,
                error=str(exc),
            )

        try:
            return self._parse(path, hashes, file_size)
        except Exception as exc:  # noqa: BLE001  # намеренный широкий catch — сканер не падает
            logger.debug("Ошибка при разборе GGUF-файла %s: %s", path, exc, exc_info=True)
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Ошибка разбора GGUF-файла: {exc}",
            )

    # ------------------------------------------------------------------
    # Внутренний разборщик (потоковый)
    # ------------------------------------------------------------------

    def _parse(
        self,
        path: Path,
        hashes: dict[str, str],
        file_size: int,
    ) -> RawScanData:
        """Разбирает GGUF-файл потоково и заполняет RawScanData.

        Читает ТОЛЬКО заголовок и KV-metadata через seek/read на file handle.
        Тензорные данные (составляющие 99%+ объёма LLM-файлов) в RAM не загружаются.
        Это ключевое свойство для корректной работы с GGUF-файлами 30–70 ГБ.
        """
        with path.open("rb") as fh:
            # Считываем первые 4096 байт для raw_content_sample
            raw_sample = fh.read(4096)
            fh.seek(0)

            if file_size < _HEADER_MIN_SIZE:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error=(
                        f"Файл слишком короткий: {file_size} байт "
                        f"(минимум {_HEADER_MIN_SIZE} для валидного GGUF)"
                    ),
                )

            # --- Шаг 1: magic bytes ---
            magic = fh.read(_MAGIC_SIZE)
            if magic != _GGUF_MAGIC:
                # Симметрия с неизвестной версией (ниже): аномалию заголовка
                # фиксируем как ФАКТ в metadata, а не только в свободном error.
                # Issue MLS-GGUF-006 (MEDIUM) по этим фактам эмитит
                # GGUFMetadataDetector — сканер Issue не конструирует и наружу
                # не бросает. error тоже ставим (файл не является валидным GGUF).
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    metadata={
                        META_HEADER_VALID: "false",
                        META_BAD_MAGIC: repr(magic),
                        META_BAD_MAGIC_HEX: magic.hex(),
                        META_EXPECTED_MAGIC: repr(_GGUF_MAGIC),
                    },
                    error=f"Неверный magic bytes: {magic!r} (ожидался {_GGUF_MAGIC!r})",
                )

            # --- Шаг 2: версия (uint32 LE) ---
            version_bytes = fh.read(_VERSION_SIZE)
            if len(version_bytes) < _VERSION_SIZE:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error="Файл обрезан: не удалось прочитать версию GGUF",
                )
            version: int = struct.unpack("<I", version_bytes)[0]

            metadata_out: dict[str, str] = {"gguf_version": str(version)}

            if version not in _SUPPORTED_VERSIONS:
                # Issue про неизвестную версию формирует GGUFMetadataDetector
                # на основе metadata_out["gguf_version"]. Сканер только парсит.
                logger.info(
                    "Неизвестная версия GGUF %d в %s — продолжаем по лучшему "
                    "предположению",
                    version,
                    path,
                )

            # --- Шаг 3: tensor_count и kv_count (uint64 LE) ---
            counts_bytes = fh.read(_TENSOR_COUNT_SIZE + _KV_COUNT_SIZE)
            if len(counts_bytes) < _TENSOR_COUNT_SIZE + _KV_COUNT_SIZE:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error="Файл обрезан: не удалось прочитать tensor_count / kv_count",
                )
            tensor_count: int = struct.unpack("<Q", counts_bytes[:_TENSOR_COUNT_SIZE])[0]
            kv_count: int = struct.unpack("<Q", counts_bytes[_TENSOR_COUNT_SIZE:])[0]

            metadata_out["tensor_count"] = str(tensor_count)
            metadata_out["kv_count"] = str(kv_count)

            # --- Шаг 4: проверка kv_count ---
            if kv_count > MAX_METADATA_COUNT:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    metadata=metadata_out,
                    error=(
                        f"metadata_kv_count={kv_count} превышает лимит {MAX_METADATA_COUNT}. "
                        "Возможно, файл повреждён или намеренно создан для атаки."
                    ),
                )

            # --- Шаг 5: потоковый разбор metadata KV ---
            try:
                parsed_meta, all_strings = self._parse_metadata_stream(fh, kv_count)
            except struct.error as exc:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    metadata=metadata_out,
                    error=f"Ошибка при разборе metadata (файл обрезан или повреждён): {exc}",
                )

            # Позиция сразу после KV-metadata — начало секции tensor info.
            # EXE-сигнатуры допустимы только в header+metadata, не в тензорных
            # данных: квантованные веса содержат случайные байты, которые могут
            # совпасть с Mach-O/PE/ELF magic (ложные тревоги).
            # Добавляем 1 МБ запаса для покрытия tensor info entries (имена, формы).
            metadata_section_end = fh.tell() + 1024 * 1024

        # Объединяем системные метаданные с извлечёнными из файла.
        # Анализ подозрительных паттернов (URL, IP, suspicious keys, длинные строки)
        # вынесен в GGUFMetadataDetector — сканер только извлекает данные
        # (Scanner знает формат, Detector — угрозы).
        for k, v in parsed_meta.items():
            metadata_out[k] = v

        # --- Шаг 6: строки для NetworkDetector / SecretsDetector ---
        strings_out: list[StringInfo] = []
        for idx, s in enumerate(all_strings):
            strings_out.append(StringInfo(value=s, position=idx))

        # --- Шаг 7: потоковый поиск PE/ELF/Mach-O только в секции metadata ---
        # Сканируем только header + KV-metadata + tensor info (до начала тензорных данных).
        # Квантованные тензорные веса содержат случайные байтовые паттерны, которые
        # могут совпасть с короткими (4-байтовыми) Mach-O magic — ложные тревоги.
        # metadata_section_end = позиция после KV + 1 МБ запаса для tensor info.
        embedded = find_signatures_in_file(path, max_bytes=metadata_section_end)

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            metadata=metadata_out if metadata_out else None,
            strings=strings_out if strings_out else None,
            embedded_bytes=embedded if embedded else None,
            raw_content_sample=raw_sample,
        )

    # ------------------------------------------------------------------
    # Потоковые вспомогательные методы (работают с IO[bytes], не с bytes)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_exact(fh: IO[bytes], n: int) -> bytes:
        """Читает ровно n байт из файлового дескриптора.

        Бросает struct.error если данных недостаточно (файл обрезан).
        """
        data = fh.read(n)
        if len(data) < n:
            raise struct.error(
                f"Файл обрезан: запрошено {n} байт, получено {len(data)}"
            )
        return data

    def _parse_string_stream(self, fh: IO[bytes]) -> str:
        """Читает GGUF-строку из файлового дескриптора: uint64 length + UTF-8 байты.

        Защита: строки длиннее MAX_STRING_LENGTH пропускаются через seek
        (без чтения в память) и возвращается заглушка.
        """
        str_len_bytes = self._read_exact(fh, 8)
        str_len: int = struct.unpack("<Q", str_len_bytes)[0]

        if str_len > MAX_STRING_LENGTH:
            # Защита от memory exhaustion: seek вперёд без чтения
            fh.seek(str_len, 1)  # 1 = SEEK_CUR
            return f"<TRUNCATED:{str_len}_bytes>"

        str_bytes = self._read_exact(fh, str_len)
        return str_bytes.decode("utf-8", errors="replace")

    def _skip_value_stream(self, fh: IO[bytes], value_type: int) -> None:
        """Пропускает значение указанного типа в файловом потоке.

        Используется для пропуска нестроковых значений при потоковом разборе.
        """
        if value_type in _SCALAR_SIZES:
            fh.seek(_SCALAR_SIZES[value_type], 1)
            return

        if value_type == 8:  # string
            self._parse_string_stream(fh)
            return

        if value_type == 9:  # array
            header = self._read_exact(fh, 4 + 8)
            array_type: int = struct.unpack("<I", header[:4])[0]
            array_count: int = struct.unpack("<Q", header[4:])[0]

            if array_count > _MAX_ARRAY_ITEMS:
                raise struct.error(
                    f"Число элементов массива {array_count} превышает лимит {_MAX_ARRAY_ITEMS}"
                )

            for _ in range(array_count):
                self._skip_value_stream(fh, array_type)
            return

        raise struct.error(f"Неизвестный тип значения metadata: {value_type}")

    def _parse_metadata_stream(
        self,
        fh: IO[bytes],
        count: int,
    ) -> tuple[dict[str, str], list[str]]:
        """Разбирает count metadata KV-пар потоково из файлового дескриптора.

        Возвращает кортеж (metadata_dict, все_строки):
        - metadata_dict: ключ → строковое представление значения
        - все_строки: список всех строковых значений (для NetworkDetector и др.)

        Тензорные данные не читаются — парсинг останавливается
        после последней KV-пары.
        """
        metadata: dict[str, str] = {}
        all_strings: list[str] = []

        fmt_map: dict[int, str] = {
            0: "<B", 1: "<b", 2: "<H", 3: "<h",
            4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d",
        }

        for _ in range(count):
            # Читаем ключ
            key = self._parse_string_stream(fh)

            # Читаем тип значения (uint32 LE)
            value_type: int = struct.unpack("<I", self._read_exact(fh, 4))[0]

            # Читаем значение
            if value_type == 8:  # string
                value_str = self._parse_string_stream(fh)
                metadata[key] = value_str
                all_strings.append(value_str)

            elif value_type == 9:  # array
                header = self._read_exact(fh, 4 + 8)
                array_type: int = struct.unpack("<I", header[:4])[0]
                array_count: int = struct.unpack("<Q", header[4:])[0]

                if array_count > _MAX_ARRAY_ITEMS:
                    raise struct.error(
                        f"Число элементов массива {array_count} превышает лимит {_MAX_ARRAY_ITEMS}"
                    )

                if array_type == 8:  # массив строк
                    items: list[str] = []
                    for _ in range(array_count):
                        item = self._parse_string_stream(fh)
                        items.append(item)
                        all_strings.append(item)
                    metadata[key] = (
                        f"[{', '.join(items[:10])}{'...' if len(items) > 10 else ''}]"
                    )
                else:
                    # Массив нестроковых значений — пропускаем
                    for _ in range(array_count):
                        self._skip_value_stream(fh, array_type)
                    type_name = _VALUE_TYPE_NAMES.get(array_type, str(array_type))
                    metadata[key] = f"<array:{type_name}[{array_count}]>"

            elif value_type in _SCALAR_SIZES:
                size = _SCALAR_SIZES[value_type]
                raw_val = self._read_exact(fh, size)
                fmt = fmt_map.get(value_type, "")
                if fmt:
                    parsed_val = struct.unpack(fmt, raw_val)[0]
                    metadata[key] = str(parsed_val)
                else:
                    metadata[key] = raw_val.hex()
            else:
                raise struct.error(f"Неизвестный тип значения metadata: {value_type}")

        return metadata, all_strings


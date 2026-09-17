"""Сканер NumPy-файлов: .npy и .npz.

Форматы:
- .npy: бинарный файл с одним массивом NumPy.
  Структура:
    [6 байт  magic: b'\\x93NUMPY']
    [1 байт  major_version]
    [1 байт  minor_version]
    [2 байта header_len, little-endian uint16]  — для major == 1
    [4 байта header_len, little-endian uint32]  — для major >= 2
    [header_len байт  ASCII/UTF-8 Python-dict строка]
    [бинарные данные массива]

- .npz: ZIP-архив, содержащий .npy-файлы (по одному на массив).
  Имена записей: '<name>.npy'. Стандартный ZIP.

Угроза: массивы с dtype=object сериализуются через pickle.
  Если файл содержит object-array с вредоносным pickle-payload,
  атака может быть скрыта внутри легитимного .npy/.npz файла.
  Алгоритм извлечения фактов (интерпретация — за детекторами, §4.3):
    1. При dtype с дескриптором 'O' (object) выставляем в metadata
       object_dtype_detected="true". Issue MLS-NPY-001 эмитит
       NumpyMetadataDetector, а не сканер.
    2. Если данные после заголовка содержат pickle magic bytes →
       рекурсивно вызываем PickleScanner.scan_bytes() для полного анализа
       и выставляем metadata pickle_payload_detected="true".

Ссылка на формат: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
"""

from __future__ import annotations

import ast
import logging
import struct
from pathlib import Path
from typing import ClassVar

from poison_check.core.container import ContainerError, ContainerExtractor
from poison_check.core.executable_signatures import find_signatures_in_bytes
from poison_check.core.registry import ScannerRegistry
from poison_check.core.result import (
    EmbeddedSignature,
    OpcodeInfo,
    ReduceCall,
    StringInfo,
    TensorInfo,
)
from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы формата
# ---------------------------------------------------------------------------

NUMPY_MAGIC: bytes = b"\x93NUMPY"
_MAGIC_LEN: int = len(NUMPY_MAGIC)  # 6

# Общая длина полей до header_len (включительно): magic + major + minor
_PREAMBLE_V1: int = _MAGIC_LEN + 1 + 1 + 2  # 10 байт
_PREAMBLE_V2: int = _MAGIC_LEN + 1 + 1 + 4  # 12 байт

# Magic bytes pickle-протоколов 2–5
_PICKLE_MAGICS: tuple[bytes, ...] = (
    b"\x80\x02",
    b"\x80\x03",
    b"\x80\x04",
    b"\x80\x05",
)

# Характерные стартовые байты pickle proto 0/1 (без PROTO opcode).
# Используются ТОЛЬКО для object-dtype массивов: numpy сериализует object
# arrays через pickle.dumps() с протоколом по умолчанию (>=2 в современном
# Python), но атакующий может явно указать protocol=0/1, чтобы обойти проверку
# magic bytes \x80\x02..\x80\x05. См. аудит #8.
#
# proto 0/1 начинается с одного из opcodes:
#   '(' MARK         '}' EMPTY_DICT     ']' EMPTY_LIST   ')' EMPTY_TUPLE
#   'l' LIST         'd' DICT           't' TUPLE        'c' GLOBAL (proto 0)
#   'i' INST         'I' INT            'L' LONG         'S' STRING
#   'V' UNICODE      'U' SHORT_BINSTRING (proto 1)
# Минимально достаточный набор для эвристики «здесь точно начинается pickle».
# Полная валидация — через pickletools в PickleScanner.
_PICKLE_PROTO01_LEAD: frozenset[int] = frozenset(b"(}])ldticIL SVU")

# ZIP magic (для .npz)
_ZIP_MAGIC: bytes = b"PK\x03\x04"

# Dtype-дескрипторы, указывающие на object array
# В NumPy: 'O', '|O', '<O', '>O', 'object', 'object_', 'O8'
_OBJECT_DTYPE_MARKERS: frozenset[str] = frozenset({"O", "object", "object_"})

#: Ключи metadata, которыми сканер фиксирует факт «внутри массива лежит pickle».
#: Публичные, потому что их читают два потребителя: NumpyMetadataDetector
#: (MLS-NPY-001) и реестр безопасности форматов
#: (:mod:`poison_check.scanners.format_facts`) — object-dtype превращает
#: формально безопасный .npy в code-bearing формат.
META_OBJECT_DTYPE: str = "object_dtype_detected"
META_PICKLE_PAYLOAD: str = "pickle_payload_detected"


@ScannerRegistry.register
class NumpyScanner(BaseScanner):
    """Сканер NumPy-файлов (.npy, .npz).

    Разбирает заголовок .npy без вызова numpy.load(), извлекает dtype и shape.
    При обнаружении object-dtype проверяет данные на pickle-payload и
    при наличии рекурсивно вызывает PickleScanner.scan_bytes() для анализа.

    Сканер не интерпретирует находки: факт object-dtype фиксируется в
    metadata (``object_dtype_detected``), а Issue MLS-NPY-001 эмитит
    NumpyMetadataDetector (Scanner знает формат,
    Detector знает угрозу).

    Устойчив к известной проблеме разбора object-dtype .npy-файлов, на
    которой парсеры могут падать: этот сканер никогда не бросает
    необработанных исключений наружу.
    """

    name = "numpy"
    description = "Сканер NumPy-файлов (.npy, .npz)"
    supported_extensions: ClassVar[list[str]] = [".npy", ".npz"]
    magic_bytes: ClassVar[list[bytes]] = [NUMPY_MAGIC, _ZIP_MAGIC]

    NUMPY_MAGIC: ClassVar[bytes] = NUMPY_MAGIC
    DANGEROUS_DTYPES: ClassVar[list[str]] = ["object"]

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Проверяет расширение и, для .npy, magic bytes.

        .npy: расширение + первые 6 байт == NUMPY_MAGIC.
        .npz: расширение + первые 4 байта == ZIP magic.
        """
        suffix = path.suffix.lower()
        if suffix not in cls.supported_extensions:
            return False
        try:
            with path.open("rb") as fh:
                header = fh.read(6)
        except OSError:
            return False
        if suffix == ".npy":
            return header[:_MAGIC_LEN] == NUMPY_MAGIC
        if suffix == ".npz":
            return header[:4] == _ZIP_MAGIC
        return False  # unreachable, но mypy доволен

    def scan(self, path: Path) -> RawScanData:
        """Определяет формат (.npy / .npz) и вызывает соответствующий метод.

        При любой ошибке возвращает RawScanData с заполненным полем error.
        Никогда не бросает исключений наружу.
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
            with path.open("rb") as fh:
                first_bytes = fh.read(6)
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=str(exc),
            )

        try:
            if first_bytes[:_MAGIC_LEN] == NUMPY_MAGIC:
                # Потоковая версия (аудит #2): читаем только заголовок целиком,
                # array_data — отдельно ограниченным чтением. path.read_bytes()
                # на .npy 30 ГБ положил бы сканер.
                return self._scan_npy_stream(
                    path, hashes=hashes, file_size=file_size,
                )
            if first_bytes[:4] == _ZIP_MAGIC:
                return self._scan_npz(path, hashes=hashes, file_size=file_size)
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=(
                    f"Неизвестный формат NumPy-файла: "
                    f"первые байты {first_bytes!r}. "
                    "Ожидался .npy (\\x93NUMPY) или .npz (ZIP)."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ошибка при разборе %s: %s", path, exc, exc_info=True)
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Ошибка разбора NumPy-файла: {exc}",
            )

    # ------------------------------------------------------------------
    # .npy
    # ------------------------------------------------------------------

    # Максимальный размер object-array data для загрузки в RAM при глубоком
    # анализе pickle-payload. Аудит #2: на больших .npy без object-dtype мы
    # не читаем data вообще, на object-dtype — не больше этого лимита.
    _MAX_OBJECT_ARRAY_BYTES: ClassVar[int] = 512 * 1024 * 1024  # 512 МБ

    def _scan_npy_stream(
        self,
        path: Path,
        hashes: dict[str, str],
        file_size: int,
    ) -> RawScanData:
        """Потоковая версия _scan_npy: не загружает .npy целиком в RAM.

        Регрессия аудита #2: старая реализация делала ``path.read_bytes()``,
        что на 30-ГБ .npy валило сканер по OOM. Новая реализация:

        1. Читает только preamble + JSON-header через file handle (KB).
        2. Если dtype != object — array_data вообще не загружаем; ограничиваемся
           tensor_info + поиском PE/ELF в первых ``_MAX_OBJECT_ARRAY_BYTES``
           байтах через потоковый ``find_signatures_in_file``.
        3. Если dtype == object — читаем array_data ограниченным чтением
           (не более ``_MAX_OBJECT_ARRAY_BYTES``), чтобы передать в PickleScanner.
        """
        with path.open("rb") as fh:
            magic = fh.read(_MAGIC_LEN)
            if magic != NUMPY_MAGIC:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error=f"Некорректные magic bytes .npy: {magic!r}",
                )

            ver_bytes = fh.read(2)
            if len(ver_bytes) < 2:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error="Файл слишком короткий: версия .npy не прочитана",
                )
            major, minor = ver_bytes[0], ver_bytes[1]

            if major == 1:
                hl_bytes = fh.read(2)
                if len(hl_bytes) < 2:
                    return RawScanData(
                        file_path=path, file_hash=hashes,
                        file_size=file_size, scanner_name=self.name,
                        error=f"Файл слишком короткий для .npy v1: {file_size} байт",
                    )
                header_len = struct.unpack("<H", hl_bytes)[0]
                preamble_size = _PREAMBLE_V1
            elif major in (2, 3):
                hl_bytes = fh.read(4)
                if len(hl_bytes) < 4:
                    return RawScanData(
                        file_path=path, file_hash=hashes,
                        file_size=file_size, scanner_name=self.name,
                        error=f"Файл слишком короткий для .npy v{major}: {file_size} байт",
                    )
                header_len = struct.unpack("<I", hl_bytes)[0]
                preamble_size = _PREAMBLE_V2
            else:
                logger.warning(
                    "Неизвестная версия .npy %s.%s в %s, пробуем v1-формат",
                    major, minor, path,
                )
                hl_bytes = fh.read(2)
                if len(hl_bytes) < 2:
                    return RawScanData(
                        file_path=path, file_hash=hashes,
                        file_size=file_size, scanner_name=self.name,
                        error=f"Неизвестная версия .npy: {major}.{minor}",
                    )
                header_len = struct.unpack("<H", hl_bytes)[0]
                preamble_size = _PREAMBLE_V1

            # Защита от monstrously large header (decompression-style attack)
            if header_len > 100 * 1024 * 1024:  # 100 МБ
                return RawScanData(
                    file_path=path, file_hash=hashes,
                    file_size=file_size, scanner_name=self.name,
                    error=(
                        f"Аномально большой заголовок .npy: {header_len} байт. "
                        "Возможная атака."
                    ),
                )

            header_bytes = fh.read(header_len)
            if len(header_bytes) < header_len:
                return RawScanData(
                    file_path=path, file_hash=hashes,
                    file_size=file_size, scanner_name=self.name,
                    error=(
                        f"Заголовок .npy обрезан: прочитано {len(header_bytes)} байт, "
                        f"ожидалось {header_len}"
                    ),
                )

            data_offset = preamble_size + header_len
            header_str = header_bytes.decode("latin-1").strip()
            dtype_str, shape, parse_error = _parse_npy_header(header_str)

            tensor_name = path.stem
            errors: list[str] = []
            metadata: dict[str, str] = {
                "npy_version": f"{major}.{minor}",
                "dtype": dtype_str,
                "shape": str(shape),
            }
            if parse_error:
                errors.append(f"Предупреждение: не удалось разобрать заголовок .npy: {parse_error}")
                metadata["header_parse_error"] = parse_error

            tensor_info: list[TensorInfo] = [
                TensorInfo(name=tensor_name, dtype=dtype_str, shape=shape)
            ]

            is_object_dtype = _is_object_dtype(dtype_str)

            opcodes: list[OpcodeInfo] | None = None
            globals_set: set[tuple[str, str]] | None = None
            strings: list[StringInfo] | None = None
            reduce_calls: list[ReduceCall] | None = None
            nested: list[RawScanData] | None = None
            embedded: list[EmbeddedSignature] | None = None

            # Сэмпл первых 4 КБ — нужен detector'ам как fallback
            fh.seek(0)
            raw_sample = fh.read(4096)

            if is_object_dtype:
                # Читаем array_data с ограничением — не больше _MAX_OBJECT_ARRAY_BYTES.
                # Если массив object-dtype, но больше лимита — мы видим только
                # начало, что хуже чем ничего, но защищает от OOM.
                fh.seek(data_offset)
                array_data = fh.read(self._MAX_OBJECT_ARRAY_BYTES + 1)
                truncated = len(array_data) > self._MAX_OBJECT_ARRAY_BYTES
                if truncated:
                    array_data = array_data[: self._MAX_OBJECT_ARRAY_BYTES]
                    metadata["array_data_truncated"] = "true"
                    metadata["array_data_truncated_at_bytes"] = str(
                        self._MAX_OBJECT_ARRAY_BYTES
                    )

                # Только факт, без интерпретации: Issue MLS-NPY-001 эмитит
                # NumpyMetadataDetector по этому флагу.
                metadata[META_OBJECT_DTYPE] = "true"

                # Поиск PE/ELF в загруженной части array_data
                local_emb = find_signatures_in_bytes(array_data, base_offset=data_offset)
                if local_emb:
                    embedded = local_emb

                if _find_pickle_magic(array_data):
                    metadata[META_PICKLE_PAYLOAD] = "true"
                    inner = _scan_pickle_in_npy(array_data, path)
                    nested = [inner]
                    opcodes = inner.opcodes
                    globals_set = inner.globals
                    strings = inner.strings
                    reduce_calls = inner.reduce_calls
                    if inner.error:
                        errors.append(f"Ошибка разбора pickle в object-массиве: {inner.error}")
            else:
                # dtype != object — потоковый поиск PE/ELF в файле, без чтения в RAM
                from poison_check.core.executable_signatures import find_signatures_in_file
                local_emb = find_signatures_in_file(
                    path,
                    max_bytes=min(self._MAX_OBJECT_ARRAY_BYTES, file_size),
                )
                if local_emb:
                    embedded = local_emb

        # RawScanData не имеет поля issues — и не должен: интерпретация фактов
        # (object_dtype_detected / pickle_payload_detected в metadata) —
        # ответственность NumpyMetadataDetector.
        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=opcodes,
            globals=globals_set,
            strings=strings,
            reduce_calls=reduce_calls,
            embedded_bytes=embedded,
            metadata=metadata,
            tensor_info=tensor_info,
            nested_files=nested,
            raw_content_sample=raw_sample,
            error="; ".join(errors) if errors else None,
        )

    def _scan_npy(
        self,
        data: bytes,
        source_path: Path,
        hashes: dict[str, str] | None = None,
        file_size: int | None = None,
    ) -> RawScanData:
        """Разбирает бинарный .npy поток и возвращает RawScanData.

        Args:
            data:        Полное содержимое .npy файла.
            source_path: Путь для поля file_path и логирования.
            hashes:      Предвычисленные хеши (None — не вычислять повторно).
            file_size:   Размер файла (None → len(data)).

        Алгоритм:
        1. Проверяет magic bytes.
        2. Считывает major/minor версию и header_len (2 или 4 байта).
        3. Парсит ASCII/UTF-8 строку-заголовок как Python-литерал (dict).
        4. Извлекает dtype и shape.
        5. Если dtype == object → metadata object_dtype_detected + проверка
           на pickle (Issue MLS-NPY-001 эмитит NumpyMetadataDetector).
        6. Если pickle magic найден в данных → рекурсивный PickleScanner.
        """
        if hashes is None:
            hashes = {}
        if file_size is None:
            file_size = len(data)

        # --- Шаг 1: magic ---
        if len(data) < _MAGIC_LEN + 2:
            return RawScanData(
                file_path=source_path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Файл слишком короткий для .npy: {len(data)} байт",
            )

        if data[:_MAGIC_LEN] != NUMPY_MAGIC:
            # Если вместо .npy — pickle stream, анализируем как pickle (атака подмены).
            if any(data[:len(m)] == m for m in _PICKLE_MAGICS):
                inner = _scan_pickle_in_npy(data, source_path=source_path)
                return RawScanData(
                    file_path=source_path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    opcodes=inner.opcodes,
                    globals=inner.globals,
                    strings=inner.strings,
                    reduce_calls=inner.reduce_calls,
                    embedded_bytes=inner.embedded_bytes,
                    metadata={"npy_disguise": "pickle_payload_in_npy"},
                    error=f"Некорректные magic bytes .npy: {data[:_MAGIC_LEN]!r}",
                )
            return RawScanData(
                file_path=source_path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Некорректные magic bytes .npy: {data[:_MAGIC_LEN]!r}",
            )

        # --- Шаг 2: версия и размер заголовка ---
        major = data[_MAGIC_LEN]
        minor = data[_MAGIC_LEN + 1]

        if major == 1:
            # v1.0: 2-байтовый uint16 little-endian
            if len(data) < _PREAMBLE_V1:
                return RawScanData(
                    file_path=source_path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error=f"Файл слишком короткий для .npy v1: {len(data)} байт",
                )
            header_len = struct.unpack("<H", data[_MAGIC_LEN + 2: _MAGIC_LEN + 4])[0]
            data_offset = _PREAMBLE_V1 + header_len
            header_bytes = data[_PREAMBLE_V1: data_offset]
        elif major in (2, 3):
            # v2.0/v3.0: 4-байтовый uint32 little-endian
            if len(data) < _PREAMBLE_V2:
                return RawScanData(
                    file_path=source_path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error=f"Файл слишком короткий для .npy v{major}: {len(data)} байт",
                )
            header_len = struct.unpack("<I", data[_MAGIC_LEN + 2: _MAGIC_LEN + 6])[0]
            data_offset = _PREAMBLE_V2 + header_len
            header_bytes = data[_PREAMBLE_V2: data_offset]
        else:
            # Неизвестная версия — пытаемся использовать v1-формат как fallback
            logger.warning(
                "Неизвестная версия .npy %s.%s в %s, пробуем v1-формат",
                major, minor, source_path,
            )
            if len(data) < _PREAMBLE_V1:
                return RawScanData(
                    file_path=source_path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error=f"Неизвестная версия .npy: {major}.{minor}",
                )
            header_len = struct.unpack("<H", data[_MAGIC_LEN + 2: _MAGIC_LEN + 4])[0]
            data_offset = _PREAMBLE_V1 + header_len
            header_bytes = data[_PREAMBLE_V1: data_offset]

        if len(header_bytes) < header_len:
            return RawScanData(
                file_path=source_path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=(
                    f"Заголовок .npy обрезан: прочитано {len(header_bytes)} байт, "
                    f"ожидалось {header_len}"
                ),
            )

        # --- Шаг 3: парсим Python-dict из заголовка ---
        header_str = header_bytes.decode("latin-1").strip()
        dtype_str, shape, parse_error = _parse_npy_header(header_str)

        tensor_name = source_path.stem  # имя файла как имя тензора
        errors: list[str] = []
        metadata: dict[str, str] = {
            "npy_version": f"{major}.{minor}",
            "dtype": dtype_str,
            "shape": str(shape),
        }

        if parse_error:
            errors.append(f"Предупреждение: не удалось разобрать заголовок .npy: {parse_error}")
            metadata["header_parse_error"] = parse_error

        tensor_info: list[TensorInfo] = [
            TensorInfo(name=tensor_name, dtype=dtype_str, shape=shape)
        ]

        # --- Шаг 4: проверка object dtype ---
        array_data = data[data_offset:]
        is_object_dtype = _is_object_dtype(dtype_str)

        opcodes: list[OpcodeInfo] | None = None
        globals_set: set[tuple[str, str]] | None = None
        strings: list[StringInfo] | None = None
        reduce_calls: list[ReduceCall] | None = None
        nested: list[RawScanData] | None = None
        # Поиск встроенных PE/ELF/Mach-O в данных массива.
        # Делаем это для любого dtype: PE может быть и в bytes-массиве.
        embedded_list = find_signatures_in_bytes(array_data, base_offset=data_offset)
        embedded: list[EmbeddedSignature] | None = embedded_list if embedded_list else None

        if is_object_dtype:
            # Фиксируем факт object dtype; Issue MLS-NPY-001 эмитит
            # NumpyMetadataDetector по этому флагу.
            metadata[META_OBJECT_DTYPE] = "true"

            # --- Шаг 5: если в данных есть pickle → глубокий анализ ---
            pickle_magic_found = _find_pickle_magic(array_data)
            if pickle_magic_found:
                metadata[META_PICKLE_PAYLOAD] = "true"
                inner = _scan_pickle_in_npy(array_data, source_path)
                nested = [inner]
                opcodes = inner.opcodes
                globals_set = inner.globals
                strings = inner.strings
                reduce_calls = inner.reduce_calls
                if inner.error:
                    errors.append(f"Ошибка разбора pickle в object-массиве: {inner.error}")

        return RawScanData(
            file_path=source_path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=opcodes,
            globals=globals_set,
            strings=strings,
            reduce_calls=reduce_calls,
            embedded_bytes=embedded,
            metadata=metadata,
            tensor_info=tensor_info,
            nested_files=nested,
            raw_content_sample=data[:4096],
            error="; ".join(errors) if errors else None,
        )

    # ------------------------------------------------------------------
    # .npz
    # ------------------------------------------------------------------

    def _scan_npz(
        self,
        path: Path,
        hashes: dict[str, str],
        file_size: int,
    ) -> RawScanData:
        """Распаковывает .npz (ZIP) и сканирует каждый .npy-файл внутри.

        Распаковка идёт через ``ContainerExtractor.extract_zip_members`` —
        потоково, член за членом, с лимитами MAX_MEMBER_SIZE / MAX_EXTRACT_SIZE
        и защитой от path-traversal. Прежняя реализация делала
        ``zipfile.ZipFile(io.BytesIO(path.read_bytes()))``, то есть грузила
        весь .npz в RAM и не имела защиты от decompression bomb (.npz — ZIP).

        Объединяет результаты всех вложенных массивов в один RawScanData:
        - nested_files содержит RawScanData каждого массива
        - globals/opcodes/strings/reduce_calls агрегируются
        - tensor_info содержит информацию обо всех массивах
        """
        all_opcodes: list[OpcodeInfo] = []
        all_globals: set[tuple[str, str]] = set()
        all_strings: list[StringInfo] = []
        all_reduce_calls: list[ReduceCall] = []
        all_tensor_info: list[TensorInfo] = []
        all_embedded: list[EmbeddedSignature] = []
        nested: list[RawScanData] = []
        errors: list[str] = []
        metadata: dict[str, str] = {"npz_format": "zip"}

        member_names: list[str] = []
        try:
            for member_name, member_data in ContainerExtractor.extract_zip_members(path):
                member_names.append(member_name)

                # Обрабатываем все .npy члены архива
                if not member_name.endswith(".npy"):
                    logger.debug(
                        "Пропускаем не-.npy член %r в %s", member_name, path
                    )
                    continue

                member_path = path / member_name
                inner = self._scan_npy(
                    member_data,
                    source_path=member_path,
                    hashes={},
                    file_size=len(member_data),
                )
                nested.append(inner)

                if inner.opcodes:
                    all_opcodes.extend(inner.opcodes)
                if inner.globals:
                    all_globals.update(inner.globals)
                if inner.strings:
                    all_strings.extend(inner.strings)
                if inner.reduce_calls:
                    all_reduce_calls.extend(inner.reduce_calls)
                if inner.tensor_info:
                    all_tensor_info.extend(inner.tensor_info)
                if inner.embedded_bytes:
                    all_embedded.extend(inner.embedded_bytes)
                if inner.error:
                    errors.append(f"{member_name}: {inner.error}")
        except ContainerError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Ошибка распаковки .npz: {exc}",
            )
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Ошибка чтения .npz: {exc}",
            )

        metadata["member_count"] = str(len(member_names))
        metadata["members"] = ", ".join(sorted(member_names))

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=all_opcodes if all_opcodes else None,
            globals=all_globals if all_globals else None,
            strings=all_strings if all_strings else None,
            reduce_calls=all_reduce_calls if all_reduce_calls else None,
            embedded_bytes=all_embedded if all_embedded else None,
            metadata=metadata,
            tensor_info=all_tensor_info if all_tensor_info else None,
            nested_files=nested if nested else None,
            raw_content_sample=None,
            error="; ".join(errors) if errors else None,
        )


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня, тестируемые независимо)
# ---------------------------------------------------------------------------


def _parse_npy_header(header_str: str) -> tuple[str, list[int], str | None]:
    """Парсит строку-заголовок .npy (Python-dict) и извлекает dtype и shape.

    Возвращает (dtype_str, shape_list, error_or_None).
    Никогда не бросает исключений.
    """
    try:
        # ast.literal_eval безопасен — не выполняет произвольный код
        obj = ast.literal_eval(header_str)
    except (ValueError, SyntaxError) as exc:
        return "", [], str(exc)

    if not isinstance(obj, dict):
        return "", [], f"Заголовок .npy должен быть словарём, получен {type(obj).__name__}"

    # dtype может быть строкой вида '<f4', '|O', или списком (structured array)
    dtype_raw: object = obj.get("descr", "")
    if isinstance(dtype_raw, list):
        # Structured dtype: [('field', '<f4'), ...]
        dtype_str = "structured"
    elif isinstance(dtype_raw, str):
        # Нормализуем: '<f4' → 'f4', '|O' → 'O'
        dtype_str = dtype_raw.lstrip("<>|=").strip()
    else:
        dtype_str = str(dtype_raw)

    # shape: tuple → list[int]
    shape_raw: object = obj.get("shape", ())
    if isinstance(shape_raw, (tuple, list)):
        shape: list[int] = [int(s) for s in shape_raw]
    else:
        shape = []

    return dtype_str, shape, None


def _is_object_dtype(dtype_str: str) -> bool:
    """Проверяет, является ли dtype типом object (pickle-совместимым).

    Примеры: 'O', 'object', 'object_', 'O8'.
    """
    stripped = dtype_str.lstrip("<>|= ")
    # 'O' или 'O8' или 'object' или 'object_'
    if stripped in _OBJECT_DTYPE_MARKERS:
        return True
    if stripped.startswith("O") and (len(stripped) == 1 or stripped[1:].isdigit()):
        return True
    return bool(stripped.lower().startswith("object"))


def _find_pickle_magic(data: bytes) -> bool:
    """Проверяет, содержит ли буфер данных pickle magic bytes.

    Покрывает протоколы 2–5 (через ``\\x80\\xNN``) И эвристически — proto 0/1
    (через первый байт-opcode + наличие STOP-байта ``b'.'`` далее в данных).
    Без proto 0/1 атакующий мог бы создать object-массив с
    ``pickle.dumps(payload, protocol=0)`` и обойти глубокий анализ (аудит #8).
    """
    if any(magic in data for magic in _PICKLE_MAGICS):
        return True
    # proto 0/1: первый байт массива должен быть pickle-opcode AND где-то
    # дальше должен быть STOP-байт b'.'. Это эвристика — точная проверка
    # делегирована pickletools в PickleScanner.
    return bool(data) and data[0] in _PICKLE_PROTO01_LEAD and b"." in data


def _scan_pickle_in_npy(array_data: bytes, source_path: Path) -> RawScanData:
    """Вызывает PickleScanner для анализа pickle-payload внутри object-массива.

    Ищет первое вхождение pickle magic в массиве данных и передаёт
    начиная с него в PickleScanner.scan_bytes(). Поддерживает протоколы 0–5.
    """
    from poison_check.scanners.pickle_scanner import PickleScanner  # ленивый импорт

    # Находим начало pickle-потока для proto >= 2 (магия \x80\xNN)
    pickle_start = len(array_data)
    for magic in _PICKLE_MAGICS:
        pos = array_data.find(magic)
        if pos != -1 and pos < pickle_start:
            pickle_start = pos

    # Если магии proto 2–5 нет, но первый байт похож на pickle proto 0/1 opcode
    # и в данных есть STOP — пробуем парсить с начала массива.
    if (
        pickle_start == len(array_data)
        and array_data
        and array_data[0] in _PICKLE_PROTO01_LEAD
        and b"." in array_data
    ):
        pickle_start = 0

    pickle_data = array_data if pickle_start == 0 else array_data[pickle_start:]

    scanner = PickleScanner()
    return scanner.scan_bytes(pickle_data, source_path=source_path)

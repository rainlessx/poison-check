"""Потоковый поиск сигнатур исполняемых файлов (PE/ELF/Mach-O).

Используется сканерами форматов для заполнения RawScanData.embedded_bytes
до того, как ExecutableDetector соберёт по ним Issues.

Ключевая разница с детектором: эта утилита работает на сыром файле или
буфере без интерпретации формата — каждый сканер сам решает, какие именно
регионы файла стоит сканировать (например, PickleScanner — весь pickle-поток,
PyTorchScanner — каждый член ZIP-архива, NumpyScanner — массив object-данных).

Раньше детектор ограничивался первыми 4 КБ raw_content_sample, что делало
его декоративным: PE/ELF, встроенный после первых 4 КБ, не находился.

Валидация структуры (fix FP на тензорных данных):
  PE  — после MZ проверяет e_lfanew → PE\\x00\\x00. Исключает случайные
        совпадения 2-байтовой последовательности в float32-весах модели.
  ELF — после \\x7fELF проверяет ei_class, ei_data, ei_version.
  Mach-O fat (\\xcafe\\xbabe) — проверяет nfat_arch (1..16) во избежание FP
  на BF16-тензорных данных; одиночные Mach-O (feeddace/cefaedfe/…) —
  проверяет cputype и ncmds.
"""

from __future__ import annotations

import logging
import re
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Final

from poison_check.core.result import EmbeddedSignature

logger = logging.getLogger(__name__)


# Сигнатуры исполняемых форматов: magic bytes → (тип, описание)
# Должно быть в синхронизации с poison_check/detectors/executable_detector.py
_SIGNATURES: Final[dict[bytes, tuple[str, str]]] = {
    b"MZ": ("PE", "Windows PE-исполняемый файл"),
    b"\x7fELF": ("ELF", "Linux ELF-исполняемый файл"),
    b"\xfe\xed\xfa\xce": ("Mach-O", "macOS Mach-O 32-bit"),
    b"\xfe\xed\xfa\xcf": ("Mach-O", "macOS Mach-O 64-bit"),
    b"\xca\xfe\xba\xbe": ("Mach-O", "macOS Mach-O Universal Binary"),
    b"\xce\xfa\xed\xfe": ("Mach-O", "macOS Mach-O 32-bit (little-endian)"),
    b"\xcf\xfa\xed\xfe": ("Mach-O", "macOS Mach-O 64-bit (little-endian)"),
}

# Скомпилированный regex с alternation всех сигнатур.
# Один проход по буферу вместо N (по числу сигнатур) — оптимизация аудита #7.
# Сигнатуры в alternation сортируются по убыванию длины: re использует
# leftmost-match, более длинные паттерны должны идти первыми, чтобы избежать
# проглатывания префиксов короткими паттернами на одной позиции.
_SIGNATURES_BY_LEN: Final[list[bytes]] = sorted(
    _SIGNATURES.keys(), key=len, reverse=True
)
_SIGNATURE_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    b"|".join(re.escape(sig) for sig in _SIGNATURES_BY_LEN)
)

#: Размер чанка при потоковом сканировании файла (1 МБ).
#: Меньше — больше системных вызовов, больше — больше пиковой памяти на оверлап.
_CHUNK_SIZE: Final[int] = 1024 * 1024  # 1 MB

#: Длина оверлапа между чанками. Должна быть >= max(len(sig)) — иначе
#: сигнатура, лежащая на стыке двух чанков, не будет найдена.
#:
#: Захардкожен 16 (а не вычисляется как max), чтобы добавление новой более
#: длинной сигнатуры (например, JAR ``PK\x03\x04..``, Mach-O fat 8-байтовый)
#: не порождало тихую регрессию на стыках чанков. Проверка инварианта —
#: assert ниже.
_OVERLAP: Final[int] = 16
assert max(len(sig) for sig in _SIGNATURES) <= _OVERLAP, (
    f"_OVERLAP ({_OVERLAP}) меньше длины самой длинной сигнатуры "
    f"({max(len(sig) for sig in _SIGNATURES)}). При добавлении новой сигнатуры "
    "увеличьте _OVERLAP."
)

#: Максимальное число найденных сигнатур, которое возвращается одной операцией.
#: Защита от файлов, у которых байт MZ встречается тысячи раз случайно — мы
#: возвращаем достаточно для детектора, но не раздуваем RawScanData.
_MAX_FINDINGS: Final[int] = 256

# Граница допустимых значений e_lfanew в DOS-заголовке PE.
# Реальные PE-файлы: от 0x40 (минимум после DOS header) до 0x400 (с богатым
# DOS stub). Верхняя граница 0x1000 даёт запас для нестандартных компиляторов.
_PE_E_LFANEW_MIN: Final[int] = 0x40
_PE_E_LFANEW_MAX: Final[int] = 0x1000

# Допустимые значения cputype в заголовке Mach-O (одиночный бинарный).
# Источник: <mach/machine.h>.  Достаточно минимального набора для фильтрации
# случайных совпадений в float32/BF16-весах.
_MACHO_KNOWN_CPUTYPES: Final[frozenset[int]] = frozenset(
    # VAX, MC68k, x86, MIPS, MC98000, ARM, SPARC, i860, ALPHA, PPC, ANY
    {1, 6, 7, 10, 11, 12, 14, 15, 16, 18, 255}
)
# При добавлении 0x01000000 (ABI64-флаг) к базовому типу получаем 64-битный вариант
_MACHO_ABI64_FLAG: Final[int] = 0x01000000

# Максимально разумное число секций load commands в реальном Mach-O
_MACHO_NCMDS_MAX: Final[int] = 256

# Максимальное число архитектур в fat-бинарике
_MACHO_FAT_NARCH_MAX: Final[int] = 16


def _validate_pe(data: bytes, pos: int) -> bool:
    """Проверяет структуру PE-файла по MZ-сигнатуре на позиции pos.

    Алгоритм:
    1. Читает e_lfanew (uint32 LE) из DOS-заголовка по смещению pos+0x3C.
    2. Проверяет, что e_lfanew в разумном диапазоне [0x40, 0x1000].
    3. Читает 4 байта по смещению pos+e_lfanew и сравнивает с b'PE\\x00\\x00'.

    На случайных данных вероятность ложного прохождения ≈ 2.3×10⁻¹⁰ —
    исключает FP из float32-тензоров, где 2-байтовый MZ встречается ~6700
    раз в модели 440 МБ.
    """
    # Нужно минимум pos+0x3C+4 байт для чтения e_lfanew
    if pos + 0x40 > len(data):
        return False
    (e_lfanew,) = struct.unpack_from("<I", data, pos + 0x3C)
    if not (_PE_E_LFANEW_MIN <= e_lfanew <= _PE_E_LFANEW_MAX):
        return False
    pe_sig_start = pos + e_lfanew
    if pe_sig_start + 4 > len(data):
        return False
    return data[pe_sig_start : pe_sig_start + 4] == b"PE\x00\x00"


def _validate_elf(data: bytes, pos: int) -> bool:
    """Проверяет структуру ELF-файла по \\x7fELF-сигнатуре на позиции pos.

    Проверяет три поля ELF-ident (e_ident[4..6]):
    - ei_class:   1 (32-bit) или 2 (64-bit)
    - ei_data:    1 (little-endian) или 2 (big-endian)
    - ei_version: должен быть 1 (текущая версия ELF)
    """
    if pos + 7 > len(data):
        return False
    ei_class = data[pos + 4]
    ei_data = data[pos + 5]
    ei_version = data[pos + 6]
    return ei_class in (1, 2) and ei_data in (1, 2) and ei_version == 1


def _validate_macho_fat(data: bytes, pos: int) -> bool:
    """Проверяет структуру Mach-O fat-бинарика (\\xcafe\\xbabe) на позиции pos.

    Читает nfat_arch (uint32 big-endian, байты pos+4..+7) и проверяет, что
    значение в диапазоне [1, 16]. В BF16/float32 тензорных данных случайное
    попадание обоих условий (cafebabe + nfat_arch ∈ [1..16]) крайне маловероятно.
    """
    if pos + 8 > len(data):
        return False
    nfat_arch: int = struct.unpack_from(">I", data, pos + 4)[0]
    return 1 <= nfat_arch <= _MACHO_FAT_NARCH_MAX


def _validate_macho_single(data: bytes, pos: int, little_endian: bool) -> bool:
    """Проверяет структуру одиночного Mach-O бинарика (feeddace/cefaedfe/…).

    Поля mach_header после magic (4 байта):
      +4: cputype  (int32) — должен быть известным CPU-типом
      +8: cpusubtype (int32) — не проверяем
      +12: filetype (uint32) — не проверяем строго
      +16: ncmds (uint32)  — число load commands, должно быть > 0 и < 256

    Сигнатура 4-байтовая, но при big-endian шанс случайного совпадения в
    тензорных данных не пренебрежимо мал — поэтому дополнительная проверка.
    """
    if pos + 20 > len(data):
        return False
    endian = "<" if little_endian else ">"
    (cputype,) = struct.unpack_from(f"{endian}i", data, pos + 4)
    (ncmds,) = struct.unpack_from(f"{endian}I", data, pos + 16)
    # Убираем флаг ABI64 для проверки базового типа
    base_cputype = cputype & ~_MACHO_ABI64_FLAG
    return base_cputype in _MACHO_KNOWN_CPUTYPES and 0 < ncmds < _MACHO_NCMDS_MAX


def find_signatures_in_bytes(
    data: bytes,
    *,
    base_offset: int = 0,
    estimated_size: int | None = None,
) -> list[EmbeddedSignature]:
    """Ищет сигнатуры исполняемых файлов в байтовом буфере.

    Используется для in-memory данных (распакованный pickle-поток, член ZIP,
    декомпрессированный joblib-блок). Для больших файлов на диске используйте
    :func:`find_signatures_in_file`.

    Реализация (аудит #7): один проход скомпилированным regex'ом с alternation
    вместо N полных проходов через ``data.find()``. Старая реализация делала
    7 полных сканов 2-ГБ буфера = ~14 ГБ работы; новая — один проход
    с автоматной семантикой. На пустом буфере поведение не отличается.

    После нахождения MZ / \\x7fELF применяется структурная валидация
    (:func:`_validate_pe`, :func:`_validate_elf`), которая отфильтровывает
    случайные совпадения в бинарных данных тензоров.

    :param data: Буфер байт для сканирования.
    :param base_offset: Смещение, добавляемое к каждому найденному offset
        (для отчётности — например, offset внутри родительского архива).
    :param estimated_size: Примерный размер исполняемого файла, если известен;
        попадает в EmbeddedSignature.size. Если None — используется длина буфера
        от offset до конца как верхняя оценка.
    :return: Список EmbeddedSignature, отсортированный по offset.
    """
    if not data:
        return []

    findings: list[EmbeddedSignature] = []
    data_len = len(data)

    for match in _SIGNATURE_PATTERN.finditer(data):
        pos = match.start()
        sig = match.group(0)
        sig_type, _desc = _SIGNATURES[sig]

        # Структурная валидация — исключает случайные совпадения в тензорах.
        if sig_type == "PE" and not _validate_pe(data, pos):
            continue
        if sig_type == "ELF" and not _validate_elf(data, pos):
            continue
        if sig_type == "Mach-O":
            is_fat = sig == b"\xca\xfe\xba\xbe"
            is_le = sig in (b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe")
            is_be = sig in (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf")
            valid = (
                (is_fat and _validate_macho_fat(data, pos))
                or (is_le and _validate_macho_single(data, pos, little_endian=True))
                or (is_be and _validate_macho_single(data, pos, little_endian=False))
            )
            if not valid:
                continue

        size = estimated_size if estimated_size is not None else data_len - pos
        findings.append(
            EmbeddedSignature(
                signature_type=sig_type,
                offset=base_offset + pos,
                size=size,
            )
        )
        if len(findings) >= _MAX_FINDINGS:
            logger.warning(
                "find_signatures_in_bytes: достигнут лимит %d находок — "
                "оставшиеся сигнатуры не будут возвращены",
                _MAX_FINDINGS,
            )
            break

    # Уже отсортированы по offset (finditer возвращает в порядке появления).
    return findings


def find_signatures_in_file(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> list[EmbeddedSignature]:
    """Ищет сигнатуры в файле на диске потоково (без загрузки в RAM).

    Используется сканерами, которые работают с большими файлами (GGUF 30–70 ГБ,
    PyTorch ZIP). Файл читается чанками по 1 МБ с оверлапом 4 байта, чтобы
    не потерять сигнатуру на стыке чанков.

    :param path: Путь к файлу.
    :param max_bytes: Максимальное количество байт для сканирования
        (None = весь файл). Используется для GGUF: тензорные данные занимают
        99% объёма, и хотя теоретически PE/ELF может быть и там, на практике
        атакующему дешевле положить его в metadata, поэтому достаточно
        сканировать первые несколько ГБ или указанный лимит.
    :return: Список EmbeddedSignature, отсортированный по offset.
    """
    findings: list[EmbeddedSignature] = []

    try:
        with path.open("rb") as fh:
            yield_count = 0
            for chunk_offset, chunk in _iter_chunks_with_overlap(fh, max_bytes):
                # estimated_size: считаем что PE начинается тут и тянется до
                # конца этого чанка. Реальный размер неизвестен без разбора.
                local = find_signatures_in_bytes(
                    chunk,
                    base_offset=chunk_offset,
                    estimated_size=len(chunk),
                )
                findings.extend(local)
                yield_count += len(local)
                if yield_count >= _MAX_FINDINGS:
                    break
    except OSError as exc:
        logger.warning(
            "find_signatures_in_file: не удалось прочитать %s: %s", path, exc
        )
        return findings

    # Дедупликация по offset (сигнатура на границе чанков может быть найдена
    # дважды — в конце предыдущего чанка и в начале следующего из-за оверлапа).
    seen_offsets: set[int] = set()
    deduped: list[EmbeddedSignature] = []
    for f in sorted(findings, key=lambda x: x.offset):
        if f.offset in seen_offsets:
            continue
        seen_offsets.add(f.offset)
        deduped.append(f)
        if len(deduped) >= _MAX_FINDINGS:
            break

    return deduped


def _iter_chunks_with_overlap(
    fh: IO[bytes],
    max_bytes: int | None,
) -> Iterator[tuple[int, bytes]]:
    """Итерируется чанками файла с оверлапом длины максимальной сигнатуры.

    Yield: (абсолютное смещение начала чанка, содержимое чанка).
    Оверлап нужен, чтобы сигнатура, разрезанная границей чанков, всё равно
    нашлась в одном из них.
    """
    pos = 0
    tail = b""

    while True:
        if max_bytes is not None and pos >= max_bytes:
            return

        to_read = _CHUNK_SIZE
        if max_bytes is not None:
            to_read = min(to_read, max_bytes - pos)

        chunk = fh.read(to_read)
        if not chunk:
            return

        # Объединяем хвост предыдущего чанка с новым; начало этого комбинированного
        # буфера соответствует pos - len(tail) в исходном файле.
        combined = tail + chunk
        chunk_start = pos - len(tail)
        yield chunk_start, combined

        # Готовим хвост для следующей итерации
        tail = combined[-_OVERLAP:] if len(combined) > _OVERLAP else combined

        pos += len(chunk)

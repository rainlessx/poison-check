"""Сканер joblib-файлов: .joblib и .pkl созданных через joblib.dump."""

from __future__ import annotations

import bz2
import gzip
import logging
import lzma
import re
import zlib
from pathlib import Path
from typing import Any, ClassVar

from poison_check.core.registry import ScannerRegistry
from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Лимит декомпрессии: защита от decompression bomb атак
# ---------------------------------------------------------------------------

#: Максимально допустимый размер декомпрессированных данных (2 ГБ) — абсолютный
#: потолок. При превышении на НИЗКОМ коэффициенте распаковки (просто очень
#: большая модель) факт фиксируется как joblib_bomb_kind="absolute" — Issue
#: MLS-JOBLIB-003 (MEDIUM) эмитит JoblibMetadataDetector.
MAX_DECOMP: int = 2 * 1024 * 1024 * 1024  # 2 GB

#: Пороговый коэффициент распаковки (распакованный / сжатый) для распознавания
#: decompression bomb ДО разбора pickle. Обоснование калибровки:
#: joblib-контейнер распаковывается в pickle-поток численной модели (веса —
#: float32/float64, высокоэнтропийны) и сжимается лишь в ~1–5 раз; даже
#: повторяющиеся структуры sklearn-деревьев редко дают выше ~50×. Физический
#: потолок zlib/gzip (DEFLATE) — ~1032×, достижимый только на прогонах
#: одинаковых байт, что и есть сигнатура крафтовой bomb. Порог 500× лежит
#: заведомо выше реалистичных легитимных коэффициентов и ниже потолка DEFLATE,
#: давая устойчивое распознавание bomb без ложных срабатываний. Это отдельный
#: pre-parse-контроль сканера, не путать с post-parse ratio CompressionDetector
#: (тот считает размер вложенного pickle, а не сырую распаковку).
BOMB_RATIO_THRESHOLD: int = 500

#: Абсолютный пол распакованного объёма (8 МБ): пока распаковка не превысила
#: этот размер, ratio-порог не применяется. Защищает крошечные легитимные
#: joblib-файлы (типичный sklearn pickle — сотни КБ–единицы МБ), у которых
#: коэффициент может случайно всплеснуть, но абсолютный объём не несёт риска
#: OOM. Ниже этого пола файл никогда не считается bomb.
BOMB_OUTPUT_FLOOR: int = 8 * 1024 * 1024  # 8 MB

#: Размер чанка для инкрементальной декомпрессии (64 КБ).
_DECOMP_CHUNK: int = 65536

# ---------------------------------------------------------------------------
# Ключи фактов, которые сканер оставляет в metadata для JoblibMetadataDetector
# ---------------------------------------------------------------------------
# Разделение слоёв: сканер знает ФОРМАТ и фиксирует факт, детектор знает УГРОЗУ
# и эмитит Issue. Сканер не конструирует Issue сам (RawScanData не имеет поля
# issues — раньше находки MLS-JOBLIB-001/002/003 молча терялись).

#: Отсутствует опциональный кодек ("lz4" | "zstd") — MLS-JOBLIB-001/002 (INFO).
META_MISSING_CODEC: str = "joblib_missing_codec"

#: Сработала bomb-защита при декомпрессии — Issue эмитит JoblibMetadataDetector.
META_DECOMPRESSION_BOMB: str = "joblib_decompression_bomb"

#: Тип bomb-срабатывания: "ratio" (крафтовая bomb, крошечный вход → огромное
#: раскрытие, ratio ≥ BOMB_RATIO_THRESHOLD) → MLS-BOMB-001 (HIGH); "absolute"
#: (низкий ratio, но упёрлись в MAX_DECOMP — просто гигантская модель) →
#: MLS-JOBLIB-003 (MEDIUM). Отсутствие ключа детектор трактует как "absolute".
META_BOMB_KIND: str = "joblib_bomb_kind"

#: Наблюдённый коэффициент распаковки на момент срабатывания ratio-bomb (строка).
META_BOMB_RATIO: str = "joblib_bomb_ratio"

#: Сжатый размер файла в байтах на момент срабатывания bomb (для details Issue).
META_BOMB_COMPRESSED_BYTES: str = "joblib_bomb_compressed_bytes"

#: Прочитанный распакованный объём в байтах на момент прерывания (для details).
META_BOMB_DECOMPRESSED_BYTES: str = "joblib_bomb_decompressed_bytes"

#: Метод компрессии, на котором сработала bomb-защита (для details Issue).
META_BOMB_METHOD: str = "joblib_bomb_method"

#: Абсолютный лимит декомпрессии в байтах на момент срабатывания (для details).
META_BOMB_LIMIT_BYTES: str = "joblib_bomb_limit_bytes"

# ---------------------------------------------------------------------------
# Magic-байты форматов сжатия (из joblib/compressor.py)
# ---------------------------------------------------------------------------

# Устаревший ZF-формат joblib (до версии 0.9.3): b"ZF" + длина в hex
_ZFILE_PREFIX: bytes = b"ZF"

# Современные форматы — просто стандартные заголовки сжатых потоков
_ZLIB_PREFIX: bytes = b"\x78"        # zlib (0x78 0x9C, 0x78 0x01, 0x78 0x5E, 0x78 0xDA)
_GZIP_PREFIX: bytes = b"\x1f\x8b"   # gzip
_BZ2_PREFIX: bytes = b"BZ"          # bz2
_XZ_PREFIX: bytes = b"\xfd7zXZ"     # xz (LZMA XZ format)
_LZMA_PREFIX: bytes = b"]\x00"      # lzma (FORMAT_ALONE)
_LZ4_PREFIX: bytes = b"\x04\x22\x4d\x18"  # lz4 frame

# Zstd magic: 0xFD2FB528 в little-endian
_ZSTD_PREFIX: bytes = b"\x28\xb5\x2f\xfd"

# Pickle magic: первый байт 0x80, второй — версия протокола (2–5)
_PICKLE_PROTO_BYTE: int = 0x80

# Однобайтовый маркер PROTO-опкода для поиска через bytes.find (без срезов).
_PROTO_MARKER: bytes = bytes((_PICKLE_PROTO_BYTE,))

# Версии pickle-протокола, которые начинаются с PROTO-опкода (0x80 + версия).
_PICKLE_PROTO_VERSIONS: frozenset[int] = frozenset((2, 3, 4, 5))

# ---------------------------------------------------------------------------
# Лимиты защиты от resync-бомбы (квадратичный DoS)
# ---------------------------------------------------------------------------
# Старая реализация _resync_pickle_frames шла по КАЖДОМУ байту и на каждом
# PROTO-маркере вызывала scan_bytes(data[i:]) — копия всего хвоста + полный
# повторный парсинг. Файл в 100 МБ из \x80\x02 при ранней parse error давал
# десятки млн вызовов на копиях до 100 МБ → зависание/OOM. Новая реализация
# ищет маркеры одним проходом (bytes.find), парсит окно фиксированного размера
# и ограничивает число попыток и суммарный объём.

#: Максимум попыток повторного парсинга (вызовов scan_bytes) при resync.
_MAX_RESYNC_ATTEMPTS: int = 64

#: Размер окна повторного парсинга одного PROTO-маркера (1 МБ). Достаточно, чтобы
#: захватить вредоносный global во втором фрейме, но не копирует весь хвост.
_RESYNC_WINDOW: int = 1024 * 1024

#: Суммарный объём повторно распарсенных байт (защита от накопления копий).
_MAX_RESYNC_TOTAL: int = 16 * 1024 * 1024

#: Максимум PROTO-маркеров (0x80), проверяемых при поиске. Ограничивает flood из
#: 0x80 с невалидной версией, который иначе гонял бы find по всему потоку.
_MAX_RESYNC_MARKERS: int = 1 << 16  # 65536

# Длина зарезервированной строки в заголовке ZF-файла (hex uint64 = 18 символов)
_ZFILE_HEX_LEN: int = 18  # len(hex_str(2**64)) = len("0x10000000000000000") = 19 → joblib uses 18

# Паттерны для поиска embedded source-code payload после ошибки парсинга pickle.
# Используются _string_scan_payload при parse-stop атаке.
_SUSPICIOUS_PAYLOAD_PATTERNS: tuple[bytes, ...] = (
    b"import os",
    b"import socket",
    b"import subprocess",
    b"reverse_shell",
    b"s.connect(",
    b"os.system(",
    b"subprocess.Popen(",
    b"__import__(",
)


@ScannerRegistry.register
class JoblibScanner(BaseScanner):
    """Сканер joblib-файлов (.joblib, .pkl сохранённых через joblib.dump).

    Joblib не использует собственный бинарный формат контейнера — вместо этого
    он пишет данные напрямую в один из поддерживаемых форматов сжатия
    (или без сжатия — тогда это чистый pickle-поток). Сканер определяет
    метод компрессии по magic bytes, декомпрессирует данные и передаёт
    полученный pickle-поток в PickleScanner.scan_bytes().
    """

    name = "joblib"
    description = "Сканер joblib-файлов (.joblib, .pkl созданных через joblib.dump)"
    supported_extensions = [".joblib", ".pkl"]
    magic_bytes: ClassVar[list[bytes]] = []  # определяется динамически по содержимому

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Проверяет расширение файла и (для .pkl) magic bytes сжатия.

        Joblib не имеет единственного magic bytes. Сжатые joblib-файлы начинаются
        с заголовка соответствующего кодека (zlib/gzip/bz2/xz/lz4/zstd/ZF), а
        несжатые — с pickle PROTO opcode (0x80). Чтобы не перехватывать чистые
        pickle-файлы у PickleScanner (нарушение разделения слоёв и потеря корректного
        scanner_name="pickle" в RawScanData):

        * ``.joblib`` — забираем всегда (это наш формат по определению).
        * ``.pkl`` — забираем только если файл явно сжат одним из joblib-кодеков.
          Plain pickle (0x80 + 0x02..0x05) или Python 2/protocol-0/1 pickle
          оставляем PickleScanner.
        """
        suffix = path.suffix.lower()
        if suffix not in cls.supported_extensions:
            return False
        if suffix == ".joblib":
            return True
        # .pkl — забираем только сжатые joblib-файлы
        try:
            with path.open("rb") as fh:
                head = fh.read(8)
        except OSError:
            return False
        return cls._has_joblib_compression_magic(head)

    @staticmethod
    def _has_joblib_compression_magic(head: bytes) -> bool:
        """Проверяет, начинается ли буфер с magic bytes одного из joblib-кодеков.

        Не считает чистый pickle joblib-файлом (он должен попасть в PickleScanner).
        """
        if not head:
            return False
        # Проверяем заголовки сжатых joblib-форматов в порядке убывания длины prefix
        if head[:4] == _LZ4_PREFIX:
            return True
        if head[:4] == _ZSTD_PREFIX:
            return True
        if head[:5] == _XZ_PREFIX:
            return True
        if head[:2] == _ZFILE_PREFIX:  # legacy ZF (joblib < 0.9.3)
            return True
        if head[:2] == _GZIP_PREFIX:
            return True
        if head[:2] == _BZ2_PREFIX:
            return True
        if head[:2] == _LZMA_PREFIX:
            return True
        # zlib: первый байт 0x78, второй из {0x01, 0x5E, 0x9C, 0xDA}
        return (
            head[:1] == _ZLIB_PREFIX
            and len(head) >= 2
            and head[1] in (0x01, 0x5E, 0x9C, 0xDA)
        )

    def scan(self, path: Path) -> RawScanData:
        """Читает joblib-файл, декомпрессирует и сканирует pickle-поток.

        Алгоритм (аудит #3 — потоковая обработка):
        1. Читает первые 4096 байт для определения формата компрессии.
        2. Открывает файл как file handle и стримит декомпрессию: gzip / bz2 /
           lzma / lz4 / zstd принимают file-like напрямую; zlib и compat
           работают через decompressobj() с инкрементальным чтением.
        3. Декомпрессированные данные накапливаются в bytearray с проверкой
           MAX_DECOMP лимита — защита от decompression bomb.
        4. Передаёт pickle-поток в PickleScanner.scan_bytes().
        5. При любой ошибке возвращает RawScanData с error — не бросает исключений.

        Старая реализация делала ``b"".join(chunks)`` на всём файле — для
        joblib XGBoost/LightGBM (5–20 ГБ) это OOM. Новая держит в RAM только
        декомпрессированные данные (≤ MAX_DECOMP).
        """
        # Ленивый импорт, чтобы избежать циклических зависимостей
        # (joblib_scanner → pickle_scanner, но не наоборот)
        from poison_check.scanners.pickle_scanner import PickleScanner

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

        # Читаем только preamble (для определения формата + raw_content_sample)
        try:
            with path.open("rb") as fh:
                preamble = fh.read(4096)
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Ошибка чтения файла: {exc}",
            )

        compression_method = self._detect_compression(preamble)

        if compression_method == "unknown":
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=(
                    f"Неизвестный формат файла: первые байты {preamble[:8]!r}. "
                    "Файл не распознан как joblib или pickle."
                ),
                metadata={"joblib_compression": "unknown"},
            )

        decompressed, decomp_error, decomp_facts = self._decompress_stream(
            path, compression_method, file_size,
        )

        if decompressed is None:
            # Ранний возврат: bomb (error задан) или отсутствие кодека (error=None).
            # В обоих случаях факт из decomp_facts должен попасть в metadata,
            # иначе JoblibMetadataDetector не увидит MLS-JOBLIB-001/002/003.
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=decomp_error,
                metadata={
                    "joblib_compression": compression_method,
                    **decomp_facts,
                },
            )

        # Передаём распакованный pickle-поток в PickleScanner.
        # Сканер pickle сам найдёт PE/ELF в распакованных данных и заполнит
        # inner.embedded_bytes — отдельный поиск здесь не нужен.
        pickle_scanner = PickleScanner()
        inner = pickle_scanner.scan_bytes(decompressed, source_path=path)

        # Parse-stop bypass: если pickle-парсинг прервался (напр. на raw NumpyArrayWrapper
        # байтах), ищем дополнительные фреймы после точки ошибки и embedded source payload.
        extra_globals: set[tuple[str, str]] = set()
        embedded_strings: list[str] = []
        if inner.error:
            pos_match = re.search(r"at position (\d+)", inner.error)
            error_pos = int(pos_match.group(1)) if pos_match else 0
            extra_globals = self._resync_pickle_frames(decompressed, error_pos, path)
            embedded_strings = self._string_scan_payload(decompressed, error_pos)

        all_globals = set(inner.globals or []) | extra_globals
        meta: dict[str, Any] = {
            "joblib_compression": compression_method,
            **decomp_facts,  # на успешном пути пусто; включено для полноты
            **(inner.metadata or {}),
        }
        if extra_globals or embedded_strings:
            meta["parse_stop_attack"] = True
        if embedded_strings:
            meta["embedded_strings"] = embedded_strings

        # Объединяем: возвращаем RawScanData верхнего уровня со всеми полями pickle,
        # плюс nested_files для forensics (там оригинальный inner)
        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=inner.opcodes,
            globals=all_globals if all_globals else None,
            strings=inner.strings,
            reduce_calls=inner.reduce_calls,
            embedded_bytes=inner.embedded_bytes,
            metadata=meta,
            nested_files=[inner],
            raw_content_sample=preamble,
            error=inner.error or (decomp_error if decomp_error else None),
        )

    # ------------------------------------------------------------------
    # Parse-stop bypass: resync и string-scan
    # ------------------------------------------------------------------

    @staticmethod
    def _resync_pickle_frames(
        data: bytes, error_pos: int, source_path: Path
    ) -> set[tuple[str, str]]:
        """Ищет дополнительные pickle-фреймы после позиции ошибки.

        Защита от parse-stop атаки: первый FRAME прерывается сырыми байтами
        (например, inline-данными NumpyArrayWrapper), затем следует второй
        pickle-фрейм с вредоносным payload. pickletools.genops останавливается
        и не видит второй фрейм. Этот метод ищет PROTO-маркеры (0x80 + 2-5)
        после error_pos и парсит каждый фрагмент через PickleScanner.

        DoS-стойкость (аудит #6): маркеры ищутся одним проходом через
        ``bytes.find`` (без итерации по каждому байту), парсится окно
        фиксированного размера :data:`_RESYNC_WINDOW` (без среза всего хвоста
        ``data[i:]``), а число попыток и суммарный объём ограничены
        (:data:`_MAX_RESYNC_ATTEMPTS`, :data:`_MAX_RESYNC_TOTAL`,
        :data:`_MAX_RESYNC_MARKERS`). Это исключает квадратичное поведение на
        «resync-бомбе» — потоке из тысяч ``\\x80\\x02`` с ранней parse error.

        Вредоносный global во втором фрейме собирается инкрементально ещё до
        STOP-опкода, поэтому обрезка окна не мешает его детекции.
        """
        from poison_check.scanners.pickle_scanner import PickleScanner

        extra: set[tuple[str, str]] = set()
        scanner = PickleScanner()

        data_len = len(data)
        pos = max(0, error_pos + 1)
        attempts = 0
        reparsed_total = 0
        markers_seen = 0

        while (
            attempts < _MAX_RESYNC_ATTEMPTS
            and reparsed_total < _MAX_RESYNC_TOTAL
            and markers_seen < _MAX_RESYNC_MARKERS
        ):
            marker = data.find(_PROTO_MARKER, pos)
            # Нет следующего маркера, либо 0x80 — последний байт (нет версии).
            if marker < 0 or marker >= data_len - 1:
                break
            markers_seen += 1
            if data[marker + 1] in _PICKLE_PROTO_VERSIONS:
                # Окно фиксированного размера — не копируем весь хвост потока.
                window = data[marker : marker + _RESYNC_WINDOW]
                reparsed_total += len(window)
                attempts += 1
                result = scanner.scan_bytes(window, source_path)
                if result.globals:
                    extra |= set(result.globals)
                # Продолжаем сразу за версией — не переисследуем тот же маркер.
                pos = marker + 2
            else:
                pos = marker + 1

        return extra

    @staticmethod
    def _string_scan_payload(data: bytes, start: int = 0) -> list[str]:
        """Ищет признаки embedded source-code payload в сырых байтах.

        Используется при parse-stop атаке, когда вредоносный код вставлен
        как исходный текст (ASCII), а не через pickle-глобал. Сканирует
        байты начиная с позиции ``start`` (по умолчанию с начала).

        Поиск идёт через ``bytes.find(pattern, start)`` — без среза хвоста
        ``data[start:]``, чтобы не копировать потенциально гигантский поток.
        """
        found: list[str] = []
        seen: set[str] = set()
        begin = max(0, start)
        for pattern in _SUSPICIOUS_PAYLOAD_PATTERNS:
            if data.find(pattern, begin) != -1:
                decoded = pattern.decode("ascii", errors="replace")
                if decoded not in seen:
                    seen.add(decoded)
                    found.append(decoded)
        return found

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_compression(data: bytes) -> str:
        """Определяет метод компрессии по magic bytes.

        Возвращает метод — одну из строк:
        'none', 'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'lz4', 'zstd', 'compat', 'unknown'.
        'none' означает raw pickle-поток без компрессии.
        'compat' означает устаревший ZF-формат joblib (до версии 0.9.3).
        """
        if len(data) == 0:
            return "unknown"

        # Устаревший ZF-формат (joblib < 0.9.3)
        if data[:2] == _ZFILE_PREFIX:
            return "compat"

        # LZ4 (4 байта prefix)
        if data[:4] == _LZ4_PREFIX:
            return "lz4"

        # Zstd (4 байта prefix)
        if data[:4] == _ZSTD_PREFIX:
            return "zstd"

        # XZ (5 байт prefix)
        if data[:5] == _XZ_PREFIX:
            return "xz"

        # GZip (2 байта)
        if data[:2] == _GZIP_PREFIX:
            return "gzip"

        # BZ2 (2 байта 'BZ')
        if data[:2] == _BZ2_PREFIX:
            return "bz2"

        # LZMA FORMAT_ALONE (2 байта ']' + '\x00')
        if data[:2] == _LZMA_PREFIX:
            return "lzma"

        # Zlib (первый байт 0x78, второй — уровень/стратегия)
        # Допустимые вторые байты: 0x01, 0x5E, 0x9C, 0xDA
        if data[:1] == _ZLIB_PREFIX and len(data) >= 2 and data[1] in (
            0x01, 0x5E, 0x9C, 0xDA
        ):
            return "zlib"

        # Raw pickle (первый байт 0x80 — PROTO opcode)
        if data[0] == _PICKLE_PROTO_BYTE and len(data) >= 2 and data[1] in range(2, 6):
            return "none"

        # Pickle протокол 0 или 1 (без PROTO opcode): первый байт — opcode
        # Характерные первые байты: '(' (MARK), '}' (EMPTY_DICT), ']' (EMPTY_LIST),
        # ')' (EMPTY_TUPLE), 'l' (LIST), 'd' (DICT), 'c' (GLOBAL)...
        # Проверяем по наличию b'.' (STOP) в данных — очень слабая эвристика,
        # но достаточная для определения формата.
        if data[0] in b"()[]{}cdlti" and b"." in data:
            return "none"

        return "unknown"

    @staticmethod
    def _decompress_compat(data: bytes) -> bytes:
        """Декомпрессирует устаревший ZF-формат joblib (zlib с заголовком длины).

        Формат: b'ZF' + hex-длина (18 символов, lj-justified) + zlib-данные.
        """
        # Заголовок: 'ZF' (2) + hex-число из 18 символов + возможный пробел
        header_end = 2 + _ZFILE_HEX_LEN
        length_str = data[2:header_end].rstrip()
        try:
            expected_length = int(length_str, 16)
        except ValueError as exc:
            raise ValueError(
                f"Некорректная длина в ZF-заголовке: {length_str!r}"
            ) from exc

        # Пропускаем возможный пробел-разделитель
        zlib_start = header_end
        if zlib_start < len(data) and data[zlib_start:zlib_start + 1] == b" ":
            zlib_start += 1

        # Используем инкрементальную декомпрессию с ограничением размера
        decomp_obj = zlib.decompressobj(15)
        return JoblibScanner._stream_decompress_bytes(
            data[zlib_start:], decomp_obj, "compat/zlib", max_size=expected_length
        )

    @staticmethod
    def _stream_decompress_bytes(
        data: bytes,
        decomp_obj: Any,
        method: str,
        max_size: int | None = None,
    ) -> bytes:
        """Инкрементально декомпрессирует bytes через объект zlib.Decompress.

        Используется для zlib и ZF-compat форматов.

        Args:
            data: Сжатые байты.
            decomp_obj: Объект декомпрессии (zlib.decompressobj).
            method: Название метода для сообщений об ошибках.
            max_size: Лимит размера результата (None = MAX_DECOMP).

        Raises:
            ValueError: При превышении лимита размера.
            zlib.error: При ошибке декомпрессии.
        """
        limit = max_size if max_size is not None else MAX_DECOMP
        output = bytearray()
        pos = 0
        while pos < len(data):
            chunk = decomp_obj.decompress(data[pos : pos + _DECOMP_CHUNK])
            output.extend(chunk)
            if len(output) > limit:
                raise ValueError(
                    f"Декомпрессия {method} превысила лимит "
                    f"{MAX_DECOMP // 1_000_000_000:.0f} ГБ — "
                    "возможная decompression bomb атака."
                )
            pos += _DECOMP_CHUNK
        # Flush оставшегося буфера
        tail = decomp_obj.flush()
        output.extend(tail)
        if len(output) > limit:
            raise ValueError(
                f"Декомпрессия {method} превысила лимит "
                f"{MAX_DECOMP // 1_000_000_000:.0f} ГБ — "
                "возможная decompression bomb атака."
            )
        return bytes(output)

    @staticmethod
    def _effective_decomp_limit(compressed_size: int) -> int:
        """Верхняя граница распакованного объёма, до которой сканер аллоцирует.

        Выбирается как отношение к сжатому размеру (ratio-порог
        :data:`BOMB_RATIO_THRESHOLD`), но не ниже абсолютного пола
        :data:`BOMB_OUTPUT_FLOOR` и не выше абсолютного потолка
        :data:`MAX_DECOMP`. Как только распаковка переваливает за этот лимит,
        а поток не закончился — это сигнал bomb: сканер прекращает чтение,
        поэтому полный объём bomb в память не аллоцируется (потоковый инвариант).
        Значения берутся динамически (учитывают monkeypatch в тестах).
        """
        ratio_cap = compressed_size * BOMB_RATIO_THRESHOLD
        return min(MAX_DECOMP, max(ratio_cap, BOMB_OUTPUT_FLOOR))

    @staticmethod
    def _bomb_over_limit(
        output_len: int, compressed_size: int, method: str
    ) -> tuple[str, dict[str, str]]:
        """Классифицирует превышение лимита декомпрессии и строит (error, facts).

        Сканер только фиксирует ФАКТ (зона Scanner) — Issue конструирует
        JoblibMetadataDetector (зона Detector):

        * ``ratio >= BOMB_RATIO_THRESHOLD`` → крафтовая decompression bomb
          (крошечный вход, огромное раскрытие) → ``joblib_bomb_kind="ratio"`` →
          MLS-BOMB-001 (HIGH).
        * иначе (низкий ratio, но упёрлись в MAX_DECOMP — просто гигантская
          модель) → ``joblib_bomb_kind="absolute"`` → MLS-JOBLIB-003 (MEDIUM).
        """
        ratio = output_len / max(compressed_size, 1)
        if ratio >= BOMB_RATIO_THRESHOLD:
            facts = {
                META_DECOMPRESSION_BOMB: "true",
                META_BOMB_KIND: "ratio",
                META_BOMB_METHOD: method,
                META_BOMB_RATIO: f"{ratio:.0f}",
                META_BOMB_COMPRESSED_BYTES: str(compressed_size),
                META_BOMB_DECOMPRESSED_BYTES: str(output_len),
            }
            error = (
                f"Decompression bomb ({method}): распаковано ≥ {output_len} байт "
                f"из {compressed_size} (соотношение ~{ratio:.0f}x), "
                "чтение прервано."
            )
            return error, facts
        facts = {
            META_DECOMPRESSION_BOMB: "true",
            META_BOMB_KIND: "absolute",
            META_BOMB_METHOD: method,
            META_BOMB_LIMIT_BYTES: str(MAX_DECOMP),
        }
        error = (
            f"Декомпрессия {method} превысила лимит "
            f"{MAX_DECOMP // 1_000_000_000:.0f} ГБ"
        )
        return error, facts

    def _decompress_stream(
        self,
        path: Path,
        method: str,
        compressed_size: int,
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Потоковая декомпрессия: читает файл с диска через file handle.

        Не загружает исходный сжатый файл в RAM целиком. На joblib XGBoost
        (5–20 ГБ сжатый) наивный ``b"".join(chunks)`` валился с OOM ещё до
        декомпрессии.

        Все ветки разворачиваются с проверкой MAX_DECOMP — защита от bomb.
        Возвращает ``(decompressed_bytes | None, error | None, facts)``, где
        ``facts`` — словарь фактов для metadata (пустой при успехе). При bomb
        ставится :data:`META_DECOMPRESSION_BOMB` и error; при отсутствии кодека
        (lz4/zstd) — :data:`META_MISSING_CODEC` без error. Issue по этим фактам
        эмитит JoblibMetadataDetector — сканер сам Issue не конструирует.
        """
        try:
            if method == "none":
                # Нет сжатия — просто читаем файл с лимитом, чтобы не вытащить
                # всё в RAM, если он гигантский. Здесь ratio ≈ 1, поэтому
                # эффективный лимит упирается в MAX_DECOMP (absolute).
                limit = self._effective_decomp_limit(compressed_size)
                with path.open("rb") as fh:
                    out = bytearray()
                    while True:
                        chunk = fh.read(_DECOMP_CHUNK)
                        if not chunk:
                            break
                        out.extend(chunk)
                        if len(out) > limit:
                            error, facts = self._bomb_over_limit(
                                len(out), compressed_size, "none"
                            )
                            return None, error, facts
                    return bytes(out), None, {}

            if method == "compat":
                # Legacy ZF: читаем файл целиком в bytes (формат включает
                # заголовок длины — потоково парсить без рефакторинга нельзя).
                # Лимит на этот случай — проверка размера файла уже сделана
                # ранее через _check_file_size.
                data = path.read_bytes()
                try:
                    return self._decompress_compat(data), None, {}
                except ValueError as exc:
                    msg = str(exc)
                    compat_facts: dict[str, str] = {}
                    if "decompression bomb" in msg.lower() or "превысила лимит" in msg:
                        # Legacy ZF идёт через _stream_decompress_bytes с
                        # абсолютным max_size — это absolute-тип (MLS-JOBLIB-003).
                        compat_facts = {
                            META_DECOMPRESSION_BOMB: "true",
                            META_BOMB_KIND: "absolute",
                            META_BOMB_METHOD: "compat/zlib",
                            META_BOMB_LIMIT_BYTES: str(MAX_DECOMP),
                        }
                    return None, msg, compat_facts
                except zlib.error as exc:
                    return None, f"Ошибка декомпрессии ZF (legacy joblib): {exc}", {}

            if method == "zlib":
                return self._stream_zlib(path, compressed_size)

            if method == "gzip":
                return self._stream_gzip(path, compressed_size)

            if method == "bz2":
                return self._stream_bz2(path, compressed_size)

            if method in ("lzma", "xz"):
                return self._stream_lzma(path, method, compressed_size)

            if method == "lz4":
                return self._stream_lz4(path, compressed_size)

            if method == "zstd":
                return self._stream_zstd(path, compressed_size)

            return None, f"Неподдерживаемый метод компрессии: {method!r}", {}
        except OSError as exc:
            return None, f"Ошибка чтения файла: {exc}", {}

    def _stream_zlib(
        self, path: Path, compressed_size: int
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Потоковый zlib через decompressobj с ограничением ВЫВОДА на шаг.

        Один сжатый чанк может раскрыться в десятки МБ (bomb). ``decompress``
        с ``max_length=_DECOMP_CHUNK`` не даёт аллоцировать это разом — остаток
        входа хранится в ``unconsumed_tail`` и дожёвывается тем же лимитом.
        Как только суммарный вывод переваливает эффективный лимит — читаем
        стоп (потоковый инвариант: полный объём bomb не аллоцируется).
        """
        limit = self._effective_decomp_limit(compressed_size)
        try:
            decomp = zlib.decompressobj()
            output = bytearray()
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(_DECOMP_CHUNK)
                    if not chunk:
                        break
                    data = chunk
                    while data:
                        piece = decomp.decompress(data, _DECOMP_CHUNK)
                        output.extend(piece)
                        if len(output) > limit:
                            error, facts = self._bomb_over_limit(
                                len(output), compressed_size, "zlib"
                            )
                            return None, error, facts
                        data = decomp.unconsumed_tail
                output.extend(decomp.flush())
            if len(output) > limit:
                error, facts = self._bomb_over_limit(
                    len(output), compressed_size, "zlib"
                )
                return None, error, facts
            return bytes(output), None, {}
        except zlib.error as exc:
            return None, f"Ошибка декомпрессии zlib: {exc}", {}

    def _stream_gzip(
        self, path: Path, compressed_size: int
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Потоковый gzip — ``GzipFile.read(size)`` уже ограничивает вывод шага."""
        limit = self._effective_decomp_limit(compressed_size)
        try:
            output = bytearray()
            with path.open("rb") as fh, gzip.GzipFile(fileobj=fh) as gf:
                while True:
                    chunk = gf.read(_DECOMP_CHUNK)
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > limit:
                        error, facts = self._bomb_over_limit(
                            len(output), compressed_size, "gzip"
                        )
                        return None, error, facts
            return bytes(output), None, {}
        except (OSError, EOFError) as exc:
            return None, f"Ошибка декомпрессии gzip: {exc}", {}

    def _stream_bz2(
        self, path: Path, compressed_size: int
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Потоковый bz2 с ограничением вывода шага через max_length."""
        limit = self._effective_decomp_limit(compressed_size)
        try:
            return self._drain_incremental(
                bz2.BZ2Decompressor(), path, limit, "bz2", compressed_size
            )
        except OSError as exc:
            return None, f"Ошибка декомпрессии bz2: {exc}", {}

    def _stream_lzma(
        self, path: Path, method: str, compressed_size: int
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Потоковый lzma/xz с ограничением вывода шага через max_length."""
        limit = self._effective_decomp_limit(compressed_size)
        try:
            return self._drain_incremental(
                lzma.LZMADecompressor(), path, limit, method, compressed_size
            )
        except lzma.LZMAError as exc:
            return None, f"Ошибка декомпрессии {method}: {exc}", {}

    def _drain_incremental(
        self,
        decomp: Any,
        path: Path,
        limit: int,
        method: str,
        compressed_size: int,
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Инкрементально распаковывает bz2/lzma с ограничением вывода на шаг.

        Оба декомпрессора (``BZ2Decompressor`` / ``LZMADecompressor``) имеют
        одинаковый API ``needs_input`` / ``eof`` и параметр ``max_length``.
        Читаем новый сжатый чанк только когда декомпрессор просит вход, а вывод
        каждого шага ограничен ``_DECOMP_CHUNK`` — bomb-раскрытие не аллоцируется
        разом. При превышении эффективного лимита читаем стоп.
        """
        output = bytearray()
        with path.open("rb") as fh:
            while not decomp.eof:
                if decomp.needs_input:
                    chunk = fh.read(_DECOMP_CHUNK)
                    if not chunk:
                        break  # усечённый поток — отдаём что успели
                    piece = decomp.decompress(chunk, _DECOMP_CHUNK)
                else:
                    piece = decomp.decompress(b"", _DECOMP_CHUNK)
                output.extend(piece)
                if len(output) > limit:
                    error, facts = self._bomb_over_limit(
                        len(output), compressed_size, method
                    )
                    return None, error, facts
        return bytes(output), None, {}

    def _stream_lz4(
        self, path: Path, compressed_size: int
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Потоковый lz4 (если установлена python-lz4).

        Вывод каждого шага ограничен через ``max_length`` (LZ4FrameDecompressor
        поддерживает его) — bomb-раскрытие не аллоцируется разом. Остаток входа
        хранится во внутреннем буфере декомпрессора и дожёвывается пустой
        подачей. При отсутствии библиотеки возвращает факт
        joblib_missing_codec="lz4" (error=None); Issue MLS-JOBLIB-001 (INFO)
        эмитит JoblibMetadataDetector.
        """
        try:
            import lz4.frame
        except ImportError:
            return None, None, {META_MISSING_CODEC: "lz4"}
        limit = self._effective_decomp_limit(compressed_size)
        try:
            decomp = lz4.frame.LZ4FrameDecompressor()
            output = bytearray()
            with path.open("rb") as fh:
                while not decomp.eof:
                    chunk = fh.read(_DECOMP_CHUNK)
                    if not chunk:
                        break
                    # Подаём сжатый чанк, затем дренируем внутренний буфер
                    # порциями по _DECOMP_CHUNK (пустой подачей), проверяя лимит
                    # после каждой — bomb-раскрытие не аллоцируется разом.
                    piece = decomp.decompress(chunk, max_length=_DECOMP_CHUNK)
                    while True:
                        if piece:
                            output.extend(piece)
                            if len(output) > limit:
                                error, facts = self._bomb_over_limit(
                                    len(output), compressed_size, "lz4"
                                )
                                return None, error, facts
                        if decomp.eof or not piece:
                            break
                        piece = decomp.decompress(b"", max_length=_DECOMP_CHUNK)
            return bytes(output), None, {}
        except (RuntimeError, ValueError, OSError) as exc:
            return None, f"Ошибка декомпрессии lz4: {exc}", {}

    def _stream_zstd(
        self, path: Path, compressed_size: int
    ) -> tuple[bytes | None, str | None, dict[str, str]]:
        """Потоковый zstd через ZstdDecompressor.stream_reader.

        ``reader.read(size)`` уже ограничивает вывод шага размером ``size``,
        поэтому bomb-раскрытие не аллоцируется разом. При отсутствии библиотеки
        возвращает факт joblib_missing_codec="zstd" (error=None); Issue
        MLS-JOBLIB-002 (INFO) эмитит JoblibMetadataDetector.
        """
        try:
            import zstandard
        except ImportError:
            return None, None, {META_MISSING_CODEC: "zstd"}
        limit = self._effective_decomp_limit(compressed_size)
        try:
            ctx = zstandard.ZstdDecompressor()
            output = bytearray()
            with path.open("rb") as fh, ctx.stream_reader(fh) as reader:
                while True:
                    chunk = reader.read(_DECOMP_CHUNK)
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > limit:
                        error, facts = self._bomb_over_limit(
                            len(output), compressed_size, "zstd"
                        )
                        return None, error, facts
            return bytes(output), None, {}
        except (zstandard.ZstdError, OSError, ValueError) as exc:
            return None, f"Ошибка декомпрессии zstd: {exc}", {}

"""Сканер pickle-файлов: .pkl, .pickle, .dill, .pt."""

from __future__ import annotations

import contextlib
import io
import logging
import pickletools
from pathlib import Path
from typing import Any, ClassVar

from poison_check.core.container import ContainerError, ContainerExtractor
from poison_check.core.executable_signatures import (
    find_signatures_in_bytes,
    find_signatures_in_file,
)
from poison_check.core.registry import ScannerRegistry
from poison_check.core.result import EmbeddedSignature, OpcodeInfo, ReduceCall, StringInfo
from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

_MIN_STRING_LEN = 4

# ---------------------------------------------------------------------------
# Лимиты накопления (защита от OOM)
# ---------------------------------------------------------------------------
# Чтение потоковое (genops поверх handle), но списки opcodes/strings/reduce_calls
# накапливаются в RAM. Лимит размера файла — 10 ГБ (scanner_base.MAX_FILE_SIZE),
# а pickle из ~2 ГБ однобайтовых опкодов породил бы миллиарды OpcodeInfo → OOM
# (как на злонамеренном, так и на случайно-повреждённом большом файле).
#
# При достижении лимата opcodes разбор аккуратно прерывается: продолжение цикла
# копило бы memo/stack без предела (та же DoS-поверхность). globals_set при этом
# НЕ ограничивается — детекторы должны видеть все опасные глобалы, собранные ДО
# лимита. Само усечение сигнализируется через metadata["opcode_limit_exceeded"];
# предупреждение MLS-PKL-006 эмитирует blocklist_detector, а не сканер.
MAX_OPCODES: int = 2_000_000
MAX_STRINGS: int = 2_000_000
MAX_REDUCE_CALLS: int = 1_000_000

_STRING_OPCODES = frozenset(
    {
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "BINUNICODE8",
        "UNICODE",
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
    }
)

_STACK_PUSH_OPCODES = frozenset(
    {
        "BININT", "BININT1", "BININT2", "INT", "LONG",
        "LONG1", "LONG4", "FLOAT", "BINFLOAT",
        "NONE", "NEWTRUE", "NEWFALSE",
        # GET-семейство убрано отсюда — обрабатывается отдельно с lookup в memo.
    }
)

_NOOP_OPCODES = frozenset(
    {
        "APPEND", "APPENDS", "SETITEM", "SETITEMS",
        "ADDITEMS", "BUILD", "DUP",
        # PUT/MEMOIZE убраны — обрабатываются отдельно для ведения memo.
    }
)

_MARK_SENTINEL = object()


def _is_legit_torch_storage_persid(pid: Any) -> bool:
    """Проверяет, является ли BINPERSID pid ссылкой на PyTorch tensor storage.

    ``torch.save`` для каждого тензора сохраняет persistent_id вида::

        ('storage', <storage_class>, <obj_key>, <location>, <numel>)

    Где ``<storage_class>`` после разбора pickle-потока — это кортеж
    ``('torch', 'FloatStorage'|'DoubleStorage'|…)`` (результат GLOBAL opcode).
    Это стандартный формат сериализации PyTorch с 2019 года: присутствует
    во ВСЕХ файлах ``.pt``/``.pth``/``.bin``. Флагировать такие ссылки как
    подозрительные — гарантированный ложный срабат на любой реальной
    PyTorch-модели (ResNet18: 100+ ложных issue).

    Кастомные ``persistent_load`` в кастомном Unpickler могут быть RCE-вектором,
    но такие pid имеют другой формат (строка вроде ``'os.system'`` в bypass_09
    или произвольные структуры). Их продолжаем ловить как MLS-PKL-003.
    """
    if not isinstance(pid, tuple) or len(pid) < 2:
        return False
    if pid[0] != "storage":
        return False
    storage_type = pid[1]
    if not isinstance(storage_type, tuple) or len(storage_type) != 2:
        return False
    module, name = storage_type
    return (
        isinstance(module, str)
        and isinstance(name, str)
        and module == "torch"
        and name.endswith("Storage")
    )


@ScannerRegistry.register
class PickleScanner(BaseScanner):
    """Сканер pickle-файлов (.pkl, .pickle, .dill, .pt)."""

    name = "pickle"
    description = "Сканер pickle-файлов (.pkl, .pickle, .dill)"
    supported_extensions = [".pkl", ".pickle", ".dill", ".pt"]
    # Magic bytes для pickle-протоколов 2-5 (opcode PROTO 0x80 + номер версии).
    # Оставлено для документации: сам ``can_handle`` их больше не сверяет.
    magic_bytes = [b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"]

    # Первые байты для реальных pickle-потоков — pickletools.opcodes с code
    # значением на позиции 0. Список исчерпывающий (все pickle-опкоды, которые
    # могут быть первыми). Служит быстрым фильтром: сначала отсеиваем случайные
    # бинарники по первому байту, потом валидируем поток через genops.
    _VALID_START_BYTES: ClassVar[frozenset[int]] = frozenset(
        ord(op.code) for op in pickletools.opcodes
    )

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Обрабатывает файлы с pickle-расширениями через гибридную проверку.

        Ранее принимались только magic bytes protocol 2-5 (``\\x80\\x02..05``),
        из-за чего legitimate pickle protocol 0/1 отбрасывались как
        «неподдерживаемый формат». Malware часто использует protocol 0 (text),
        потому что он читаем — как раз то, что нужно ловить сканеру.

        Гибридный критерий:
          1. Расширение — одно из поддерживаемых (``.pkl``/``.pickle``/``.dill``/``.pt``).
          2. Первый байт — один из известных pickle-опкодов (быстрый отсев мусора).
          3. ``pickletools.genops`` парсит хотя бы один опкод без исключения
             (валидация: за первым байтом идёт синтаксически корректный поток).

        Второй шаг фильтрует крайний случай, когда `genops` мог бы «случайно»
        успешно распарсить 1-байтовый файл с STOP-опкодом (``.``): расширение
        совпадает, байт один, поток валиден — но это точно не осмысленный
        pickle. Такой файл малый по вероятности, но при этом сам факт
        нескольких таких отказов удобно логировать без ложных «unsupported».

        Читаем не более 4 КБ файла — этого достаточно genops'у для выдачи
        первого опкода при любом валидном pickle, и это ограничивает
        потребление памяти на файлах-приманках.
        """
        if path.suffix.lower() not in cls.supported_extensions:
            return False
        try:
            with path.open("rb") as fh:
                header = fh.read(4096)
        except OSError:
            return False
        if not header:
            return False
        if header[0] not in cls._VALID_START_BYTES:
            return False
        gen = pickletools.genops(io.BytesIO(header))
        try:
            next(gen)
        except (
            StopIteration,
            ValueError,
            KeyError,
            IndexError,
            EOFError,
            OSError,
            UnicodeDecodeError,
        ):
            return False
        except Exception:  # noqa: BLE001 — genops бросает разные типы для «мусорных» байт
            return False
        return True

    def scan(self, path: Path) -> RawScanData:
        """Читает файл и возвращает RawScanData; при ошибке — с заполненным error.

        Реализован потоково: вместо ``path.read_bytes()`` (что грузит весь
        pickle-файл в RAM, нарушая принцип CLAUDE.md «не читать файл целиком,
        если размер неизвестен») используется ``pickletools.genops`` поверх
        file handle. Это критично для крупных pickle (несколько ГБ).
        Для ZIP-файлов делегируется в _scan_zip, который читает архив через
        zipfile (тоже потоково).
        """
        try:
            self._check_file_size(path)
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

        # Читаем только первые 4 байта для определения ZIP-magic — потоковая
        # альтернатива data[:4] из старой реализации.
        try:
            with path.open("rb") as fh:
                head = fh.read(4)
        except OSError as exc:
            hashes: dict[str, str] = {}
            with contextlib.suppress(OSError):
                hashes = self._compute_hashes(path)
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=0,
                scanner_name=self.name,
                error=str(exc),
            )

        if head[:4] == b"PK\x03\x04":
            return self._scan_zip(path)
        return self._scan_pickle_stream(path)

    def _scan_pickle_stream(self, path: Path) -> RawScanData:
        """Сканирует pickle-файл потоково, не загружая его в RAM целиком.

        Альтернатива scan_bytes для случая «у нас есть путь к файлу, файл
        может быть большим». pickletools.genops принимает file-like и
        итеративно читает opcode-ы; накопленный список opcodes/globals/strings
        полностью аналогичен результату scan_bytes.

        Для поиска PE/ELF/Mach-O используется find_signatures_in_file —
        тоже потоково (чанки по 1 МБ с оверлапом).
        """
        try:
            hashes = self._compute_hashes(path)
            file_size = path.stat().st_size
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash={},
                file_size=0,
                scanner_name=self.name,
                error=str(exc),
            )

        # Sample первых 4 КБ — нужен детекторам как fallback; читаем отдельно.
        try:
            with path.open("rb") as fh:
                raw_sample = fh.read(4096)
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=str(exc),
            )

        protocol = _detect_protocol(raw_sample)

        # Парсим pickle потоково через file handle.
        opcodes: list[OpcodeInfo] = []
        globals_set: set[tuple[str, str]] = set()
        strings: list[StringInfo] = []
        reduce_calls: list[ReduceCall] = []
        error: str | None = None
        walk_meta: dict[str, str] = {}

        try:
            with path.open("rb") as fh:
                walk_meta = _walk_opcodes(fh, opcodes, globals_set, strings, reduce_calls)
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            # pickletools может бросить ValueError, struct.error, EOFError…
            logger.warning(
                "Ошибка парсинга pickle %s: %s. Частичные результаты сохранены.",
                path,
                exc,
            )
            error = str(exc)
            # Homoglyph-bypass: UnicodeDecodeError при декодировании GLOBAL/INST
            # означает не-ASCII байты в имени модуля — признак homoglyph-атаки.
            if "ascii" in error.lower() and ("codec" in error.lower() or "decode" in error.lower()):
                homoglyphs = _find_homoglyph_globals(raw_sample)
                if homoglyphs:
                    hg_meta: dict[str, Any] = {}
                    if protocol is not None:
                        hg_meta["protocol"] = str(protocol)
                    hg_meta["homoglyph_globals"] = homoglyphs
                    # Возвращаем RawScanData сразу с homoglyph-данными
                    embedded = find_signatures_in_file(path) if file_size > 0 else []
                    return RawScanData(
                        file_path=path,
                        file_hash=hashes,
                        file_size=file_size,
                        scanner_name=self.name,
                        opcodes=opcodes if opcodes else None,
                        globals=globals_set if globals_set else None,
                        strings=strings if strings else None,
                        reduce_calls=reduce_calls if reduce_calls else None,
                        embedded_bytes=embedded if embedded else None,
                        raw_content_sample=raw_sample,
                        error=error,
                        metadata=hg_meta,
                    )

        # Поиск PE/ELF потоково — без загрузки файла в RAM.
        embedded = find_signatures_in_file(path) if file_size > 0 else []

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=opcodes if opcodes else None,
            globals=globals_set if globals_set else None,
            strings=strings if strings else None,
            reduce_calls=reduce_calls if reduce_calls else None,
            embedded_bytes=embedded if embedded else None,
            raw_content_sample=raw_sample,
            error=error,
            metadata=_build_metadata(protocol, walk_meta),
        )

    def _scan_zip(self, path: Path) -> RawScanData:
        """Сканирует ZIP-архив (.pt, .npz) — ищет pickle-payload во вложенных файлах."""
        hashes = self._compute_hashes(path)
        file_size = path.stat().st_size

        combined_opcodes: list[OpcodeInfo] = []
        combined_globals: set[tuple[str, str]] = set()
        combined_strings: list[StringInfo] = []
        combined_reduce_calls: list[ReduceCall] = []
        combined_embedded: list[EmbeddedSignature] = []
        errors: list[str] = []

        try:
            for name, entry_data in ContainerExtractor.extract_zip_members(path):
                # PE/ELF/Mach-O ищем во ВСЕХ членах архива, не только в pickle —
                # вредоносный исполняемый payload часто кладут отдельным членом.
                combined_embedded.extend(find_signatures_in_bytes(entry_data))

                if not any(entry_data.startswith(m) for m in self.magic_bytes):
                    continue
                inner = self.scan_bytes(entry_data, path / name)
                if inner.opcodes:
                    combined_opcodes.extend(inner.opcodes)
                if inner.globals:
                    combined_globals.update(inner.globals)
                if inner.strings:
                    combined_strings.extend(inner.strings)
                if inner.reduce_calls:
                    combined_reduce_calls.extend(inner.reduce_calls)
                if inner.embedded_bytes:
                    combined_embedded.extend(inner.embedded_bytes)
                if inner.error:
                    errors.append(f"{name}: {inner.error}")
        except ContainerError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=str(exc),
            )

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=combined_opcodes if combined_opcodes else None,
            globals=combined_globals if combined_globals else None,
            strings=combined_strings if combined_strings else None,
            reduce_calls=combined_reduce_calls if combined_reduce_calls else None,
            embedded_bytes=combined_embedded if combined_embedded else None,
            error="; ".join(errors) if errors else None,
        )

    def scan_bytes(self, data: bytes, source_path: Path) -> RawScanData:
        """Парсит pickle из bytes и возвращает RawScanData.

        Используется PickleScanner напрямую и вызывается из JoblibScanner /
        PyTorchScanner, которые распаковывают внутренние потоки.
        При ошибке парсинга возвращает частично заполненный RawScanData
        с полем error — сканер никогда не бросает исключений наружу.
        """
        hashes = {}
        if source_path.exists():
            hashes = self._compute_hashes(source_path)

        file_size = len(data)
        raw_sample = data[:4096]
        protocol = _detect_protocol(data)

        opcodes: list[OpcodeInfo] = []
        globals_set: set[tuple[str, str]] = set()
        strings: list[StringInfo] = []
        reduce_calls: list[ReduceCall] = []
        error: str | None = None
        walk_meta: dict[str, str] = {}

        try:
            walk_meta = _walk_opcodes(
                io.BytesIO(data),
                opcodes,
                globals_set,
                strings,
                reduce_calls,
            )
        except Exception as exc:  # noqa: BLE001
            # pickletools может бросить ValueError, struct.error, EOFError и др.
            logger.warning(
                "Ошибка парсинга pickle %s: %s. Частичные результаты сохранены.",
                source_path,
                exc,
            )
            error = str(exc)
            # Homoglyph-bypass: UnicodeDecodeError при декодировании GLOBAL/INST
            if "ascii" in error.lower() and ("codec" in error.lower() or "decode" in error.lower()):
                homoglyphs = _find_homoglyph_globals(data)
                if homoglyphs:
                    embedded = find_signatures_in_bytes(data) if data else []
                    hg_meta: dict[str, Any] = {}
                    if protocol is not None:
                        hg_meta["protocol"] = str(protocol)
                    hg_meta["homoglyph_globals"] = homoglyphs
                    return RawScanData(
                        file_path=source_path,
                        file_hash=hashes,
                        file_size=file_size,
                        scanner_name=self.name,
                        opcodes=opcodes if opcodes else None,
                        globals=globals_set if globals_set else None,
                        strings=strings if strings else None,
                        reduce_calls=reduce_calls if reduce_calls else None,
                        embedded_bytes=embedded if embedded else None,
                        raw_content_sample=raw_sample,
                        error=error,
                        metadata=hg_meta,
                    )

        # Поиск встроенных PE/ELF/Mach-O сигнатур по всему pickle-потоку.
        # Раньше ExecutableDetector смотрел только на первые 4 КБ — теперь
        # сканер сам передаёт детектору полный список сигнатур.
        embedded = find_signatures_in_bytes(data) if data else []

        return RawScanData(
            file_path=source_path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=opcodes if opcodes else None,
            globals=globals_set if globals_set else None,
            strings=strings if strings else None,
            reduce_calls=reduce_calls if reduce_calls else None,
            embedded_bytes=embedded if embedded else None,
            raw_content_sample=raw_sample,
            error=error,
            metadata=_build_metadata(protocol, walk_meta),
        )


#: Ключи metadata с фактами getattr (для сужения MLS-PKL-001, калибровка правки 2).
#: Контейнерные сканеры (pytorch/joblib/numpy) делегируют разбор pickle сюда, но
#: собирают свой metadata — эти факты нужно ПРОБРОСИТЬ наверх, иначе детектор их
#: не увидит и getattr останется HIGH даже на легит-реконструкции в .pt/.joblib.
GETATTR_META_KEYS: tuple[str, ...] = (
    "getattr_present", "getattr_literal_attrs", "getattr_dynamic",
)


def merge_getattr_meta(acc: dict[str, str], inner_meta: dict[str, Any] | None) -> None:
    """Сливает getattr-факты из внутреннего pickle-скана в аккумулятор ``acc``.

    Объединяет литеральные имена атрибутов (union), логическое ИЛИ для флагов
    present/dynamic. Вызывается контейнерными сканерами по каждому вложенному
    pickle. Мутирует ``acc`` на месте.
    """
    if not inner_meta:
        return
    if inner_meta.get("getattr_present") == "true":
        acc["getattr_present"] = "true"
    if inner_meta.get("getattr_dynamic") == "true":
        acc["getattr_dynamic"] = "true"
    inner_attrs = inner_meta.get("getattr_literal_attrs")
    if inner_attrs:
        merged = {a for a in acc.get("getattr_literal_attrs", "").split(",") if a}
        merged.update(a for a in str(inner_attrs).split(",") if a)
        acc["getattr_literal_attrs"] = ",".join(sorted(merged))


def _build_metadata(protocol: int | None, extra: dict[str, str]) -> dict[str, Any] | None:
    """Собирает metadata из версии протокола и доп-флагов ``_walk_opcodes``.

    ``extra`` содержит флаги усечения (``opcode_limit_exceeded`` и т. п.), которые
    ``_walk_opcodes`` возвращает при срабатывании OOM-лимитов. Если оба источника
    пусты — возвращает ``None`` (RawScanData.metadata остаётся незаполненным).
    """
    meta: dict[str, Any] = {}
    if protocol is not None:
        meta["protocol"] = str(protocol)
    meta.update(extra)
    return meta if meta else None


def _find_homoglyph_globals(data: bytes) -> list[tuple[str, str]]:
    """Ищет GLOBAL/INST опкоды с не-ASCII байтами в аргументах (homoglyph-атаки).

    Вызывается когда pickletools.genops завершается с UnicodeDecodeError.
    Возвращает список (module_repr, name_repr) найденных подозрительных записей.
    """
    results: list[tuple[str, str]] = []
    i = 0
    while i < len(data):
        # GLOBAL = b'c' (0x63), INST = b'i' (0x69)
        if data[i] in (0x63, 0x69):
            rest = data[i + 1:]
            nl1 = rest.find(b"\n")
            if nl1 < 0:
                i += 1
                continue
            module_bytes = rest[:nl1]
            rest2 = rest[nl1 + 1:]
            nl2 = rest2.find(b"\n")
            if nl2 < 0:
                i += 1
                continue
            name_bytes = rest2[:nl2]
            has_non_ascii = False
            try:
                module_bytes.decode("ascii")
            except UnicodeDecodeError:
                has_non_ascii = True
            try:
                name_bytes.decode("ascii")
            except UnicodeDecodeError:
                has_non_ascii = True
            if has_non_ascii:
                results.append((
                    module_bytes.decode("utf-8", errors="replace"),
                    name_bytes.decode("utf-8", errors="replace"),
                ))
                i += 1 + nl1 + 1 + nl2 + 1
                continue
        i += 1
    return results


def _detect_protocol(data: bytes) -> int | None:
    """Определяет версию протокола pickle.

    Протоколы 2–5: первые два байта — ``\\x80\\xNN`` (PROTO opcode + версия).
    Протоколы 0/1: PROTO opcode отсутствует, файл начинается прямо с opcode.
    Различить proto 0 vs 1 без полного парсинга нельзя, поэтому при отсутствии
    PROTO возвращаем 0 как наиболее консервативное значение.

    Эвристика для proto 0/1: первый байт — один из характерных opcodes
    (см. ``_PROTO01_LEAD_OPCODES``), И где-то в данных есть STOP-байт ``b'.'``.
    Иначе возвращаем None (это не pickle, а проверка вызывающим стороной).
    """
    if len(data) >= 2 and data[0] == 0x80:
        return data[1]
    if data and data[0] in _PROTO01_LEAD_OPCODES and b"." in data:
        return 0
    return None


# Характерные стартовые opcodes pickle proto 0/1 (без \x80 PROTO).
# Используется в _detect_protocol для эвристической детекции legacy pickle.
# Симметрично с numpy_scanner._PICKLE_PROTO01_LEAD (аудит #13).
_PROTO01_LEAD_OPCODES: frozenset[int] = frozenset(b"(}])ldticIL SVU")


def _split_global_arg(arg: str) -> tuple[str, str]:
    """Разбивает аргумент GLOBAL/INST вида 'module name' на (module, name)."""
    parts = arg.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return arg, ""


def _walk_opcodes(
    fh: Any,
    opcodes: list[OpcodeInfo],
    globals_set: set[tuple[str, str]],
    strings: list[StringInfo],
    reduce_calls: list[ReduceCall],
) -> dict[str, str]:
    """Итерируется по opcode-ам pickle-потока и заполняет переданные коллекции.

    Принимает любой file-like (file handle / BytesIO), что позволяет работать
    как потоково с диска, так и с in-memory данными. Это общий цикл для
    PickleScanner.scan() и PickleScanner.scan_bytes() — раньше дублировался.

    Накопление ограничено (:data:`MAX_OPCODES` / :data:`MAX_STRINGS` /
    :data:`MAX_REDUCE_CALLS`) для защиты от OOM на аномально длинных потоках.
    При достижении лимита opcodes цикл аккуратно прерывается — иначе memo/stack
    росли бы без предела. ``globals_set`` НЕ ограничивается: все опасные глобалы,
    собранные ДО лимита, остаются доступны детекторам. Потоковая природа genops
    сохранена — файл целиком в RAM не грузится.

    Возвращает словарь доп-метаданных: если анализ был усечён, содержит
    ``{"opcode_limit_exceeded": "true", "opcode_limit": str(MAX_OPCODES)}``,
    иначе пустой словарь. Предупреждение MLS-PKL-006 по этому флагу эмитирует
    blocklist_detector, а не сканер (Scanner знает формат, Detector — угрозу).

    Может бросить любые исключения, которые pickletools.genops пробрасывает
    наружу (struct.error, ValueError, EOFError) — вызывающий должен их ловить.
    """
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    last_callable: tuple[str, str] | None = None
    limit_exceeded = False
    # Факты о getattr для сужения MLS-PKL-001 (калибровка правки 2): при вызове
    # getattr(obj, "attr") записываем ЛИТЕРАЛЬНОЕ имя атрибута; при не-литеральном
    # 2-м аргументе помечаем динамику. Blocklist-детектор по этим фактам решает,
    # опасно ли (getattr(os,"system")) или это легит-реконструкция (getattr(m,"Cls")).
    getattr_literal_attrs: set[str] = set()
    getattr_dynamic = False
    getattr_reduced = 0

    def _add_string(info: StringInfo) -> None:
        """Добавляет строку с учётом лимита MAX_STRINGS (защита от OOM)."""
        nonlocal limit_exceeded
        if len(strings) < MAX_STRINGS:
            strings.append(info)
        else:
            limit_exceeded = True

    def _add_reduce(call: ReduceCall) -> None:
        """Добавляет reduce-вызов с учётом лимита MAX_REDUCE_CALLS."""
        nonlocal limit_exceeded
        if len(reduce_calls) < MAX_REDUCE_CALLS:
            reduce_calls.append(call)
        else:
            limit_exceeded = True

    for opcode_obj, arg, pos in pickletools.genops(fh):
        if len(opcodes) >= MAX_OPCODES:
            # Кап на opcodes достигнут — прерываем разбор во избежание OOM.
            # Продолжение цикла копило бы memo/stack без предела (та же DoS-
            # поверхность). globals_set, собранный ДО лимита, полностью сохранён
            # и остаётся доступен детекторам.
            limit_exceeded = True
            break
        opname: str = opcode_obj.name
        ipos: int = pos if pos is not None else 0
        opcodes.append(OpcodeInfo(position=ipos, opcode=opname, arg=arg))

        if opname in _STRING_OPCODES and isinstance(arg, str):
            stack.append(arg)
            if len(arg) > _MIN_STRING_LEN:
                _add_string(StringInfo(value=arg, position=ipos))
        elif opname == "GLOBAL" and isinstance(arg, str):
            module, name = _split_global_arg(arg)
            globals_set.add((module, name))
            last_callable = (module, name)
            stack.append((module, name))
        elif opname == "STACK_GLOBAL":
            name_val = stack.pop() if stack else None
            module_val = stack.pop() if stack else None
            if isinstance(module_val, str) and isinstance(name_val, str):
                globals_set.add((module_val, name_val))
                last_callable = (module_val, name_val)
                stack.append((module_val, name_val))
            else:
                stack.append(None)
        elif opname == "REDUCE":
            _args = stack.pop() if stack else None
            callable_ref = stack.pop() if stack else None
            if isinstance(callable_ref, tuple) and len(callable_ref) == 2:
                mod, nm = callable_ref
                if nm == "getattr" and mod in ("builtins", "__builtin__", "__builtins__"):
                    # 2-й аргумент getattr(obj, "attr") — имя атрибута. Литерал →
                    # запоминаем; не-str (вычисляемое/не восстановлено) → динамика.
                    getattr_reduced += 1
                    attr = _args[1] if isinstance(_args, tuple) and len(_args) >= 2 else None
                    if isinstance(attr, str):
                        getattr_literal_attrs.add(attr)
                    else:
                        getattr_dynamic = True
                _add_reduce(ReduceCall(module=mod, name=nm, position=ipos))
            elif last_callable is not None:
                _add_reduce(
                    ReduceCall(
                        module=last_callable[0],
                        name=last_callable[1],
                        position=ipos,
                    )
                )
            stack.append(None)
        elif opname == "INST" and isinstance(arg, str):
            module, name = _split_global_arg(arg)
            globals_set.add((module, name))
            _add_reduce(ReduceCall(module=module, name=name, position=ipos))
            stack = [item for item in stack if item is not _MARK_SENTINEL]
            stack.append(None)
        elif opname in ("NEWOBJ", "NEWOBJ_EX"):
            _args = stack.pop() if stack else None
            if opname == "NEWOBJ_EX":
                _kwargs = stack.pop() if stack else None
            callable_ref = stack.pop() if stack else None
            if isinstance(callable_ref, tuple) and len(callable_ref) == 2:
                mod, nm = callable_ref
                _add_reduce(ReduceCall(module=mod, name=nm, position=ipos))
            stack.append(None)
        elif opname == "MARK":
            stack.append(_MARK_SENTINEL)
        elif opname in ("TUPLE", "LIST", "DICT"):
            items: list[Any] = []
            while stack and stack[-1] is not _MARK_SENTINEL:
                items.insert(0, stack.pop())
            if stack and stack[-1] is _MARK_SENTINEL:
                stack.pop()
            stack.append(tuple(items) if opname == "TUPLE" else None)
        elif opname == "TUPLE1":
            val = stack.pop() if stack else None
            stack.append((val,))
        elif opname == "TUPLE2":
            bval = stack.pop() if stack else None
            aval = stack.pop() if stack else None
            stack.append((aval, bval))
        elif opname == "TUPLE3":
            cval = stack.pop() if stack else None
            bval = stack.pop() if stack else None
            aval = stack.pop() if stack else None
            stack.append((aval, bval, cval))
        elif opname in ("EMPTY_TUPLE", "EMPTY_LIST", "EMPTY_DICT", "EMPTY_SET"):
            stack.append(None)
        elif opname == "POP_MARK":
            while stack and stack[-1] is not _MARK_SENTINEL:
                stack.pop()
            if stack:
                stack.pop()
        elif opname == "POP":
            if stack:
                stack.pop()
        elif opname in ("PUT", "BINPUT", "LONG_BINPUT"):
            # PUT arg — строка "0", BINPUT/LONG_BINPUT arg — int.
            key = int(arg) if isinstance(arg, str) else arg
            if isinstance(key, int) and stack:
                memo[key] = stack[-1]
        elif opname == "MEMOIZE":
            # MEMOIZE неявно использует len(memo) как следующий индекс.
            if stack:
                memo[len(memo)] = stack[-1]
        elif opname in ("GET", "BINGET", "LONG_BINGET"):
            # GET arg — строка "0", BINGET/LONG_BINGET arg — int.
            key = int(arg) if isinstance(arg, str) else arg
            stack.append(memo.get(key) if isinstance(key, int) else None)
        elif opname == "PERSID" and isinstance(arg, str):
            # PERSID 'id' — персистентная ссылка с текстовым id.
            # В ML-файлах PERSID быть не должен — флагируем как подозрительный.
            if len(arg) > _MIN_STRING_LEN:
                _add_string(StringInfo(value=arg, position=ipos))
            globals_set.add(("__persid__", arg))
            stack.append(None)
        elif opname == "BINPERSID":
            # BINPERSID — id берётся с вершины стека.
            pid = stack.pop() if stack else None
            # Легитимный PyTorch tensor storage persistent_id — не флагируем:
            # такой формат ('storage', ('torch', '<Xxx>Storage'), key, device, numel)
            # присутствует в каждом .pt/.pth/.bin. См. _is_legit_torch_storage_persid.
            if not _is_legit_torch_storage_persid(pid):
                pid_str = str(pid) if pid is not None else "__binpersid__"
                if isinstance(pid, str) and len(pid) > _MIN_STRING_LEN:
                    _add_string(StringInfo(value=pid, position=ipos))
                globals_set.add(("__persid__", pid_str))
            stack.append(None)
        elif opname in _STACK_PUSH_OPCODES:
            stack.append(arg)
        # остальные opcodes (_NOOP_OPCODES и неизвестные) не требуют действий

    meta: dict[str, str] = {}
    if limit_exceeded:
        # Анализ усечён: сигнализируем детектору (MLS-PKL-006). Сам сканер угрозу
        # не эмитит — Scanner знает формат, Detector знает угрозу.
        meta["opcode_limit_exceeded"] = "true"
        meta["opcode_limit"] = str(MAX_OPCODES)
    # Факты getattr для сужения MLS-PKL-001. getattr есть в globals, но REDUCE с
    # литеральным 2-м аргументом не встречен (pushed-not-called / вычисляемое имя)
    # → считаем динамикой (консервативно, HIGH сохраняется).
    getattr_in_globals = any(
        nm == "getattr" and mod in ("builtins", "__builtin__", "__builtins__")
        for mod, nm in globals_set
    )
    if getattr_in_globals:
        meta["getattr_present"] = "true"
        if getattr_literal_attrs:
            meta["getattr_literal_attrs"] = ",".join(sorted(getattr_literal_attrs))
        if getattr_dynamic or getattr_reduced == 0:
            meta["getattr_dynamic"] = "true"
    return meta

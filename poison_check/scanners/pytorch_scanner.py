"""Сканер PyTorch-файлов: .pt, .pth, .bin, .ckpt.

PyTorch-файлы нового формата — это ZIP-архивы со следующей структурой:
  archive/
    data.pkl       ← главный pickle с весами (может быть вложен глубже)
    data/          ← директория с бинарными тензорами (числа)
    record.json    ← опциональные метаданные

.bin-файлы HuggingFace тоже используют этот формат.
Старый (legacy) формат — plain pickle-поток без ZIP.

Важно: известная категория ошибки разбора — пропуск содержимого файла,
для которого _is_zipfile() возвращает True. Мы всегда распаковываем ZIP
и сканируем data.pkl — именно там находится payload.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import ClassVar

from poison_check.core.container import ContainerError, ContainerExtractor
from poison_check.core.executable_signatures import find_signatures_in_bytes
from poison_check.core.registry import ScannerRegistry
from poison_check.core.result import EmbeddedSignature, OpcodeInfo, ReduceCall, StringInfo
from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

# Возможные имена главного pickle-файла внутри архива (в порядке приоритета)
_PYTORCH_PICKLE_CANDIDATES: tuple[str, ...] = (
    "archive/data.pkl",
    "data.pkl",
)

# Файл с метаданными сериализации (PyTorch >= 2.x)
_PYTORCH_SERIALIZATION_ID = "archive/.data/serialization_id"

# Файл с record-метаданными (старые форматы)
_PYTORCH_RECORD_JSON = "archive/record.json"

# Подозрительные ключи в record.json / metadata
_SUSPICIOUS_RECORD_KEYS: frozenset[str] = frozenset(
    {"__reduce__", "__reduce_ex__", "exec", "eval", "compile"}
)

# Члены ZIP-архива PyTorch с сырыми тензорными данными (archive/data/N).
# Содержат float32/float16/int8 числа — сотни МБ случайных байтов, которые
# дают тысячи ложных совпадений magic bytes PE/ELF. Исполняемый payload
# технически мог бы быть спрятан и здесь, но после структурной валидации
# PE/ELF в find_signatures_in_bytes это уже исключено. Пропускаем их для
# производительности (не загружаем сотни МБ тензоров в память).
_TENSOR_MEMBER_RE: re.Pattern[str] = re.compile(r"^(?:archive/)?data/\d+$")

# Magic bytes ZIP
_ZIP_MAGIC: bytes = b"\x50\x4b\x03\x04"

# Magic bytes pickle (протоколы 2-5)
_PICKLE_MAGICS: tuple[bytes, ...] = (
    b"\x80\x02",
    b"\x80\x03",
    b"\x80\x04",
    b"\x80\x05",
)


@ScannerRegistry.register
class PyTorchScanner(BaseScanner):
    """Сканер PyTorch-файлов (.pt, .pth, .bin, .ckpt).

    Открывает файл как ZIP, находит внутри data.pkl и сканирует его
    через PickleScanner.scan_bytes(). Поддерживает:
    - Современный ZIP-формат (новый torch.save)
    - Legacy plain-pickle формат (старый torch.save без ZIP)
    - HuggingFace .bin-файлы (тот же ZIP-формат)

    Принципиально не пропускает data.pkl, даже если файл определён
    как ZIP (известная категория ошибки — пропуск ZIP-содержимого).
    """

    name = "pytorch"
    description = "Сканер PyTorch-файлов (.pt, .pth, .bin, .ckpt)"
    supported_extensions: ClassVar[list[str]] = [".pt", ".pth", ".bin", ".ckpt"]
    magic_bytes: ClassVar[list[bytes]] = [_ZIP_MAGIC]

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Проверяет расширение и magic bytes (ZIP или pickle).

        Принимает файл, если:
        - расширение в supported_extensions И
        - файл начинается с ZIP magic bytes ИЛИ с pickle magic bytes
          (последнее — legacy формат).
        """
        if path.suffix.lower() not in cls.supported_extensions:
            return False
        try:
            with path.open("rb") as fh:
                header = fh.read(6)
        except OSError:
            return False
        if header[:4] == _ZIP_MAGIC:
            return True
        return any(header[:2] == m for m in _PICKLE_MAGICS)

    def scan(self, path: Path) -> RawScanData:
        """Сканирует PyTorch-файл и возвращает RawScanData.

        Определяет формат (ZIP или legacy pickle) по magic bytes и
        делегирует соответствующему обработчику. При любой ошибке
        возвращает RawScanData с заполненным полем error — никогда
        не бросает исключений наружу.
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
                header = fh.read(6)
        except OSError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=str(exc),
            )

        if header[:4] == _ZIP_MAGIC:
            return self._scan_zip(path, hashes, file_size)

        # Legacy plain-pickle (старый формат torch.save без ZIP)
        if any(header[:2] == m for m in _PICKLE_MAGICS):
            return self._scan_legacy_pickle(path, hashes, file_size)

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            error=(
                f"Неизвестный формат PyTorch-файла: "
                f"первые байты {header[:4]!r}. "
                "Ожидался ZIP-архив или pickle-поток."
            ),
        )

    # ------------------------------------------------------------------
    # Внутренние обработчики форматов
    # ------------------------------------------------------------------

    def _scan_zip(
        self,
        path: Path,
        hashes: dict[str, str],
        file_size: int,
    ) -> RawScanData:
        """Сканирует ZIP-архив PyTorch.

        Алгоритм:
        1. Собирает список всех членов архива.
        2. Ищет главный pickle по _PYTORCH_PICKLE_CANDIDATES.
        3. Сканирует все .pkl-файлы через PickleScanner.scan_bytes().
        4. Проверяет record.json / serialization_id на подозрительные ключи.
        5. Объединяет результаты в RawScanData с nested_files.
        """
        # Ленивый импорт для соблюдения порядка зависимостей
        from poison_check.scanners.pickle_scanner import PickleScanner, merge_getattr_meta

        pickle_scanner = PickleScanner()

        # Читаем все члены архива в память
        try:
            members: dict[str, bytes] = {}
            for name, data in ContainerExtractor.extract_zip_members(path):
                members[name] = data
        except ContainerError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=str(exc),
            )

        # Если архив пустой — предупреждаем
        if not members:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error="ZIP-архив пустой: нет файлов-членов",
                metadata={"pytorch_format": "zip", "members_count": "0"},
            )

        # Ищем главный pickle-файл
        main_pkl_name: str | None = None
        for candidate in _PYTORCH_PICKLE_CANDIDATES:
            if candidate in members:
                main_pkl_name = candidate
                break

        # Собираем все .pkl-файлы в архиве для полного покрытия
        pkl_names: list[str] = [
            n for n in members if n.endswith(".pkl")
        ]

        if not pkl_names and main_pkl_name is None:
            # Нет ни одного pickle — возвращаем предупреждение, не исключение
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=(
                    "ZIP-архив не содержит pickle-файлов (data.pkl не найден). "
                    f"Найденные члены: {sorted(members)!r}"
                ),
                metadata={
                    "pytorch_format": "zip",
                    "members_count": str(len(members)),
                    "members": ", ".join(sorted(members)),
                },
            )

        # Если в архиве вообще нет .pkl, но main_pkl_name найден —
        # добавляем его явно в список (не должно происходить, но защита)
        if main_pkl_name and main_pkl_name not in pkl_names:
            pkl_names.insert(0, main_pkl_name)

        # Сканируем все .pkl-файлы
        nested: list[RawScanData] = []
        combined_opcodes: list[OpcodeInfo] = []
        combined_globals: set[tuple[str, str]] = set()
        combined_strings: list[StringInfo] = []
        combined_reduce_calls: list[ReduceCall] = []
        combined_embedded: list[EmbeddedSignature] = []
        # Факты getattr из вложенных pickle — пробрасываем наверх (сужение
        # MLS-PKL-001): без этого getattr(m,"Cls") в .pt остаётся ложным HIGH.
        getattr_meta: dict[str, str] = {}
        errors: list[str] = []

        # PE/ELF/Mach-O может лежать в любом члене архива, а не только в .pkl.
        # Пропускаем сырые тензорные данные (archive/data/N) — это float-числа,
        # сотни МБ бинарных данных, которые будут давать FP на magic bytes.
        # Структурная валидация PE/ELF в find_signatures_in_bytes уже
        # фильтрует случайные совпадения, но загружать тензоры в RAM
        # ради этого не нужно — полезный payload будет в metadata, не в весах.
        for member_name, member_data in members.items():
            if member_name in pkl_names:
                continue  # pickle-члены просканируются ниже через PickleScanner
            if _TENSOR_MEMBER_RE.match(member_name):
                continue  # сырые тензорные данные — пропускаем
            combined_embedded.extend(find_signatures_in_bytes(member_data))

        for pkl_name in pkl_names:
            pkl_data = members[pkl_name]
            inner = pickle_scanner.scan_bytes(pkl_data, source_path=path / pkl_name)
            nested.append(inner)
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
            merge_getattr_meta(getattr_meta, inner.metadata)
            if inner.error:
                errors.append(f"{pkl_name}: {inner.error}")

        # Проверяем record.json и serialization_id на подозрительные метаданные
        metadata_issues = self._check_metadata(members)
        # Суммарный распакованный размер всех членов — нужен CompressionDetector
        # для проверки zip-bomb (аудит #15).
        total_uncompressed = sum(len(b) for b in members.values())
        metadata: dict[str, str] = {
            "pytorch_format": "zip",
            "members_count": str(len(members)),
            "main_pkl": main_pkl_name or "(не найден)",
            "total_uncompressed_bytes": str(total_uncompressed),
        }
        if metadata_issues:
            metadata["suspicious_metadata"] = "; ".join(metadata_issues)
            errors.extend(metadata_issues)
        metadata.update(getattr_meta)  # проброс getattr-фактов вложенного pickle

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
            metadata=metadata,
            nested_files=nested if nested else None,
            error="; ".join(errors) if errors else None,
        )

    def _scan_legacy_pickle(
        self,
        path: Path,
        hashes: dict[str, str],
        file_size: int,
    ) -> RawScanData:
        """Сканирует legacy plain-pickle PyTorch-файл (без ZIP).

        Старые версии torch.save() сохраняли pickle напрямую без ZIP-обёртки.
        Использует потоковый метод PickleScanner — файл не загружается в RAM
        целиком (критично для моделей 400-500 МБ вида BERT, GPT-2).
        """
        from poison_check.scanners.pickle_scanner import PickleScanner

        pickle_scanner = PickleScanner()
        # _scan_pickle_stream: потоковый парсинг opcodes + потоковый поиск
        # PE/ELF через find_signatures_in_file (чанки 1 МБ, без загрузки в RAM).
        inner = pickle_scanner._scan_pickle_stream(path)

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            opcodes=inner.opcodes,
            globals=inner.globals,
            strings=inner.strings,
            reduce_calls=inner.reduce_calls,
            embedded_bytes=inner.embedded_bytes,
            metadata={
                "pytorch_format": "legacy_pickle",
                **(inner.metadata or {}),
            },
            nested_files=[inner],
            raw_content_sample=inner.raw_content_sample,
            error=inner.error,
        )

    # ------------------------------------------------------------------
    # Проверка метаданных архива
    # ------------------------------------------------------------------

    @staticmethod
    def _check_metadata(members: dict[str, bytes]) -> list[str]:
        """Проверяет record.json и serialization_id на подозрительные ключи.

        Возвращает список строк с описаниями найденных аномалий.
        При ошибке разбора JSON — молча игнорирует (не наша задача валидировать формат).
        """
        issues: list[str] = []

        # Проверяем record.json
        if _PYTORCH_RECORD_JSON in members:
            try:
                record = json.loads(members[_PYTORCH_RECORD_JSON].decode("utf-8", errors="replace"))
                if isinstance(record, dict):
                    found = _SUSPICIOUS_RECORD_KEYS & set(record.keys())
                    if found:
                        issues.append(
                            f"Подозрительные ключи в record.json: {sorted(found)!r}"
                        )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Повреждённый/нестандартный JSON — не падаем

        # Проверяем serialization_id
        if _PYTORCH_SERIALIZATION_ID in members:
            try:
                sid = members[_PYTORCH_SERIALIZATION_ID].decode("utf-8", errors="replace").strip()
                # serialization_id обычно выглядит как UUID или короткая строка
                # Подозрительно, если он содержит код-like конструкции
                for suspicious in ("import", "exec(", "eval(", "__import__"):
                    if suspicious in sid:
                        issues.append(
                            f"Подозрительное содержимое serialization_id: {sid!r}"
                        )
                        break
            except UnicodeDecodeError:
                pass

        return issues

    # ------------------------------------------------------------------
    # Публичный вспомогательный метод для тестов и forensics API
    # ------------------------------------------------------------------

    @staticmethod
    def scan_bytes(pkl_bytes: bytes, source_path: Path) -> RawScanData:
        """Сканирует pickle-байты из PyTorch-архива через PickleScanner.

        Вспомогательный метод для forensics-режима: принимает уже
        распакованный pickle-поток и возвращает RawScanData.
        """
        from poison_check.scanners.pickle_scanner import PickleScanner

        return PickleScanner().scan_bytes(pkl_bytes, source_path=source_path)


# ------------------------------------------------------------------
# Публичная утилита для генерации тестовых .pt-файлов
# ------------------------------------------------------------------

def make_pytorch_zip(pickle_bytes: bytes) -> bytes:
    """Создаёт .pt-файл как ZIP без импорта torch.

    Используется в тестах и generate_fixtures.py для создания
    валидных PyTorch-архивов с произвольным pickle-payload.

    Args:
        pickle_bytes: Произвольный pickle-поток (безопасный или вредоносный).

    Returns:
        bytes: Валидный ZIP-архив с archive/data.pkl внутри.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", pickle_bytes)
    return buf.getvalue()

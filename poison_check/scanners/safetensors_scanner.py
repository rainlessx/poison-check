"""Сканер файлов в формате SafeTensors (.safetensors).

Формат SafeTensors (разработан Hugging Face) специально спроектирован
как безопасная альтернатива pickle — в нём отсутствует возможность
выполнения произвольного кода. Тем не менее файл требует проверки:

  - Заголовок не должен быть аномально большим (decompression attack).
  - Поле __metadata__ не должно содержать секреты (API-ключи, токены).
  - Типы данных тензоров должны соответствовать ожидаемым.
  - Структура файла должна быть корректной (malformed JSON).

Бинарная структура:
  [8 байт little-endian uint64 = длина JSON-заголовка]
  [JSON-заголовок — UTF-8]
  [бинарные данные тензоров]

Ссылка: https://github.com/huggingface/safetensors
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import ClassVar

from poison_check.core.executable_signatures import find_signatures_in_bytes
from poison_check.core.registry import ScannerRegistry
from poison_check.core.result import StringInfo, TensorInfo
from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

# Magic bytes отсутствуют: формат начинается с uint64 (длина заголовка).
# Идентификация — только по расширению.

# Защита от decompression attack: заголовок не может быть > 100 МБ.
_MAX_HEADER_SIZE: int = 100 * 1024 * 1024  # 100 MB

# Размер поля с длиной заголовка.
_HEADER_LEN_FIELD: int = 8

# Dtype-значения, нетипичные для обычных весов моделей.
_UNUSUAL_DTYPES: frozenset[str] = frozenset(
    {"BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64"}
)

# Стандартные "числовые" dtype, используемые в весах моделей.
_EXPECTED_DTYPES: frozenset[str] = frozenset(
    {"F64", "F32", "F16", "BF16", "F8_E5M2", "F8_E4M3"}
)


@ScannerRegistry.register
class SafetensorsScanner(BaseScanner):
    """Сканер файлов формата SafeTensors (.safetensors).

    SafeTensors — безопасный формат без pickle, разработанный Hugging Face.
    Сканер проверяет корректность структуры, отсутствие аномалий заголовка
    и отсутствие секретных данных в метаданных.

    Не выполняет загрузку тензоров — только разбирает JSON-заголовок.
    """

    name = "safetensors"
    description = "Сканер SafeTensors-файлов (.safetensors)"
    supported_extensions: ClassVar[list[str]] = [".safetensors"]
    magic_bytes: ClassVar[list[bytes]] = []  # нет фиксированных magic bytes

    MAX_HEADER_SIZE: ClassVar[int] = _MAX_HEADER_SIZE

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Проверяет, что расширение файла — .safetensors.

        SafeTensors не имеет фиксированных magic bytes, поэтому
        определение формата выполняется только по расширению.
        """
        return path.suffix.lower() in cls.supported_extensions

    def scan(self, path: Path) -> RawScanData:
        """Сканирует SafeTensors-файл и возвращает RawScanData.

        Алгоритм:
        1. Проверяет размер файла через _check_file_size (аудит #2 — защита
           от OOM на огромных файлах). SafeTensors-файлы 30 ГБ — норма для LLM,
           тензоры всё равно не загружаются, проверка нужна для отказа от
           заведомо ненормальных размеров.
        2. Читает первые 8 байт → длина JSON-заголовка (uint64 little-endian).
        3. Проверяет, что длина ≤ MAX_HEADER_SIZE (защита от decompression attack).
        4. Читает JSON-заголовок (только его, тензорные данные не читаются).
        5. Извлекает __metadata__ и строки для SecretsDetector.
        6. Собирает tensor_info из остальных ключей заголовка.
        7. Отмечает необычные dtype.

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
            return self._parse(path, hashes, file_size)
        except Exception as exc:  # noqa: BLE001  # намеренный широкий catch
            # Сканер никогда не падает с необработанным исключением
            logger.debug("Ошибка при разборе %s: %s", path, exc, exc_info=True)
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Ошибка разбора SafeTensors-файла: {exc}",
            )

    # ------------------------------------------------------------------
    # Внутренний разборщик
    # ------------------------------------------------------------------

    def _parse(
        self,
        path: Path,
        hashes: dict[str, str],
        file_size: int,
    ) -> RawScanData:
        """Разбирает SafeTensors-файл и заполняет RawScanData.

        Потоковое чтение: сначала 8-байтовое поле длины, затем только
        JSON-заголовок. Бинарные данные тензоров не читаются в память.
        """
        errors: list[str] = []
        metadata: dict[str, str] = {}
        tensor_info: list[TensorInfo] = []
        strings: list[StringInfo] = []

        with path.open("rb") as fh:
            # --- Шаг 1: читаем поле с длиной заголовка ---
            len_field = fh.read(_HEADER_LEN_FIELD)
            if len(len_field) < _HEADER_LEN_FIELD:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error=(
                        f"Файл слишком короткий: {len(len_field)} байт "
                        f"(ожидалось минимум {_HEADER_LEN_FIELD})"
                    ),
                )

            header_size = struct.unpack("<Q", len_field)[0]

            # --- Шаг 2: проверка размера заголовка ---
            if header_size > self.MAX_HEADER_SIZE:
                # Возвращаем issue-уровня HIGH через errors + metadata,
                # детектор разберётся через RawScanData.metadata["anomaly_header_size"]
                errors.append(
                    f"Аномально большой заголовок: {header_size} байт "
                    f"(лимит {self.MAX_HEADER_SIZE} байт). "
                    "Возможная атака типа decompression bomb."
                )
                metadata["anomaly_header_size"] = str(header_size)
                # Не пытаемся читать потенциально гигантский заголовок
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    metadata=metadata,
                    error="; ".join(errors),
                )

            # --- Шаг 3: читаем JSON-заголовок ---
            header_bytes = fh.read(header_size)
            if len(header_bytes) < header_size:
                return RawScanData(
                    file_path=path,
                    file_hash=hashes,
                    file_size=file_size,
                    scanner_name=self.name,
                    error=(
                        f"Заголовок обрезан: прочитано {len(header_bytes)} байт, "
                        f"ожидалось {header_size}"
                    ),
                )

        # --- Шаг 4: парсим JSON ---
        try:
            header_json: object = json.loads(
                header_bytes.decode("utf-8", errors="replace")
            )
        except json.JSONDecodeError as exc:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=f"Некорректный JSON в заголовке SafeTensors: {exc}",
            )

        if not isinstance(header_json, dict):
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                error=(
                    f"Заголовок SafeTensors должен быть JSON-объектом, "
                    f"получен {type(header_json).__name__}"
                ),
            )

        # --- Шаг 5: извлекаем __metadata__ ---
        raw_metadata: object = header_json.get("__metadata__")
        if isinstance(raw_metadata, dict):
            # Приводим все значения к str (по спецификации — string → string)
            for k, v in raw_metadata.items():
                str_k = str(k)
                str_v = str(v)
                metadata[str_k] = str_v
                # Добавляем значение в strings для SecretsDetector
                strings.append(StringInfo(value=str_v, position=0))

        # --- Шаг 6: собираем tensor_info ---
        unusual_dtypes: list[str] = []
        for key, value in header_json.items():
            if key == "__metadata__":
                continue
            if not isinstance(value, dict):
                continue

            dtype: str = str(value.get("dtype", ""))
            shape_raw: object = value.get("shape", [])
            shape: list[int] = (
                [int(s) for s in shape_raw]
                if isinstance(shape_raw, list)
                else []
            )

            tensor_info.append(TensorInfo(name=key, dtype=dtype, shape=shape))

            if dtype and dtype not in _EXPECTED_DTYPES:
                unusual_dtypes.append(f"{key}: {dtype}")

        if unusual_dtypes:
            metadata["unusual_dtypes"] = "; ".join(unusual_dtypes)

        metadata["tensor_count"] = str(len(tensor_info))
        metadata["header_size_bytes"] = str(header_size)

        # Поиск встроенных PE/ELF/Mach-O в JSON-заголовке. Сами тензорные данные
        # — это сырые float-числа, риск PE/ELF в них минимальный, поэтому
        # ограничиваемся header_bytes. Если кто-то заложил base64-PE
        # в __metadata__-строку, он засветится либо тут, либо в SecretsDetector.
        embedded = find_signatures_in_bytes(header_bytes) if header_bytes else []

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            metadata=metadata if metadata else None,
            tensor_info=tensor_info if tensor_info else None,
            strings=strings if strings else None,
            embedded_bytes=embedded if embedded else None,
            raw_content_sample=header_bytes[:4096],
            error="; ".join(errors) if errors else None,
        )

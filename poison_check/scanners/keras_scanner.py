"""Сканер Keras-моделей: .keras (ZIP) и .h5 / .hdf5 (HDF5).

Форматы:
- ``.keras`` — ZIP-архив нового формата Keras 3. Внутри:
    config.json    ← архитектура модели (слои, в т.ч. Lambda)
    metadata.json  ← версия Keras
    model.weights.h5
  Разбираем config.json через ContainerExtractor, БЕЗ загрузки keras/tensorflow.

- ``.h5`` / ``.hdf5`` — legacy HDF5-формат tf.keras. Архитектура хранится в
  root-атрибуте ``model_config`` (JSON-строка). Читается через h5py, если он
  установлен; сам HDF5-контейнер кода не исполняет. Без h5py сканер деградирует
  до факта ``h5py_available="false"`` (как joblib при отсутствии lz4/zstd) —
  Issue INFO о зависимости эмитит KerasThreatDetector.

Угроза (CVE-2025-1550): слой ``Lambda`` сериализует произвольный Python-код
(байткод функции) и исполняет его при ``keras.models.load_model()`` — даже с
``safe_mode=True`` в уязвимых версиях. Также произвольный код может прийти через
``registered_name`` (пользовательские custom-объекты).

Разделение слоёв: сканер извлекает ФАКТЫ (найденные
Lambda-слои, custom-объекты, тело функций) в metadata/strings; УГРОЗУ (Issue
MLS-KERAS-*) эмитит KerasThreatDetector, а не сканер.

Ссылки:
- https://nvd.nist.gov/vuln/detail/CVE-2025-1550
- https://keras.io/api/models/model_saving_apis/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from poison_check.core.container import ContainerError, ContainerExtractor
from poison_check.core.executable_signatures import find_signatures_in_bytes
from poison_check.core.registry import ScannerRegistry
from poison_check.core.result import StringInfo
from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы формата
# ---------------------------------------------------------------------------

_ZIP_MAGIC: bytes = b"PK\x03\x04"
# Сигнатура HDF5 (8 байт): \x89 H D F \r \n \x1a \n
_HDF5_MAGIC: bytes = b"\x89HDF\r\n\x1a\n"

# Имя файла архитектуры внутри .keras-архива (Keras 3) и его legacy-вариант.
_CONFIG_MEMBERS: tuple[str, ...] = ("config.json", "model.json")
_METADATA_MEMBER: str = "metadata.json"

# Максимальная глубина обхода JSON-дерева (защита от аномально вложенного config).
_MAX_WALK_DEPTH: int = 200

# Максимальная длина одного сохраняемого фрагмента тела функции (в strings).
_MAX_PAYLOAD_LEN: int = 4096

# class_name-маркеры сериализованного Python-callable внутри слоя Lambda.
# Их наличие как registered_name не считаем «пользовательским custom-объектом»
# (это часть штатной сериализации Lambda, уже покрытая lambda_layer_count).
_FUNCTION_CLASS_NAMES: frozenset[str] = frozenset({"function", "__lambda__"})


@dataclass
class _KerasFacts:
    """Факты, извлечённые из config.json / model_config без интерпретации угроз."""

    lambda_layers: list[str] = field(default_factory=list)
    function_payloads: list[str] = field(default_factory=list)
    custom_objects: list[str] = field(default_factory=list)


@ScannerRegistry.register
class KerasScanner(BaseScanner):
    """Сканер Keras-моделей (.keras, .h5, .hdf5).

    Разбирает архитектуру модели без загрузки keras/tensorflow:
    ``.keras`` — как ZIP c config.json, ``.h5`` — через h5py (опционально).
    Находит слои Lambda и пользовательские custom-объекты и складывает факты в
    metadata/strings. Issue про угрозу (RCE) эмитит KerasThreatDetector.

    Никогда не бросает необработанных исключений: повреждённый / неизвестный
    файл → RawScanData с полем error.
    """

    name = "keras"
    description = "Сканер Keras-моделей (.keras, .h5, .hdf5)"
    supported_extensions: ClassVar[list[str]] = [".keras", ".h5", ".hdf5"]
    magic_bytes: ClassVar[list[bytes]] = [_ZIP_MAGIC, _HDF5_MAGIC]

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Проверяет расширение и magic bytes.

        ``.keras`` → ZIP magic; ``.h5`` / ``.hdf5`` → HDF5 magic.
        """
        suffix = path.suffix.lower()
        if suffix not in cls.supported_extensions:
            return False
        try:
            with path.open("rb") as fh:
                header = fh.read(len(_HDF5_MAGIC))
        except OSError:
            return False
        if suffix == ".keras":
            return header[:4] == _ZIP_MAGIC
        return header[: len(_HDF5_MAGIC)] == _HDF5_MAGIC

    def scan(self, path: Path) -> RawScanData:
        """Читает Keras-модель и возвращает RawScanData.

        Определяет формат по magic bytes и делегирует обработчику. При любой
        ошибке возвращает RawScanData с заполненным полем error — никогда не
        бросает исключений наружу.
        """
        try:
            self._check_file_size(path)
            hashes = self._compute_hashes(path)
            file_size = path.stat().st_size
        except ValueError as exc:
            return self._error_raw(path, {}, 0, str(exc))
        except OSError as exc:
            return self._error_raw(path, {}, 0, str(exc))

        try:
            with path.open("rb") as fh:
                header = fh.read(len(_HDF5_MAGIC))
        except OSError as exc:
            return self._error_raw(path, hashes, file_size, str(exc))

        try:
            if header[:4] == _ZIP_MAGIC:
                return self._scan_keras_zip(path, hashes, file_size)
            if header[: len(_HDF5_MAGIC)] == _HDF5_MAGIC:
                return self._scan_h5(path, hashes, file_size)
            return self._error_raw(
                path,
                hashes,
                file_size,
                (
                    f"Неизвестный формат Keras-файла: первые байты {header[:4]!r}. "
                    "Ожидался .keras (ZIP) или .h5/.hdf5 (HDF5)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — сканер не падает на пользовательском файле
            logger.debug("Ошибка при разборе Keras-файла %s: %s", path, exc, exc_info=True)
            return self._error_raw(
                path, hashes, file_size, f"Ошибка разбора Keras-файла: {exc}"
            )

    # ------------------------------------------------------------------
    # .keras (ZIP)
    # ------------------------------------------------------------------

    def _scan_keras_zip(
        self, path: Path, hashes: dict[str, str], file_size: int
    ) -> RawScanData:
        """Разбирает .keras-архив: читает config.json и извлекает факты."""
        members: dict[str, bytes] = {}
        try:
            for name, data in ContainerExtractor.extract_zip_members(path):
                # Нам нужны только манифесты; веса (model.weights.h5) пропускаем.
                base = name.rsplit("/", 1)[-1]
                if base in _CONFIG_MEMBERS or base == _METADATA_MEMBER:
                    members[base] = data
        except ContainerError as exc:
            return self._error_raw(path, hashes, file_size, str(exc))

        config_bytes: bytes | None = None
        for candidate in _CONFIG_MEMBERS:
            if candidate in members:
                config_bytes = members[candidate]
                break

        if config_bytes is None:
            return self._error_raw(
                path,
                hashes,
                file_size,
                "В .keras-архиве не найден config.json (модель без архитектуры?).",
                metadata={"keras_format": "keras_zip"},
            )

        metadata: dict[str, str] = {"keras_format": "keras_zip"}
        version = _extract_keras_version(members.get(_METADATA_MEMBER))
        if version:
            metadata["keras_version"] = version

        try:
            config = json.loads(config_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            return self._error_raw(
                path,
                hashes,
                file_size,
                f"Некорректный JSON в config.json: {exc}",
                metadata=metadata,
            )

        facts = _collect_keras_facts(config)
        _apply_facts_to_metadata(metadata, facts)

        embedded = find_signatures_in_bytes(config_bytes)
        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            strings=_facts_to_strings(facts) or None,
            embedded_bytes=embedded if embedded else None,
            metadata=metadata,
            raw_content_sample=config_bytes[:4096],
        )

    # ------------------------------------------------------------------
    # .h5 / .hdf5 (HDF5)
    # ------------------------------------------------------------------

    def _scan_h5(
        self, path: Path, hashes: dict[str, str], file_size: int
    ) -> RawScanData:
        """Разбирает .h5-модель через h5py (если установлен).

        Без h5py — graceful degradation: фиксируем факт ``h5py_available="false"``,
        Issue INFO о зависимости эмитит KerasThreatDetector (аналогично joblib
        при отсутствии lz4/zstd). Сам HDF5-контейнер кода не исполняет.
        """
        try:
            import h5py  # noqa: PLC0415
        except ImportError:
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                metadata={"keras_format": "h5", "h5py_available": "false"},
                raw_content_sample=self._read_sample(path),
            )

        metadata: dict[str, str] = {"keras_format": "h5", "h5py_available": "true"}
        model_config_raw: str | None = None
        try:
            with h5py.File(path, "r") as f:
                attrs = f.attrs
                version = attrs.get("keras_version")
                if version is not None:
                    metadata["keras_version"] = _attr_to_str(version)
                mc = attrs.get("model_config")
                if mc is not None:
                    model_config_raw = _attr_to_str(mc)
        except (OSError, KeyError, ValueError) as exc:
            return self._error_raw(
                path,
                hashes,
                file_size,
                f"Ошибка чтения HDF5 через h5py: {exc}",
                metadata=metadata,
            )

        if not model_config_raw:
            # Валидный HDF5, но без keras-архитектуры (например, чистые веса).
            metadata["model_config_present"] = "false"
            return RawScanData(
                file_path=path,
                file_hash=hashes,
                file_size=file_size,
                scanner_name=self.name,
                metadata=metadata,
            )

        try:
            config = json.loads(model_config_raw)
        except json.JSONDecodeError as exc:
            return self._error_raw(
                path,
                hashes,
                file_size,
                f"Некорректный JSON в model_config: {exc}",
                metadata=metadata,
            )

        facts = _collect_keras_facts(config)
        _apply_facts_to_metadata(metadata, facts)

        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            strings=_facts_to_strings(facts) or None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Вспомогательное
    # ------------------------------------------------------------------

    def _error_raw(
        self,
        path: Path,
        hashes: dict[str, str],
        file_size: int,
        error: str,
        metadata: dict[str, str] | None = None,
    ) -> RawScanData:
        """Собирает RawScanData с полем error (единая точка для ошибок)."""
        return RawScanData(
            file_path=path,
            file_hash=hashes,
            file_size=file_size,
            scanner_name=self.name,
            metadata=metadata,
            error=error,
        )


# ---------------------------------------------------------------------------
# Функции модульного уровня (тестируются независимо)
# ---------------------------------------------------------------------------


def _collect_keras_facts(config: object) -> _KerasFacts:
    """Рекурсивно обходит config-дерево и собирает факты про Lambda/custom.

    НЕ интерпретирует угрозы (это задача детектора) — только извлекает:
    - имена слоёв с class_name == "Lambda";
    - тело сериализованных функций (для strings / forensics);
    - пользовательские registered_name (custom-объекты).
    """
    facts = _KerasFacts()
    _walk(config, facts, depth=0)
    return facts


def _walk(node: object, facts: _KerasFacts, depth: int) -> None:
    """Обходит dict/list config-дерева, наполняя facts. Ограничен по глубине."""
    if depth > _MAX_WALK_DEPTH:
        return

    if isinstance(node, dict):
        class_name = node.get("class_name")
        config_node = node.get("config")

        if class_name == "Lambda":
            name = "Lambda"
            if isinstance(config_node, dict):
                raw_name = config_node.get("name")
                if isinstance(raw_name, str) and raw_name:
                    name = raw_name
                function_repr = _stringify(config_node.get("function"))
                if function_repr:
                    facts.function_payloads.append(function_repr[:_MAX_PAYLOAD_LEN])
            facts.lambda_layers.append(name)

        if isinstance(class_name, str) and class_name in _FUNCTION_CLASS_NAMES:
            payload = _stringify(config_node if config_node is not None else node)
            if payload:
                facts.function_payloads.append(payload[:_MAX_PAYLOAD_LEN])

        registered = node.get("registered_name")
        if (
            isinstance(registered, str)
            and registered
            and registered not in _FUNCTION_CLASS_NAMES
        ):
            facts.custom_objects.append(registered)

        for value in node.values():
            _walk(value, facts, depth + 1)

    elif isinstance(node, list):
        for item in node:
            _walk(item, facts, depth + 1)


def _stringify(obj: object) -> str:
    """Приводит произвольный JSON-фрагмент к компактной строке (для strings)."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(obj)


def _apply_facts_to_metadata(metadata: dict[str, str], facts: _KerasFacts) -> None:
    """Проставляет счётчики и списки найденных фактов в metadata."""
    metadata["lambda_layer_count"] = str(len(facts.lambda_layers))
    if facts.lambda_layers:
        metadata["lambda_layers"] = ", ".join(facts.lambda_layers)
    metadata["custom_object_count"] = str(len(facts.custom_objects))
    if facts.custom_objects:
        # Уникализируем, сохраняя порядок появления.
        seen: dict[str, None] = dict.fromkeys(facts.custom_objects)
        metadata["custom_objects"] = ", ".join(seen)


def _facts_to_strings(facts: _KerasFacts) -> list[StringInfo]:
    """Тела функций Lambda → strings (для SecretsDetector / NetworkDetector)."""
    return [StringInfo(value=payload, position=0) for payload in facts.function_payloads]


def _extract_keras_version(metadata_bytes: bytes | None) -> str | None:
    """Достаёт keras_version из metadata.json (.keras) — best effort."""
    if not metadata_bytes:
        return None
    try:
        meta = json.loads(metadata_bytes.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    if isinstance(meta, dict):
        version = meta.get("keras_version")
        if isinstance(version, str) and version:
            return version
    return None


def _attr_to_str(value: Any) -> str:
    """Приводит HDF5-атрибут (bytes / numpy-строка / str) к str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# Экспортируем сигнатуры для регистрации / форензики.
__all__ = ["KerasScanner"]

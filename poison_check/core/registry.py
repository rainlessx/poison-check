"""Реестры сканеров и детекторов с регистрацией через декоратор.

Thread-safety (аудит #20): операции мутирования реестров (register / _reset)
защищены ``threading.Lock``. Чтения (find_scanner, all_scanners, get) намеренно
делаются без захвата лока — `dict` на CPython атомарен по отдельным операциям,
а полный snapshot `dict.values()` копируется в list. На бо́льшую гарантию
нет нужды: реестр заполняется один раз при импорте модулей и почти не меняется
во время работы.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, ClassVar

from poison_check.core.detector_base import BaseDetector
from poison_check.core.scanner_base import BaseScanner


class ScannerRegistry:
    """Глобальный реестр сканеров ML-файлов (потокобезопасный для register)."""

    _scanners: ClassVar[dict[str, type[BaseScanner]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, scanner_class: type[BaseScanner]) -> type[BaseScanner]:
        """Декоратор регистрации сканера.

        Потокобезопасно: проверка на дубликат и вставка выполняются под
        одним локом, исключая race при параллельной регистрации.

        Raises:
            ValueError: если сканер с таким именем уже зарегистрирован.
        """
        with cls._lock:
            if scanner_class.name in cls._scanners:
                raise ValueError(
                    f"Сканер '{scanner_class.name}' уже зарегистрирован"
                )
            cls._scanners[scanner_class.name] = scanner_class
        return scanner_class

    @classmethod
    def find_scanner(cls, path: Path) -> type[BaseScanner] | None:
        """Возвращает первый сканер, способный обработать указанный файл."""
        # Снимок значений на момент вызова — защита от изменения dict во время итерации.
        for scanner_class in list(cls._scanners.values()):
            if scanner_class.can_handle(path):
                return scanner_class
        return None

    @classmethod
    def all_scanners(cls) -> list[type[BaseScanner]]:
        """Возвращает список всех зарегистрированных сканеров."""
        return list(cls._scanners.values())

    @classmethod
    def get(cls, name: str) -> type[BaseScanner] | None:
        """Возвращает сканер по имени или None."""
        return cls._scanners.get(name)

    @classmethod
    def _reset(cls) -> None:
        """Сбрасывает реестр. Только для тестов."""
        with cls._lock:
            cls._scanners = {}


class DetectorRegistry:
    """Глобальный реестр детекторов угроз (потокобезопасный для register)."""

    _detectors: ClassVar[dict[str, type[BaseDetector]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, detector_class: type[BaseDetector]) -> type[BaseDetector]:
        """Декоратор регистрации детектора (потокобезопасно).

        Raises:
            ValueError: если детектор с таким именем уже зарегистрирован.
        """
        with cls._lock:
            if detector_class.name in cls._detectors:
                raise ValueError(
                    f"Детектор '{detector_class.name}' уже зарегистрирован"
                )
            cls._detectors[detector_class.name] = detector_class
        return detector_class

    @classmethod
    def all_detectors(cls) -> list[type[BaseDetector]]:
        """Возвращает список всех зарегистрированных детекторов."""
        return list(cls._detectors.values())

    @classmethod
    def get(cls, name: str) -> type[BaseDetector] | None:
        """Возвращает детектор по имени или None."""
        return cls._detectors.get(name)

    @classmethod
    def enabled_for_policy(cls, policy: dict[str, Any]) -> list[type[BaseDetector]]:
        """Возвращает детекторы, активные согласно политике.

        Поддерживает два формата политики:

        * Новый (allowlist): ключ ``enabled_detectors`` — список имён детекторов,
          которые должны работать. ``None`` или отсутствие ключа означает «все».
        * Устаревший (blocklist): ключ ``disabled_detectors`` — список имён
          детекторов, которые нужно отключить. Поддерживается для совместимости.

        Новый формат имеет приоритет над устаревшим.
        """
        enabled_raw = policy.get("enabled_detectors")

        # Новый формат: enabled_detectors задан явно
        if enabled_raw is not None:
            enabled_set: set[str] = set(enabled_raw)
            return [d for d in cls._detectors.values() if d.name in enabled_set]

        # Устаревший формат: disabled_detectors (обратная совместимость)
        disabled: set[str] = set(policy.get("disabled_detectors", []))
        return [d for d in cls._detectors.values() if d.name not in disabled]

    @classmethod
    def _reset(cls) -> None:
        """Сбрасывает реестр. Только для тестов."""
        with cls._lock:
            cls._detectors = {}

"""Базовый класс для детекторов угроз в ML-файлах."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from poison_check.core.result import Issue, MLContext, Severity
from poison_check.core.scanner_base import RawScanData


class BaseDetector(ABC):
    """Абстрактный детектор одного класса угроз в ML-файлах."""

    name: ClassVar[str]
    description: ClassVar[str]
    severity_range: ClassVar[tuple[Severity, Severity]]  # (мин, макс)

    @abstractmethod
    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Анализирует результат сканирования и возвращает список найденных проблем.

        context определяет ML-фреймворк (PyTorch/sklearn/etc)
        для выбора соответствующих правил allowlist.
        """

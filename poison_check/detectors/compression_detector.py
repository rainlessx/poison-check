"""Детектор атак через сжатие — zip-бомбы и decompression bombs.

Анализирует соотношение сжатого размера файла к суммарному размеру
вложенных файлов (nested_files). Подозрительно большое соотношение
или количество вложений указывает на потенциальную zip-бомбу.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Severity,
)
from poison_check.core.scanner_base import RawScanData

logger = logging.getLogger(__name__)

# Пороговое соотношение: распакованный / сжатый размер.
# Soft-порог: > 1000 — подозрительно (HIGH).
# Hard-порог: > 10000 — почти точно zip-bomb.
#
# Аудит #25: sklearn.dump(compress=9) на повторяющихся весах легко даёт
# ratio 200×, lz4 на разреженных tensor'ах — 500×, gzip на text-моделях — 800×.
# Поэтому ratio 1000–10000 это «возможно bomb», а не «точно».
_MAX_COMPRESSION_RATIO: int = 1000
_EXTREME_COMPRESSION_RATIO: int = 10_000

# Подозрительно маленький размер файла при большом распакованном объёме
_SMALL_FILE_THRESHOLD_BYTES: int = 1 * 1024 * 1024  # 1 MB

# Порог суммарного распакованного размера для "маленьких" файлов
_LARGE_UNPACKED_THRESHOLD_BYTES: int = 1 * 1024 * 1024 * 1024  # 1 GB

# Подозрительно большое количество вложенных файлов
_MAX_NESTED_FILES_HIGH: int = 1000
_MAX_NESTED_FILES_MEDIUM: int = 100


@DetectorRegistry.register
class CompressionDetector(BaseDetector):
    """Детектор атак через сжатие (zip-бомбы, decompression bombs).

    Анализирует raw_data.nested_files и raw_data.file_size для выявления:

    1. Zip-бомбы:
       - Маленький сжатый файл (< 1 МБ), но суммарный размер nested_files > 1 ГБ.
       - Общее соотношение распакованного к сжатому > MAX_COMPRESSION_RATIO.

    2. Подозрительное количество вложений:
       - > 1000 вложенных файлов → HIGH (нетипично для ML-файлов).
       - > 100 вложенных файлов → MEDIUM (подозрительно, но возможно).
    """

    name: ClassVar[str] = "compression"
    description: ClassVar[str] = "Детектор zip-бомб и decompression-атак в ML-файлах"
    severity_range: ClassVar[tuple[Severity, Severity]] = (Severity.MEDIUM, Severity.CRITICAL)

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Проверяет соотношение размеров и количество вложений.

        Алгоритм:
        1. Если nested_files отсутствуют — нечего анализировать.
        2. Вычисляет суммарный размер вложенных файлов (рекурсивно).
        3. Проверяет признаки zip-бомбы по абсолютным и относительным порогам.
        4. Проверяет количество вложений.
        """
        if raw_data.error is not None:
            return []

        if not raw_data.nested_files:
            return []

        issues: list[Issue] = []

        # Подсчёт вложений и суммарного распакованного размера
        nested_count = _count_nested_recursive(raw_data)
        total_unpacked = _sum_nested_size_recursive(raw_data)
        file_size = raw_data.file_size

        location = str(raw_data.file_path)

        # --- Проверка 1: zip-бомба по соотношению ---
        if file_size > 0 and total_unpacked > 0:
            ratio = total_unpacked / file_size
            if (
                file_size < _SMALL_FILE_THRESHOLD_BYTES
                and total_unpacked > _LARGE_UNPACKED_THRESHOLD_BYTES
            ):
                issues.append(
                    _make_zipbomb_issue(
                        raw_data=raw_data,
                        location=location,
                        compressed_size=file_size,
                        unpacked_size=total_unpacked,
                        ratio=ratio,
                    )
                )
            elif ratio > _MAX_COMPRESSION_RATIO:
                issues.append(
                    _make_high_ratio_issue(
                        raw_data=raw_data,
                        location=location,
                        compressed_size=file_size,
                        unpacked_size=total_unpacked,
                        ratio=ratio,
                    )
                )

        # --- Проверка 2: подозрительное количество вложений ---
        if nested_count > _MAX_NESTED_FILES_HIGH:
            issues.append(
                _make_many_files_issue(
                    raw_data=raw_data,
                    location=location,
                    count=nested_count,
                    severity=Severity.HIGH,
                )
            )
        elif nested_count > _MAX_NESTED_FILES_MEDIUM:
            issues.append(
                _make_many_files_issue(
                    raw_data=raw_data,
                    location=location,
                    count=nested_count,
                    severity=Severity.MEDIUM,
                )
            )

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _count_nested_recursive(raw_data: RawScanData) -> int:
    """Подсчитывает общее количество вложенных файлов (рекурсивно).

    Учитывает файлы на всех уровнях вложенности.
    """
    if not raw_data.nested_files:
        return 0
    count = len(raw_data.nested_files)
    for nested in raw_data.nested_files:
        count += _count_nested_recursive(nested)
    return count


def _sum_nested_size_recursive(raw_data: RawScanData) -> int:
    """Суммирует распакованный объём всех вложенных файлов рекурсивно.

    Логика выбора источника (аудит #15):

    1. Если в metadata есть ``total_uncompressed_bytes`` — используем его.
       Это PyTorchScanner записывает ``sum(len(member_bytes))`` после
       распаковки ZIP. Без этого ``file_size`` nested-объекта показывает
       только размер pickle-payload, а не суммарный размер всех членов
       архива (включая бинарные тензоры). 1 МБ ZIP с 50 ГБ тензоров не
       детектировался как zip-bomb.

    2. Иначе fallback к старой логике: сумма ``file_size`` nested-объектов
       и их вложений. Это работает для JoblibScanner (один nested = один
       декомпрессированный pickle).
    """
    # Приоритет 1: явный размер из metadata (PyTorchScanner записывает после ZIP-распаковки)
    if raw_data.metadata is not None:
        explicit = raw_data.metadata.get("total_uncompressed_bytes")
        if explicit is not None:
            try:
                return int(explicit)
            except (ValueError, TypeError):
                pass

    # Приоритет 2: рекурсивная сумма file_size nested-файлов
    if not raw_data.nested_files:
        return 0
    total = 0
    for nested in raw_data.nested_files:
        total += nested.file_size
        total += _sum_nested_size_recursive(nested)
    return total


def _format_size(size_bytes: int) -> str:
    """Форматирует размер в байтах в человекочитаемый вид."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 ** 3):.1f} ГБ"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 ** 2):.1f} МБ"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} КБ"
    return f"{size_bytes} байт"


def _make_zipbomb_issue(
    raw_data: RawScanData,
    location: str,
    compressed_size: int,
    unpacked_size: int,
    ratio: float,
) -> Issue:
    """Создаёт Issue для подозрения на zip-бомбу.

    Severity и Confidence зависят от значения ratio (аудит #25):

    * ratio > 10 000 → CRITICAL/HIGH (extreme — почти точно bomb)
    * ratio 1000–10 000 → HIGH/MEDIUM (подозрительно, но возможно
      легитимное сжатие text-моделей или повторяющихся весов через
      ``joblib.dump(compress=9)`` или ``lz4``)

    Раньше всё это было CRITICAL/HIGH, что давало false positive на чистых
    sklearn-моделях с агрессивным сжатием.
    """
    is_extreme = ratio > _EXTREME_COMPRESSION_RATIO
    severity = Severity.CRITICAL if is_extreme else Severity.HIGH
    confidence = Confidence.HIGH if is_extreme else Confidence.MEDIUM

    return Issue(
        code="MLS-CMP-001",
        severity=severity,
        confidence=confidence,
        message=(
            f"Подозрение на zip-бомбу: файл {_format_size(compressed_size)}, "
            f"распакованный размер {_format_size(unpacked_size)} "
            f"(соотношение {ratio:.0f}x)"
        ),
        location=location,
        details={
            "compressed_size_bytes": compressed_size,
            "unpacked_size_bytes": unpacked_size,
            "compression_ratio": ratio,
            "threshold_ratio": _MAX_COMPRESSION_RATIO,
        },
        why=(
            "Аномально высокое соотношение сжатого и распакованного размеров "
            "(zip-бомба). При попытке распаковать такой файл может быть исчерпана "
            "оперативная память или дисковое пространство сервера."
        ),
        remediation=(
            "Не распаковывайте этот файл без изолированного окружения. "
            "Проверьте источник модели. Отклоните файл, если источник ненадёжен."
        ),
        compliance_tags=[
            "owasp-ml:ml10",
            "fstec:ubi-111",
            "gost:56939-2024:5.3",
            "gost:56939-2024:6.1",
        ],
    )


def _make_high_ratio_issue(
    raw_data: RawScanData,
    location: str,
    compressed_size: int,
    unpacked_size: int,
    ratio: float,
) -> Issue:
    """Создаёт Issue HIGH для подозрительно высокого соотношения сжатия."""
    return Issue(
        code="MLS-CMP-002",
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        message=(
            f"Подозрительное соотношение сжатия: {ratio:.0f}x "
            f"({_format_size(compressed_size)} → {_format_size(unpacked_size)})"
        ),
        location=location,
        details={
            "compressed_size_bytes": compressed_size,
            "unpacked_size_bytes": unpacked_size,
            "compression_ratio": ratio,
            "threshold_ratio": _MAX_COMPRESSION_RATIO,
        },
        why=(
            f"Соотношение распакованного к сжатому размеру ({ratio:.0f}x) "
            f"превышает порог {_MAX_COMPRESSION_RATIO}x. "
            "Это может указывать на намеренно созданный архив для исчерпания ресурсов."
        ),
        remediation=(
            "Распаковывайте с ограничением суммарного размера. "
            "Проверьте источник файла перед использованием."
        ),
        compliance_tags=[
            "owasp-ml:ml10",
            "fstec:ubi-111",
            "gost:56939-2024:5.3",
        ],
    )


def _make_many_files_issue(
    raw_data: RawScanData,
    location: str,
    count: int,
    severity: Severity,
) -> Issue:
    """Создаёт Issue HIGH/MEDIUM для подозрительного количества вложенных файлов."""
    threshold = (
        _MAX_NESTED_FILES_HIGH if severity == Severity.HIGH else _MAX_NESTED_FILES_MEDIUM
    )
    return Issue(
        code="MLS-CMP-003",
        severity=severity,
        confidence=Confidence.MEDIUM,
        message=f"Подозрительное количество вложенных файлов: {count}",
        location=location,
        details={
            "nested_file_count": count,
            "threshold": threshold,
        },
        why=(
            f"Обнаружено {count} вложенных файлов (порог: {threshold}). "
            "Большое количество вложений нетипично для ML-моделей и может "
            "быть признаком zip-бомбы или намеренно усложнённой структуры для "
            "обхода сканеров."
        ),
        remediation=(
            "Проверьте содержимое архива вручную. "
            "Типичная PyTorch-модель содержит не более 10-20 внутренних файлов."
        ),
        compliance_tags=[
            "owasp-ml:ml10",
            "fstec:ubi-111",
            "gost:56939-2024:5.3",
        ],
    )

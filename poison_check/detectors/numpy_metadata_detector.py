"""Детектор object-dtype массивов в NumPy-файлах (.npy, .npz).

Извлечён из NumpyScanner: сканер знает формат,
детектор знает угрозу. Раньше сканер сам конструировал Issue MLS-NPY-001,
но RawScanData не имеет поля ``issues`` — находка никуда не возвращалась
и пользователь её не видел.

Анализирует факты, оставленные сканером в metadata:
- ``object_dtype_detected`` — dtype массива равен object ('O');
- ``pickle_payload_detected`` — в данных массива найден pickle-поток;
- ``dtype`` / ``shape`` / ``npy_version`` — технические детали для отчёта.

Угроза: NumPy сериализует object-массивы через ``pickle.dumps()``, поэтому
любой object-массив — потенциальный носитель RCE-payload, срабатывающего
при ``numpy.load(allow_pickle=True)``. Сам по себе pickle-payload внутри
разбирается PickleScanner'ом, а его globals проверяются
AllowlistDetector / BlocklistDetector — этот детектор отвечает только за
сам факт object-dtype.

Для .npz каждый член архива — отдельный вложенный RawScanData, поэтому
детектор обходит ``nested_files`` рекурсивно и выдаёт issue с указанием
конкретного члена архива в location.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Reference,
    Severity,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.numpy_scanner import (
    META_OBJECT_DTYPE,
    META_PICKLE_PAYLOAD,
)

logger = logging.getLogger(__name__)

# Код находки object-dtype (namespace MLS-NPY-NNN закреплён за NumPy-форматом)
_ISSUE_CODE_OBJECT_DTYPE: str = "MLS-NPY-001"

# Имя сканера, чьи RawScanData обрабатывает детектор. Без этой проверки
# детектор был бы вынужден угадывать формат по содержимому metadata,
# что нарушает разделение слоёв.
_NUMPY_SCANNER_NAME: str = "numpy"

# Флаги, оставляемые NumpyScanner в metadata. Импортируются у производителя
# факта (зависимость Detector → Scanner разрешена архитектурой), чтобы имя
# ключа имело один источник истины.
_META_OBJECT_DTYPE: str = META_OBJECT_DTYPE
_META_PICKLE_PAYLOAD: str = META_PICKLE_PAYLOAD

# Ограничение глубины обхода nested_files: .npz может содержать .npy,
# внутри которого pickle — дальше вложенности не бывает. Лимит защищает
# от бесконечной рекурсии на некорректно собранном RawScanData.
_MAX_NESTED_DEPTH: int = 8


@DetectorRegistry.register
class NumpyMetadataDetector(BaseDetector):
    """Эмитит MLS-NPY-001 для массивов с dtype=object в .npy/.npz.

    Срабатывает только для RawScanData, полученных от NumpyScanner
    (проверяется по scanner_name). Обходит вложенные члены .npz, чтобы
    object-массив внутри архива не остался незамеченным.
    """

    name: ClassVar[str] = "numpy_metadata"
    description: ClassVar[str] = (
        "Детектор object-dtype массивов в NumPy-файлах (.npy, .npz)"
    )
    severity_range: ClassVar[tuple[Severity, Severity]] = (
        Severity.MEDIUM,
        Severity.MEDIUM,
    )

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Возвращает Issues по metadata NumPy-файла.

        :param raw_data: Результат сканирования.
        :param context: ML-контекст (не используется — object-dtype опасен
            независимо от определённого фреймворка).
        :return: Список Issues; пустой если scanner_name != 'numpy' либо
            object-dtype не обнаружен.
        """
        if raw_data.scanner_name != _NUMPY_SCANNER_NAME:
            return []

        issues: list[Issue] = []
        for entry in _iter_numpy_entries(raw_data):
            issue = _make_object_dtype_issue(entry)
            if issue is not None:
                issues.append(issue)
        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def _iter_numpy_entries(
    raw_data: RawScanData,
    depth: int = 0,
) -> Iterator[RawScanData]:
    """Обходит RawScanData и его numpy-вложения (члены .npz).

    Вложения от других сканеров (pickle-поток внутри object-массива)
    пропускаются: их анализируют свои детекторы.
    """
    if raw_data.scanner_name == _NUMPY_SCANNER_NAME:
        yield raw_data

    if depth >= _MAX_NESTED_DEPTH or not raw_data.nested_files:
        return

    for nested in raw_data.nested_files:
        if nested.scanner_name != _NUMPY_SCANNER_NAME:
            continue
        yield from _iter_numpy_entries(nested, depth + 1)


def _make_object_dtype_issue(raw_data: RawScanData) -> Issue | None:
    """Строит Issue MLS-NPY-001 если в metadata есть флаг object-dtype.

    Возвращает None если массив не object-dtype (либо metadata пуста).
    """
    metadata = raw_data.metadata
    if not metadata or metadata.get(_META_OBJECT_DTYPE) != "true":
        return None

    location = str(raw_data.file_path)
    display_name = Path(raw_data.file_path).name
    has_pickle = metadata.get(_META_PICKLE_PAYLOAD) == "true"

    details: dict[str, object] = {
        "dtype": metadata.get("dtype", ""),
        "shape": metadata.get("shape", ""),
        "npy_version": metadata.get("npy_version", ""),
        "pickle_payload_detected": has_pickle,
    }

    message = (
        f"Массив dtype=object в {display_name}: "
        "object-массивы сериализуются через pickle и могут "
        "содержать вредоносный код."
    )
    if has_pickle:
        message += " В данных массива найден pickle-поток."

    return Issue(
        code=_ISSUE_CODE_OBJECT_DTYPE,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        message=message,
        location=location,
        details=details,
        why=(
            "NumPy сохраняет object-массивы через pickle.dumps(). "
            "Злоумышленник может создать массив, payload которого "
            "выполняет произвольный код при вызове numpy.load()."
        ),
        remediation=(
            "Используйте numpy.load(allow_pickle=False) или "
            "конвертируйте модель в safetensors."
        ),
        references=[Reference(type="cwe", id="CWE-502")],
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )

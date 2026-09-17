"""Детектор угроз в Keras-моделях (.keras, .h5).

Извлечён из KerasScanner по принципу разделения слоёв:
сканер знает ФОРМАТ и складывает факты в metadata/strings, детектор знает
УГРОЗУ и эмитит Issue.

Анализирует факты KerasScanner:
- ``lambda_layer_count`` / ``lambda_layers`` — слои Lambda (CVE-2025-1550,
  произвольный код при load_model);
- ``custom_object_count`` / ``custom_objects`` — пользовательские custom-объекты
  (registered_name) — потенциальный код через custom_objects;
- ``h5py_available`` == "false" — .h5 не разобран, потому что обязательная
  зависимость h5py не импортируется: файл остался НЕПРОВЕРЕННЫМ (MLS-KERAS-003).

Срабатывает только для RawScanData от KerasScanner (scanner_name == "keras").
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
    Reference,
    Severity,
)
from poison_check.core.scanner_base import RawScanData

logger = logging.getLogger(__name__)

# Namespace MLS-KERAS-NNN закреплён за Keras-форматом.
_CODE_LAMBDA: str = "MLS-KERAS-001"
_CODE_CUSTOM: str = "MLS-KERAS-002"
_CODE_H5PY_MISSING: str = "MLS-KERAS-003"

_KERAS_SCANNER_NAME: str = "keras"

#: Severity непроверенного HDF5-файла (MLS-KERAS-003).
#: Базовый уровень MEDIUM. Выше нельзя: это не находка (CRITICAL — «точно RCE»,
#: HIGH — «скорее всего вредонос»), а отсутствие результата проверки; выдавать
#: HIGH по умолчанию значило бы обвинять файл, которого мы не видели.
#: Ниже нельзя: INFO по шкале Severity — «для полноты картины», справочная
#: запись; непроверенный файл справкой не является. Практическое следствие
#: INFO — SARIF-level "note" (GitHub Code Scanning по умолчанию его прячет)
#: и INFO в worst_severity, из-за чего файл читается как чистый. MEDIUM даёт
#: "warning" и честное «требует внимания».
_SEVERITY_UNVERIFIED: Severity = Severity.MEDIUM
#: То же при ``extra_rules.no_unverified_files`` (banking / government / strict):
#: там «не проверено» равносильно «не допущено», а fail_on_severity: high, —
#: поэтому HIGH, чтобы непроверенный файл валил гейт.
_SEVERITY_UNVERIFIED_STRICT: Severity = Severity.HIGH

_COMPLIANCE_TAGS_RCE: tuple[str, ...] = (
    "owasp-ml:ml03",
    "owasp-ml:ml10",
    "fstec:ubi-067",
    "gost:56939-2024:5.3",
)


@DetectorRegistry.register
class KerasThreatDetector(BaseDetector):
    """Эмитит MLS-KERAS-* по фактам, оставленным KerasScanner.

    Основная угроза — слой Lambda (CVE-2025-1550): Keras сериализует байткод
    произвольной Python-функции и исполняет его при ``keras.models.load_model()``.
    """

    name: ClassVar[str] = "keras"
    description: ClassVar[str] = "Детектор угроз в Keras-моделях (Lambda RCE, CVE-2025-1550)"
    severity_range: ClassVar[tuple[Severity, Severity]] = (
        _SEVERITY_UNVERIFIED,
        Severity.CRITICAL,
    )

    def __init__(self, escalate_unverified: bool = False) -> None:
        """Создаёт детектор Keras-угроз.

        :param escalate_unverified: Если True — непроверенный HDF5-файл
            (MLS-KERAS-003, h5py не импортируется) получает HIGH вместо MEDIUM.
            Значение приходит из ``extra_rules.no_unverified_files`` политик
            banking / government / strict: в этих отраслях файл, который сканер
            не смог разобрать, считается недопустимым к использованию, а не
            просто «требующим внимания». На MLS-KERAS-001/002 флаг не влияет —
            они и так CRITICAL/HIGH.
        """
        self.escalate_unverified = escalate_unverified

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Возвращает Issues по metadata Keras-файла.

        :param raw_data: Результат сканирования.
        :param context: ML-контекст (не используется — Lambda опасен независимо
            от определённого фреймворка).
        :return: Список Issues; пустой если scanner_name != 'keras' или фактов нет.
        """
        if raw_data.scanner_name != _KERAS_SCANNER_NAME:
            return []
        metadata = raw_data.metadata
        if not metadata:
            return []

        location = str(raw_data.file_path)
        issues: list[Issue] = []

        # 1. Обязательный h5py не импортируется → файл не проверен (MEDIUM/HIGH).
        #    Сканер при этом не падает (graceful degradation), но сигнал о
        #    непроверенном файле обязан быть виден в отчёте и в гейте CI.
        if metadata.get("h5py_available") == "false":
            severity = (
                _SEVERITY_UNVERIFIED_STRICT
                if self.escalate_unverified
                else _SEVERITY_UNVERIFIED
            )
            issues.append(_make_h5py_issue(location, severity))

        # 2. Слои Lambda → CRITICAL (CVE-2025-1550).
        lambda_count = _safe_int(metadata.get("lambda_layer_count"))
        if lambda_count > 0:
            issues.append(
                _make_lambda_issue(location, lambda_count, metadata.get("lambda_layers", ""))
            )

        # 3. Пользовательские custom-объекты → HIGH.
        custom_count = _safe_int(metadata.get("custom_object_count"))
        if custom_count > 0:
            issues.append(
                _make_custom_issue(location, custom_count, metadata.get("custom_objects", ""))
            )

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def _safe_int(value: str | None) -> int:
    """Разбирает целочисленный флаг из metadata; при ошибке — 0."""
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _make_lambda_issue(location: str, count: int, layers: str) -> Issue:
    """Строит MLS-KERAS-001 (Lambda RCE, CVE-2025-1550)."""
    layers_suffix = f" ({layers})" if layers else ""
    return Issue(
        code=_CODE_LAMBDA,
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        message=(
            f"Обнаружено слоёв Lambda: {count}{layers_suffix}. "
            "Lambda-слой Keras сериализует произвольный Python-код и исполняет "
            "его при загрузке модели (CVE-2025-1550)."
        ),
        location=location,
        details={"lambda_layer_count": count, "lambda_layers": layers},
        why=(
            "Слой Lambda хранит байткод Python-функции внутри модели. При вызове "
            "keras.models.load_model() этот код выполняется — в уязвимых версиях "
            "даже при safe_mode=True (CVE-2025-1550). Это прямой вектор RCE через "
            "поставку ML-модели."
        ),
        remediation=(
            "Не загружайте модель с Lambda-слоями из недоверенного источника. "
            "Запросите модель без Lambda (замените на штатные слои или подписанный "
            "custom-код) либо переведите в safetensors. Обновите Keras до "
            "версии с исправлением CVE-2025-1550."
        ),
        references=[
            Reference(type="cve", id="CVE-2025-1550"),
            Reference(type="cwe", id="CWE-502"),
            Reference(type="cwe", id="CWE-94"),
        ],
        compliance_tags=list(_COMPLIANCE_TAGS_RCE),
    )


def _make_custom_issue(location: str, count: int, objects: str) -> Issue:
    """Строит MLS-KERAS-002 (пользовательские custom-объекты)."""
    objects_suffix = f": {objects}" if objects else ""
    return Issue(
        code=_CODE_CUSTOM,
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        message=(
            f"Обнаружено пользовательских custom-объектов: {count}{objects_suffix}. "
            "Custom-объекты Keras могут ссылаться на произвольный Python-код."
        ),
        location=location,
        details={"custom_object_count": count, "custom_objects": objects},
        why=(
            "registered_name указывает на пользовательский класс/функцию, "
            "загружаемую через custom_objects при десериализации модели. Такой "
            "код не проверяется Keras и может выполнять произвольные действия."
        ),
        remediation=(
            "Убедитесь, что custom-объекты происходят из доверенного кода. "
            "Не загружайте модель, если источник custom-объектов неизвестен."
        ),
        references=[
            Reference(type="cwe", id="CWE-502"),
            Reference(type="cwe", id="CWE-913"),
        ],
        compliance_tags=list(_COMPLIANCE_TAGS_RCE),
    )


def _make_h5py_issue(location: str, severity: Severity) -> Issue:
    """Строит MLS-KERAS-003 — HDF5-файл не проверен (нет обязательного h5py).

    :param location: Путь к непроверенному файлу.
    :param severity: MEDIUM либо HIGH (см. ``escalate_unverified``).
    """
    return Issue(
        code=_CODE_H5PY_MISSING,
        severity=severity,
        confidence=Confidence.CERTAIN,
        message=(
            "Файл в формате HDF5 (.h5/.hdf5) НЕ ПРОВЕРЕН: обязательная зависимость "
            "h5py не импортируется — установка poison-check повреждена или неполна. "
            "Архитектура модели, включая слои Lambda (CVE-2025-1550), не разобрана."
        ),
        location=location,
        details={
            "dependency": "h5py",
            "dependency_required": True,
            "file_verified": False,
        },
        why=(
            "h5py входит в обязательные зависимости poison-check: это единственный "
            "путь к архитектуре HDF5-модели (root-атрибут model_config). Если он не "
            "импортируется, файл остаётся непроверенным — результат по нему не "
            "является заключением о безопасности. Вредоносная модель со слоем Lambda "
            "(произвольный код при load_model) в такой установке не будет обнаружена, "
            "то есть сканер обходится свойством окружения, а не содержимым файла."
        ),
        remediation=(
            "Восстановите установку: pip install --force-reinstall poison-check "
            "(либо pip install h5py в текущем окружении) и повторите сканирование "
            "этого файла. Проверить окружение целиком: poison-check doctor."
        ),
    )

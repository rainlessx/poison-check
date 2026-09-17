"""Детектор фактов, оставленных JoblibScanner в metadata (.joblib-файлы).

Извлечён из JoblibScanner: сканер знает формат,
детектор знает угрозу. Раньше сканер сам конструировал Issue
MLS-JOBLIB-001/002/003, но RawScanData не имеет поля ``issues`` — находки
никуда не возвращались и пользователь их не видел (заметка проекта
project_joblib_info_issues_dropped). Это тот же класс бага, что уже починен
для NumpyScanner — см. numpy_metadata_detector.py.

Анализирует факты, оставленные сканером в metadata:
- ``joblib_missing_codec`` == "lz4"  → MLS-JOBLIB-001 (INFO): не установлена lz4;
- ``joblib_missing_codec`` == "zstd" → MLS-JOBLIB-002 (INFO): не установлена zstandard;
- ``joblib_decompression_bomb`` == "true`` → decompression bomb, тип берётся из
  ``joblib_bomb_kind``:
    * "ratio"    → MLS-BOMB-001 (HIGH): крошечный вход, огромное раскрытие
      (крафтовая bomb, распознана ДО разбора pickle);
    * "absolute" (или ключ отсутствует) → MLS-JOBLIB-003 (MEDIUM): низкий ratio,
      но упёрлись в MAX_DECOMP (просто гигантская модель, скан усечён).

КРИТИЧНО: bomb ставит в RawScanData И error, И факт в metadata. Поэтому детектор
НЕ делает ранний ``if raw_data.error is not None: return []`` — иначе
MLS-BOMB-001 / MLS-JOBLIB-003 снова терялись бы (при bomb error всегда задан).
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
from poison_check.scanners.joblib_scanner import (
    META_BOMB_COMPRESSED_BYTES,
    META_BOMB_DECOMPRESSED_BYTES,
    META_BOMB_KIND,
    META_BOMB_LIMIT_BYTES,
    META_BOMB_METHOD,
    META_BOMB_RATIO,
    META_DECOMPRESSION_BOMB,
    META_MISSING_CODEC,
)

logger = logging.getLogger(__name__)

# Имя сканера, чьи RawScanData обрабатывает детектор. Без этой проверки детектор
# был бы вынужден угадывать формат по содержимому metadata, что нарушает
# разделение слоёв.
_JOBLIB_SCANNER_NAME: str = "joblib"

# Коды находок (namespace MLS-JOBLIB-NNN закреплён за joblib-форматом)
_ISSUE_CODE_LZ4_MISSING: str = "MLS-JOBLIB-001"
_ISSUE_CODE_ZSTD_MISSING: str = "MLS-JOBLIB-002"
# MLS-JOBLIB-003 (MEDIUM) — absolute-тип bomb: низкий ratio, но упёрлись в
# MAX_DECOMP (просто гигантская модель, скан усечён).
_ISSUE_CODE_DECOMPRESSION_BOMB: str = "MLS-JOBLIB-003"
# MLS-BOMB-001 (HIGH) — ratio-тип bomb: крошечный вход, огромное раскрытие
# (крафтовая decompression bomb, распознана ДО разбора pickle). Namespace BOMB
# закреплён за decompression-bomb находками (см. test_issue_codes_unique).
_ISSUE_CODE_RATIO_BOMB: str = "MLS-BOMB-001"

# Человекочитаемые сообщения об отсутствующих кодеках (перенесены из сканера).
_MISSING_CODEC_ISSUES: dict[str, tuple[str, str, str]] = {
    # codec → (code, package_name, install_hint)
    "lz4": (_ISSUE_CODE_LZ4_MISSING, "python-lz4", "pip install lz4"),
    "zstd": (_ISSUE_CODE_ZSTD_MISSING, "zstandard", "pip install zstandard"),
}


@DetectorRegistry.register
class JoblibMetadataDetector(BaseDetector):
    """Эмитит MLS-JOBLIB-001/002/003 по фактам, оставленным JoblibScanner.

    Срабатывает только для RawScanData, полученных от JoblibScanner
    (проверяется по scanner_name). Отсутствие опционального кодека (lz4/zstd)
    даёт INFO; decompression bomb — MEDIUM.
    """

    name: ClassVar[str] = "joblib_metadata"
    description: ClassVar[str] = (
        "Детектор фактов JoblibScanner: отсутствие кодека, decompression bomb"
    )
    severity_range: ClassVar[tuple[Severity, Severity]] = (
        Severity.INFO,
        Severity.HIGH,
    )

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Возвращает Issues по metadata joblib-файла.

        :param raw_data: Результат сканирования.
        :param context: ML-контекст (не используется — факты joblib не зависят
            от определённого фреймворка).
        :return: Список Issues; пустой если scanner_name != 'joblib' либо
            релевантных фактов в metadata нет.

        .. note::
           Детектор намеренно не делает ранний возврат при ``raw_data.error``:
           decompression bomb выставляет И error, И факт, поэтому MLS-JOBLIB-003
           обязан проверяться даже при заполненном error.
        """
        if raw_data.scanner_name != _JOBLIB_SCANNER_NAME:
            return []

        metadata = raw_data.metadata
        if not metadata:
            return []

        location = str(raw_data.file_path)
        issues: list[Issue] = []

        codec = metadata.get(META_MISSING_CODEC)
        if codec is not None:
            issue = _make_missing_codec_issue(codec, location)
            if issue is not None:
                issues.append(issue)

        if metadata.get(META_DECOMPRESSION_BOMB) == "true":
            if metadata.get(META_BOMB_KIND) == "ratio":
                # Крафтовая bomb (высокий ratio) → HIGH (MLS-BOMB-001).
                issues.append(_make_ratio_bomb_issue(metadata, location))
            else:
                # Absolute-тип (низкий ratio, упёрлись в MAX_DECOMP) либо старый
                # факт без kind → MEDIUM (MLS-JOBLIB-003).
                issues.append(_make_bomb_issue(metadata, location))

        return issues


def _parse_int(value: str | None) -> int | None:
    """Безопасно парсит строку-факт в int; None при отсутствии/нечисловом значении."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def _make_missing_codec_issue(codec: str, location: str) -> Issue | None:
    """Строит INFO-Issue об отсутствующей опциональной библиотеке кодека.

    Возвращает None для незнакомого значения codec (defensive: сканер кладёт
    только 'lz4'/'zstd').
    """
    entry = _MISSING_CODEC_ISSUES.get(codec)
    if entry is None:
        return None
    code, package_name, install_hint = entry
    return Issue(
        code=code,
        severity=Severity.INFO,
        confidence=Confidence.CERTAIN,
        message=(
            f"Файл сжат алгоритмом {codec}, но библиотека {package_name} "
            f"не установлена. Установите: {install_hint}"
        ),
        location=location,
        details={"compression": codec},
        why=(
            "Без библиотеки кодека сканер не может распаковать и проверить "
            "pickle-поток внутри joblib-файла — вредоносный payload остаётся "
            "непроверенным."
        ),
        remediation=(
            f"Установите библиотеку ({install_hint}) и повторите сканирование."
        ),
    )


def _make_bomb_issue(metadata: dict[str, str], location: str) -> Issue:
    """Строит MEDIUM-Issue MLS-JOBLIB-003 для обнаруженной decompression bomb.

    Текст перенесён из бывшего ``JoblibScanner._make_bomb_issue``. Метод и лимит
    берутся из фактов metadata (joblib_bomb_method / joblib_bomb_limit_bytes).
    """
    method = metadata.get(META_BOMB_METHOD, "unknown")
    limit_str = metadata.get(META_BOMB_LIMIT_BYTES)
    details: dict[str, object] = {"compression": method}
    limit_gb_text = ""
    if limit_str is not None:
        try:
            limit_bytes = int(limit_str)
        except ValueError:
            limit_bytes = None
        if limit_bytes is not None:
            details["limit_bytes"] = limit_bytes
            limit_gb_text = f" {limit_bytes // 1_000_000_000:.0f} ГБ"

    return Issue(
        code=_ISSUE_CODE_DECOMPRESSION_BOMB,
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        message=(
            f"Обнаружена возможная decompression bomb в {method}-потоке: "
            f"декомпрессированный размер превышает лимит{limit_gb_text}."
        ),
        location=location,
        details=details,
        why=(
            "Decompression bomb — файл, который после распаковки занимает "
            "в сотни раз больше места, чем в сжатом виде. "
            "Может привести к исчерпанию оперативной памяти (OOM) и отказу "
            "в обслуживании."
        ),
        remediation=(
            "Не используйте этот файл. Если источник доверенный — "
            "пересохраните модель без компрессии или в формате safetensors."
        ),
    )


def _make_ratio_bomb_issue(metadata: dict[str, str], location: str) -> Issue:
    """Строит HIGH-Issue MLS-BOMB-001 для ratio-типа decompression bomb.

    Ratio-bomb распознана сканером ДО разбора pickle: крошечный сжатый вход
    раскрывается в объём, во много раз превышающий его (коэффициент
    ``joblib_bomb_ratio`` ≥ порога). Это сигнатура крафтовой bomb, поэтому
    severity выше, чем у absolute-типа (MLS-JOBLIB-003). Факты берутся из
    metadata; Issue конструирует детектор (сканер знает формат, детектор —
    угрозу).
    """
    method = metadata.get(META_BOMB_METHOD, "unknown")
    ratio = metadata.get(META_BOMB_RATIO, "?")
    details: dict[str, object] = {"compression": method, "ratio": ratio}

    compressed_bytes = _parse_int(metadata.get(META_BOMB_COMPRESSED_BYTES))
    if compressed_bytes is not None:
        details["compressed_bytes"] = compressed_bytes
    read_bytes = _parse_int(metadata.get(META_BOMB_DECOMPRESSED_BYTES))
    if read_bytes is not None:
        details["decompressed_bytes_read"] = read_bytes

    return Issue(
        code=_ISSUE_CODE_RATIO_BOMB,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=(
            f"Обнаружена decompression bomb в {method}-потоке: коэффициент "
            f"распаковки ≥ {ratio}x (крошечный вход, огромное раскрытие). "
            "Чтение прервано до полной распаковки — истинный коэффициент выше."
        ),
        location=location,
        details=details,
        why=(
            "Decompression bomb — намеренно сжатый файл, который при штатной "
            "загрузке (joblib.load) разворачивается в сотни–тысячи раз, "
            "исчерпывая оперативную память и вызывая отказ в обслуживании "
            "(DoS/OOM). Аномально высокий коэффициент распаковки — прямой "
            "признак такой атаки."
        ),
        remediation=(
            "Не загружайте этот файл. Если источник доверенный — попросите "
            "пересохранить модель без агрессивного сжатия или в формате "
            "safetensors."
        ),
        compliance_tags=[
            "owasp-ml:ml10",
            "fstec:ubi-111",
            "gost:56939-2024:5.3",
        ],
    )

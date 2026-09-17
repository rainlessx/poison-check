"""Общий детектор ошибок разбора файла → Issue MLS-PARSE-001.

Закрывает инвариант «сканер никогда не теряет файл» на уровне отчёта: любой
пользовательский файл, который сканер не смог разобрать, ДОЛЖЕН породить хотя бы
одну Issue. Раньше сканер клал причину в ``RawScanData.error`` (и она доходила
до ``FileResult.error`` и консоли), но НИ ОДИН детектор не превращал общий сбой
разбора в Issue — файл выпадал из severity-таблицы (0 находок), хотя по нему
была ошибка. Это тот же класс «молча теряющегося сигнала», что уже чинился для
JoblibScanner (project_joblib_info_issues_dropped) и NumpyScanner.

Разделение слоёв: факт «не распарсилось» фиксирует
Scanner (поле ``error``), перевод факта в Issue — зона Detector. Этот детектор
общий: работает поверх RawScanData любого формата.

Чтобы не шуметь дублями, MLS-PARSE-001 эмитится ТОЛЬКО когда сбой разбора не
породил иного, более информативного сигнала. Подавляется, если сканер всё же
что-то извлёк или пометил специфический факт, у которого есть свой детектор:

* ``globals`` / ``reduce_calls`` / ``embedded_bytes`` — allowlist/blocklist/
  executable-детекторы уже сработают;
* metadata-флаги ``homoglyph_globals`` / ``parse_stop_attack`` /
  ``embedded_strings`` — их разбирает BlocklistDetector;
* decompression bomb (``joblib_decompression_bomb``) — по ней эмитится
  MLS-BOMB-001 / MLS-JOBLIB-003 (JoblibMetadataDetector), она важнее и выше;
* неверный magic GGUF (``gguf_bad_magic``) — по нему эмитится MLS-GGUF-006
  (GGUFMetadataDetector), структурированный и с GGUF-контекстом.

Во всех этих случаях файл НЕ потерян — по нему уже будет находка. MLS-PARSE-001
остаётся сеткой безопасности ровно для «чистого» сбоя разбора: обрыв опкода,
мусор после валидного заголовка, битый/усечённый контейнер.

Краевой случай «payload + оборванный хвост» (forensic-инвариант). Если поток
опкодов СНАЧАЛА извлёк опасный глобал (os.system + REDUCE → КРИТ), а ПОТОМ
genops упал на битом хвосте — это более сильный forensic-сигнал, чем просто
payload (признак попытки сорвать полный анализ после срабатывания). Отдельный
LOW-Issue тут не нужен (шум/дубль поверх КРИТ), НО сам факт обрыва не должен
исчезнуть: SARIF-форматтер строит вывод ТОЛЬКО из Issues и не эмитит
``FileResult.error`` — значит в CI/SIEM-канале факт терялся. Поэтому
``annotate_suppressed_parse_error`` протаскивает факт обрыва в ``details`` уже
эмитируемых находок (``parse_truncated`` / ``parse_error``), где он виден во всех
форматах (console/JSON/SARIF) И скоррелирован с самим payload — «найден И
оборван». Сырое имя глобала и сырой текст ошибки genops не нормализуются.
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

# Импорт факта bomb из joblib-сканера — зависимость Detector → Scanner разрешена
# архитектурой. По bomb эмитится своя, более специфичная Issue (MLS-BOMB-001 /
# MLS-JOBLIB-003), поэтому общий MLS-PARSE-001 для неё подавляется.
from poison_check.scanners.format_facts import META_FORMAT_UNSUPPORTED
from poison_check.scanners.gguf_scanner import META_BAD_MAGIC
from poison_check.scanners.joblib_scanner import META_DECOMPRESSION_BOMB

logger = logging.getLogger(__name__)

# Namespace PARSE закреплён за общими ошибками разбора формата
# (см. test_issue_codes_unique).
_ISSUE_CODE_PARSE_ERROR: str = "MLS-PARSE-001"

# metadata-флаги, у которых есть собственный детектор — при их наличии файл не
# потерян, и общий MLS-PARSE-001 не нужен (иначе дубль-шум).
_HANDLED_METADATA_FLAGS: tuple[str, ...] = (
    "homoglyph_globals",
    "parse_stop_attack",
    "embedded_strings",
    META_DECOMPRESSION_BOMB,
    META_BAD_MAGIC,  # неверный magic GGUF → MLS-GGUF-006 (GGUFMetadataDetector)
    # Неподдерживаемый формат — не сбой разбора: ни один сканер за файл не
    # брался, разбирать было нечего. Такой файл виден пользователю через
    # FileResult.error (консоль, JSON) и канал файловых фактов (SARIF), а
    # MLS-PARSE-001 по нему не эмитился никогда — детекторы для него просто не
    # запускались. Флаг сохраняет это поведение теперь, когда детекторы
    # запускаются и на неподдержанных файлах ради MLS-FMT-001: иначе каждый
    # README.md рядом с моделью давал бы LOW-находку «не удалось разобрать».
    META_FORMAT_UNSUPPORTED,
)

# Ключи details, которыми факт обрыва разбора протаскивается в уже эмитируемую
# находку (краевой случай «payload + оборванный хвост»). Оба видны в JSON/SARIF.
_DETAIL_PARSE_TRUNCATED: str = "parse_truncated"
_DETAIL_PARSE_ERROR: str = "parse_error"


@DetectorRegistry.register
class ParseErrorDetector(BaseDetector):
    """Эмитит MLS-PARSE-001 (LOW) для файла, который не удалось разобрать.

    Сетка безопасности: гарантирует, что упавший при разборе файл никогда не
    даёт ноль Issue. Срабатывает поверх любого формата — ключ только по
    ``raw_data.error`` и общим полям RawScanData, без привязки к сканеру.
    """

    name: ClassVar[str] = "parse_error"
    description: ClassVar[str] = (
        "Общий детектор ошибок разбора файла (MLS-PARSE-001)"
    )
    severity_range: ClassVar[tuple[Severity, Severity]] = (
        Severity.LOW,
        Severity.LOW,
    )

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Возвращает [MLS-PARSE-001] если файл не распарсился и нет иного сигнала.

        :param raw_data: Результат сканирования.
        :param context: ML-контекст (не используется — сбой разбора не зависит
            от определённого фреймворка).
        :return: Ровно одна Issue LOW при «чистом» сбое разбора, иначе [].
        """
        if raw_data.error is None:
            return []

        if _has_other_signal(raw_data):
            # По файлу уже будет находка от другого детектора — не дублируем.
            return []

        return [_make_parse_error_issue(raw_data)]


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def annotate_suppressed_parse_error(
    issues: list[Issue], raw_data: RawScanData
) -> None:
    """Протаскивает факт обрыва разбора в details уже эмитируемых находок.

    Вызывается агрегатором (Scanner-фасад / CLI) ПОСЛЕ прогона детекторов и
    дедупликации. Закрывает краевой случай «payload + оборванный хвост»: когда
    MLS-PARSE-001 подавлен (по файлу уже есть КРИТ по извлечённому глобалу), но
    поток опкодов после payload оборвался/повреждён. Отдельный Issue не
    создаётся (поведение «не шуметь» сохранено) — вместо этого в ``details``
    каждой находки кладётся ``parse_truncated=True`` и сырой текст ошибки
    ``parse_error``. Это единственный способ показать факт в SARIF (который
    несёт только Issues и игнорирует ``FileResult.error``) и одновременно
    скоррелировать его с самим payload.

    Мутирует ``issues`` на месте (симметрично dedupe_issues в том же слое).
    Срабатывает узко — только когда:

    * задан ``raw_data.error`` (разбор упал);
    * извлечён opcode-payload (``globals`` / ``reduce_calls``) — т.е. это именно
      «payload затем обрыв», а не чистый сбой (у того свой MLS-PARSE-001) и не
      bomb (у неё MLS-BOMB-001 с собственными details);
    * MLS-PARSE-001 в списке НЕТ (он подавлен — иначе факт уже в своей Issue).

    :param issues: Находки по файлу (после dedupe); мутируются на месте.
    :param raw_data: Результат сканирования файла.
    """
    if raw_data.error is None or not issues:
        return
    # Не «payload + обрыв»: чистый сбой (нет извлечённого payload) обрабатывает
    # сам MLS-PARSE-001; bomb — MLS-BOMB-001. Оба несут факт своими средствами.
    if not (raw_data.globals or raw_data.reduce_calls):
        return
    metadata = raw_data.metadata or {}
    if metadata.get(META_DECOMPRESSION_BOMB):
        return
    if any(issue.code == _ISSUE_CODE_PARSE_ERROR for issue in issues):
        return

    for issue in issues:
        # setdefault — не затираем, если детектор уже положил свой parse_*.
        issue.details.setdefault(_DETAIL_PARSE_TRUNCATED, True)
        issue.details.setdefault(_DETAIL_PARSE_ERROR, raw_data.error)


def _has_other_signal(raw_data: RawScanData) -> bool:
    """True, если сканер извлёк что-то, по чему сработает другой детектор.

    В этом случае файл не потерян и общий MLS-PARSE-001 не нужен.
    """
    if raw_data.globals or raw_data.reduce_calls or raw_data.embedded_bytes:
        return True
    metadata = raw_data.metadata
    if metadata:
        for flag in _HANDLED_METADATA_FLAGS:
            if metadata.get(flag):
                return True
    return False


def _make_parse_error_issue(raw_data: RawScanData) -> Issue:
    """Строит LOW-Issue MLS-PARSE-001 по ``raw_data.error``.

    Само сообщение об ошибке — поле данных (``details.error``), проходящее
    штатным путём в Output-слой; в Output оно не хардкодится.
    """
    location = str(raw_data.file_path)
    error_text = raw_data.error or "неизвестная ошибка разбора"
    return Issue(
        code=_ISSUE_CODE_PARSE_ERROR,
        severity=Severity.LOW,
        confidence=Confidence.LOW,
        message=(
            "Не удалось разобрать файл — возможна аномалия или обфускация "
            "формата. Содержимое осталось непроверенным."
        ),
        location=location,
        details={
            "scanner": raw_data.scanner_name,
            "error": error_text,
        },
        why=(
            "Сбой разбора означает, что статический анализатор не смог "
            "полностью проверить содержимое: файл может быть повреждён, усечён "
            "или намеренно сконструирован так, чтобы парсер остановился и "
            "пропустил вредоносный payload. Такой файл нельзя считать "
            "безопасным по умолчанию."
        ),
        remediation=(
            "Проверьте происхождение и целостность файла. Если источник "
            "доверенный — пересохраните модель в проверяемом формате "
            "(safetensors). Не загружайте файл до выяснения причины сбоя."
        ),
        compliance_tags=[
            "owasp-ml:ml03",
            "gost:56939-2024:5.3",
        ],
    )

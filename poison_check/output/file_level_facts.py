"""Единая точка сбора фактов УРОВНЯ ФАЙЛА для форматтеров вывода.

Факт уровня файла — это сведение о самом процессе/результате сканирования
файла, НЕ привязанное к конкретной находке (:class:`Issue`). Канонический
пример — ошибка разбора (``FileResult.error``): «файл не распарсился», «magic
bytes неверны», «контейнер повреждён». Такой факт НЕ является уязвимостью:
он не должен попадать в поток находок (SARIF ``results`` / severity-метрики),
но и не должен теряться — его надо донести до потребителя штатным каналом
каждого формата.

Проблема, которую закрывает модуль — системная асимметрия форматов вывода.
Console / JSON / SBOM несут ``FileResult.error`` (строка ошибки / поле
``error`` / property ``poison-check:scan_error``), а SARIF-форматтер строил
вывод ТОЛЬКО из ``file_result.issues`` и терял факт ЦЕЛИКОМ, если по файлу не
было ни одной Issue. Точечный обход (``annotate_suppressed_parse_error``
протаскивает ``parse_*`` в details находки) решает ОДИН случай — когда есть
Issue, к которой можно прицепить details, — и НЕ решает класс: файл с ошибкой и
нулём issues в SARIF пропадал бесследно.

Решение — ОДИН декларативный реестр полей-фактов (:data:`_FILE_LEVEL_FACT_FIELDS`).
Форматтеры итерируют его через :func:`iter_file_level_facts`, а не хардкодят
каждое поле. Новое поле-факт, добавленное в :class:`FileResult` завтра,
доезжает до всех форматтеров добавлением ОДНОЙ строки в реестр — без правки
логики форматтера. Это и есть «закрыть класс, а не случай».

Форензик-инвариант: текст факта берётся из поля ``FileResult`` КАК ЕСТЬ, без
нормализации (сырой текст ошибки genops / magic bytes и т.п.).

Слой: Output. Зависит только от Core (``core.result.FileResult``), что
соответствует правилу зависимостей сверху вниз.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from poison_check.core.result import FileResult

#: Стабильный идентификатор дескриптора для факта «ошибка разбора файла».
#: Используется как ``notification.descriptor.id`` в SARIF и как ключ реестра
#: дескрипторов ``tool.driver.notifications``. Формат ``tool/slug`` — принятая
#: в SARIF конвенция идентификаторов (ср. ``js/...`` у CodeQL).
SCAN_ERROR_DESCRIPTOR_ID: str = "poison-check/scan-error"


class FileFactLevel(Enum):
    """Уровень факта уровня файла в терминах SARIF notification.level.

    Значения совпадают со строками SARIF (``note`` / ``warning`` / ``error``),
    чтобы форматтер отдавал ``.value`` без дополнительного маппинга.
    """

    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class FileLevelFact:
    """Один факт уровня файла, готовый к выводу штатным каналом формата.

    :ivar descriptor_id: Стабильный идентификатор класса факта (для
        ``notification.descriptor.id`` и дедупликации дескрипторов).
    :ivar level: Уровень факта (:class:`FileFactLevel`).
    :ivar message: СЫРОЙ текст факта. Форензик-инвариант — без нормализации.
    """

    descriptor_id: str
    level: FileFactLevel
    message: str


@dataclass(frozen=True)
class _FileFactField:
    """Декларативная привязка поля ``FileResult`` к факту уровня файла.

    Единица реестра :data:`_FILE_LEVEL_FACT_FIELDS`. Чтобы завести новый факт,
    достаточно добавить сюда одну запись (и, если нужно, поле в ``FileResult``);
    форматтеры не меняются.
    """

    attr: str            # имя поля FileResult
    descriptor_id: str   # id дескриптора факта
    level: FileFactLevel  # уровень факта


#: Единый реестр полей ``FileResult``, несущих факты уровня файла.
#: ДОБАВИТЬ новый факт = добавить ОДНУ строку сюда (+ при необходимости поле в
#: ``FileResult``). Форматтеры (SARIF и любые будущие) не правятся — они
#: итерируют этот реестр через :func:`iter_file_level_facts`.
_FILE_LEVEL_FACT_FIELDS: tuple[_FileFactField, ...] = (
    _FileFactField(
        attr="error",
        descriptor_id=SCAN_ERROR_DESCRIPTOR_ID,
        level=FileFactLevel.ERROR,
    ),
)


def iter_file_level_facts(file_result: FileResult) -> list[FileLevelFact]:
    """Возвращает все факты уровня файла для одного :class:`FileResult`.

    Итерирует декларативный реестр :data:`_FILE_LEVEL_FACT_FIELDS`; для каждого
    непустого поля-факта строит :class:`FileLevelFact` с СЫРЫМ текстом значения
    (форензик-инвариант — без нормализации). Порядок фактов детерминирован
    (порядок реестра) — важно для воспроизводимости вывода в CI/CD.

    :param file_result: Результат сканирования одного файла.
    :return: Список фактов уровня файла (пустой, если фактов нет).
    """
    facts: list[FileLevelFact] = []
    for spec in _FILE_LEVEL_FACT_FIELDS:
        value = getattr(file_result, spec.attr, None)
        if value:
            facts.append(
                FileLevelFact(
                    descriptor_id=spec.descriptor_id,
                    level=spec.level,
                    message=str(value),
                )
            )
    return facts


def known_fact_descriptor_ids() -> list[str]:
    """Стабильный дедуплицированный список id всех дескрипторов фактов.

    Используется форматтерами, которым нужно заранее объявить дескрипторы
    (например, SARIF ``tool.driver.notifications``), независимо от того,
    сработал ли факт в конкретном прогоне.

    :return: id дескрипторов в порядке реестра, без повторов.
    """
    ids: list[str] = []
    for spec in _FILE_LEVEL_FACT_FIELDS:
        if spec.descriptor_id not in ids:
            ids.append(spec.descriptor_id)
    return ids

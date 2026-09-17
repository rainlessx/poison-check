"""Единая точка вывода МЕТАДАННЫХ ШАПКИ ОТЧЁТА (заказчик / аудитор).

Метаданные отчёта — это сведения не о модели и не о находках, а о самом
аудите: для кого он выполнен (``--client``) и кем (``--auditor``). Для
пентестерского отчёта по ГОСТ Р 56939-2024 это обязательная часть титульного
листа и листа подписи, поэтому терять их нельзя ни в одном формате вывода.

Проблема, которую закрывает модуль — та же системная асимметрия форматов, что
и у :mod:`poison_check.output.file_level_facts`: значения ``--client`` и
``--auditor`` доезжали ТОЛЬКО до Jinja-контекста PDF, а в console/JSON/SARIF/
SBOM молча исчезали. Опция принималась парсером и не влияла на поведение —
ровно тот класс дефекта, что и «декоративный ключ политики».

Решение — ОДИН декларативный реестр полей (:data:`REPORT_METADATA_FIELDS`).
Каждое поле сразу несёт имена, под которыми оно выводится во всех форматах
(JSON-ключ, SARIF-property, SBOM-property, i18n-ключ подписи для console).
Форматтеры итерируют реестр через :func:`iter_report_metadata`, а не хардкодят
поля: новое поле шапки завтра доезжает до всех форматов добавлением ОДНОЙ
строки в реестр.

Инвариант «пустое опускается»: :func:`iter_report_metadata` возвращает только
непустые поля, поэтому незаданный ``--client`` не превращается в ``"client":
""`` — ключа просто нет.

Слой: Output. Зависит только от Core (``core.result``), что соответствует
правилу зависимостей сверху вниз.
"""

from __future__ import annotations

from dataclasses import dataclass

from poison_check.core.result import ReportMetadata, ScanResult


@dataclass(frozen=True)
class ReportMetadataField:
    """Декларация одного поля шапки отчёта и его имён во всех форматах.

    Единица реестра :data:`REPORT_METADATA_FIELDS`.

    :ivar attr: Имя поля в :class:`~poison_check.core.result.ReportMetadata`.
    :ivar json_key: Ключ в секции ``report_metadata`` JSON-отчёта.
    :ivar sarif_key: Ключ в ``runs[0].properties`` SARIF-отчёта.
    :ivar sbom_property: Имя property в ``metadata.properties`` CycloneDX SBOM.
    :ivar i18n_key: Ключ локализованной подписи для консольного вывода.
    """

    attr: str
    json_key: str
    sarif_key: str
    sbom_property: str
    i18n_key: str


@dataclass(frozen=True)
class ReportMetadataEntry:
    """Одно ЗАПОЛНЕННОЕ поле шапки отчёта, готовое к выводу.

    :ivar field: Декларация поля (имена во всех форматах).
    :ivar value: Значение как его задал пользователь, без нормализации.
    """

    field: ReportMetadataField
    value: str


#: Единый реестр полей шапки отчёта. ДОБАВИТЬ новое поле = добавить ОДНУ строку
#: сюда (+ поле в :class:`~poison_check.core.result.ReportMetadata` и подписи в
#: ``i18n/*.yaml``). Форматтеры не правятся.
REPORT_METADATA_FIELDS: tuple[ReportMetadataField, ...] = (
    ReportMetadataField(
        attr="client",
        json_key="client",
        sarif_key="client",
        sbom_property="poison-check:client",
        i18n_key="report.client",
    ),
    ReportMetadataField(
        attr="auditor",
        json_key="auditor",
        sarif_key="auditor",
        sbom_property="poison-check:auditor",
        i18n_key="report.auditor",
    ),
)


def build_report_metadata(
    client: str | None = None,
    auditor: str | None = None,
) -> ReportMetadata | None:
    """Собирает :class:`ReportMetadata` из «сырых» значений CLI.

    Строка из пробелов приравнивается к незаданному значению: пользователь,
    передавший ``--client " "``, не должен получить пустую строку в шапке
    отчёта.

    :param client: Значение ``--client`` или ``None``.
    :param auditor: Значение ``--auditor`` или ``None``.
    :return: Метаданные или ``None``, если не задано ни одно поле.
    """
    client_value = client.strip() if client else ""
    auditor_value = auditor.strip() if auditor else ""
    if not client_value and not auditor_value:
        return None
    return ReportMetadata(
        client=client_value or None,
        auditor=auditor_value or None,
    )


def report_metadata_of(result: ScanResult) -> ReportMetadata | None:
    """Возвращает метаданные шапки отчёта из результата сканирования.

    :param result: Результат сканирования.
    :return: Метаданные или ``None``, если они не задавались.
    """
    return result.report_metadata


def iter_metadata_entries(
    metadata: ReportMetadata | None,
) -> list[ReportMetadataEntry]:
    """Возвращает ЗАПОЛНЕННЫЕ поля шапки в порядке реестра.

    Незаполненные поля не возвращаются вовсе — благодаря этому каждый форматтер
    получает инвариант «пустое опускается» бесплатно, без собственных проверок.

    :param metadata: Метаданные шапки или ``None``.
    :return: Список записей (пустой, если метаданных нет).
    """
    if metadata is None:
        return []

    entries: list[ReportMetadataEntry] = []
    for field in REPORT_METADATA_FIELDS:
        value = getattr(metadata, field.attr, None)
        if value:
            entries.append(ReportMetadataEntry(field=field, value=str(value)))
    return entries


def iter_report_metadata(result: ScanResult) -> list[ReportMetadataEntry]:
    """Возвращает ЗАПОЛНЕННЫЕ поля шапки отчёта из результата сканирования.

    :param result: Результат сканирования.
    :return: Список записей (пустой, если метаданных нет).
    """
    return iter_metadata_entries(report_metadata_of(result))


def report_metadata_json(result: ScanResult) -> dict[str, str]:
    """Секция ``report_metadata`` JSON-отчёта.

    :param result: Результат сканирования.
    :return: Словарь ``json_key → значение``; пустой, если метаданных нет
        (в этом случае форматтер не выводит секцию вообще).
    """
    return {e.field.json_key: e.value for e in iter_report_metadata(result)}


def report_metadata_sarif_properties(result: ScanResult) -> dict[str, str]:
    """Property-bag ``runs[0].properties`` SARIF-отчёта.

    :param result: Результат сканирования.
    :return: Словарь ``sarif_key → значение``; пустой, если метаданных нет.
    """
    return {e.field.sarif_key: e.value for e in iter_report_metadata(result)}


def report_metadata_sbom_properties(result: ScanResult) -> list[dict[str, str]]:
    """Записи для ``metadata.properties`` CycloneDX SBOM.

    :param result: Результат сканирования.
    :return: Список ``{"name": ..., "value": ...}``; пустой, если метаданных нет.
    """
    return [
        {"name": e.field.sbom_property, "value": e.value}
        for e in iter_report_metadata(result)
    ]

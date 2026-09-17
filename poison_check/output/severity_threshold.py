"""Применение ``severity_threshold`` — порога ВНИМАНИЯ — к выводу отчёта.

**Семантика ключа (осознанный выбор, см. docs/policies.md).**
``severity_threshold`` — порог ВНИМАНИЯ, а НЕ порог отображения. Находка ниже
порога остаётся в отчёте во всех форматах, но помечается как подпороговая:
пользователь видит её, но она не претендует на его немедленное действие.
Отдельно и независимо работает ``fail_on_severity`` — порог ДЕЙСТВИЯ, от
которого зависит exit code.

Альтернатива «порог отображения» (issue ниже порога вообще не попадает в вывод)
отвергнута: она воскрешает класс дефекта «молча теряющийся сигнал», ради
закрытия которого выстроена половина пайплайна — MLS-PARSE-001 для
непарсящегося файла, MLS-BOMB-001 для бомбы, отдельный канал файловых фактов в
SARIF. Дефолтная политика имеет ``severity_threshold: medium``, а MLS-PARSE-001
это LOW — то есть при трактовке «скрывать» самый частый сценарий «файл не
проверен» исчезал бы из отчёта по умолчанию. Порог отчётности не должен уметь
делать непроверенный файл невидимым.

**Два инварианта, которые порог не может нарушить ни при каком значении.**

1. *CRITICAL и HIGH не уходят под порог.* Даже
   ``severity_threshold: critical`` не пометит HIGH-находку подпороговой:
   порог управляет вниманием к шуму, а не к эксплуатируемым находкам
   (см. :data:`NEVER_DEMOTED_SEVERITY`).
2. *Факты уровня файла не уходят под порог никогда.* «Файл не разобран»,
   «файл не проверен из-за отсутствия зависимости», «бомба», «разбор оборван» —
   это не находки по содержимому, а сообщение о том, что проверка НЕ состоялась.
   Их severity низкий по смыслу шкалы, но скрывать их нельзя (см.
   :data:`UNSUPPRESSIBLE_CODES`). Факты, которые живут не в Issue, а в
   ``FileResult.error``, порог вообще не проходят — они идут отдельным каналом
   :mod:`poison_check.output.file_level_facts`.

Слой: Output. Зависит только от Core и i18n — сканеры и детекторы про порог не
знают и знать не должны (порог влияет на отчёт, а не на анализ).
"""

from __future__ import annotations

from poison_check.core.result import Issue, ScanResult, Severity
from poison_check.i18n.loader import I18n

#: Ниже этого уровня порог не опускает: находки CRITICAL и HIGH остаются
#: основными при любом значении ``severity_threshold``.
NEVER_DEMOTED_SEVERITY: Severity = Severity.HIGH

#: Коды находок, которые сообщают «файл не проверен / проверен не полностью».
#: Формально это Issue, по смыслу — факт уровня файла, поэтому порог внимания
#: на них не распространяется. Реестр декларативный: новый код такого класса
#: добавляется ОДНОЙ строкой, без правки форматтеров.
UNSUPPRESSIBLE_CODES: frozenset[str] = frozenset(
    {
        "MLS-PARSE-001",   # общий сбой разбора файла
        "MLS-PKL-006",     # поток опкодов усечён по лимиту — разбор неполный
        "MLS-BOMB-001",    # ratio-бомба: разбор остановлен до конца потока
        "MLS-JOBLIB-001",  # нет lz4 — часть joblib-файла не распакована
        "MLS-JOBLIB-002",  # нет zstd — часть joblib-файла не распакована
        "MLS-JOBLIB-003",  # decompression bomb в joblib — разбор остановлен
        "MLS-GGUF-005",    # неизвестная версия GGUF — разбор мог быть неполным
        "MLS-GGUF-006",    # неверный magic GGUF — формат не тот, что заявлен
        "MLS-KERAS-003",   # нет h5py — .h5 не проверен на Lambda-RCE
        "MLS-CMP-001",     # zip-бомба: контейнер не развёрнут целиком
        "MLS-CMP-002",     # подозрительное соотношение сжатия
        "MLS-CMP-003",     # подозрительное число вложенных файлов
    }
)


def threshold_of(result: ScanResult) -> Severity | None:
    """Возвращает порог внимания прогона или ``None``, если он не задан.

    ``None`` означает поведение «как до появления порога»: ни одна находка не
    помечается подпороговой. Так ведут себя ``ScanResult``, собранные вручную
    (Python API, тесты) и политики без ключа ``severity_threshold``.

    :param result: Результат сканирования.
    :return: Порог внимания или ``None``.
    """
    if result.policy_thresholds is None:
        return None
    return result.policy_thresholds.severity_threshold


def is_below_threshold(issue: Issue, threshold: Severity | None) -> bool:
    """Является ли находка подпороговой при данном пороге внимания.

    :param issue: Находка.
    :param threshold: Порог внимания; ``None`` — порога нет.
    :return: ``True``, если находку нужно пометить как подпороговую.
    """
    if threshold is None:
        return False
    if issue.severity >= NEVER_DEMOTED_SEVERITY:
        return False
    if issue.code in UNSUPPRESSIBLE_CODES:
        return False
    return issue.severity < threshold


def partition_issues(
    issues: list[Issue],
    threshold: Severity | None,
) -> tuple[list[Issue], list[Issue]]:
    """Делит находки на основные и подпороговые, ничего не выбрасывая.

    Сумма длин результата всегда равна длине входа — это и есть инвариант
    «сигнал не теряется» в исполняемом виде.

    :param issues: Находки одного файла (или всего прогона).
    :param threshold: Порог внимания; ``None`` — все находки основные.
    :return: Кортеж ``(основные, подпороговые)`` с сохранением порядка.
    """
    primary: list[Issue] = []
    below: list[Issue] = []
    for issue in issues:
        (below if is_below_threshold(issue, threshold) else primary).append(issue)
    return primary, below


def count_below_threshold(result: ScanResult, threshold: Severity | None) -> int:
    """Считает подпороговые находки во всём прогоне.

    :param result: Результат сканирования.
    :param threshold: Порог внимания.
    :return: Количество находок, помеченных подпороговыми.
    """
    return sum(
        1
        for file_result in result.results_per_file.values()
        for issue in file_result.issues
        if is_below_threshold(issue, threshold)
    )


def below_threshold_justification(threshold: Severity | None) -> str:
    """Локализованное пояснение, почему находка помечена подпороговой.

    Используется машинными форматами (SARIF ``suppressions[].justification``,
    CycloneDX ``analysis.detail``), чтобы отметка не выглядела немотивированной
    у потребителя отчёта.

    :param threshold: Порог внимания.
    :return: Текст пояснения.
    """
    label = threshold.value if threshold is not None else ""
    return I18n.get().t("threshold.justification", threshold=label)

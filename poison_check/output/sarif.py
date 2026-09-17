"""SARIF 2.1.0 форматтер для результатов сканирования.

Стандарт SARIF (Static Analysis Results Interchange Format) используется
для интеграции с GitHub Code Scanning, Azure DevOps и SonarQube.

Спецификация: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html

Маппинг severity:
    CRITICAL → error
    HIGH     → error
    MEDIUM   → warning
    LOW      → warning
    INFO     → note

Два разных канала — по спецификации SARIF:

* ``runs[].results`` — НАХОДКИ (уязвимости) по содержимому артефакта. Один
  :class:`Issue` → один result. Это и есть метрики CI/SIEM по severity.
* ``runs[].invocations[].toolExecutionNotifications`` — факты УРОВНЯ ФАЙЛА о
  самом процессе сканирования (например, «файл не распарсился»,
  ``FileResult.error``). По SARIF 2.1.0 (§3.58 notification, §3.20 invocation)
  ошибки уровня выполнения инструмента над артефактом выражаются именно
  notification'ами, а НЕ result'ами. Обоснование выбора канала — в докстринге
  :meth:`SarifFormatter._build_invocations`. Факты уровня файла собираются
  через единую точку :mod:`poison_check.output.file_level_facts`, поэтому новый
  файловый факт, добавленный завтра, доезжает до SARIF без правки форматтера.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from poison_check.core.result import FileResult, Issue, ScanResult, Severity
from poison_check.output.file_level_facts import (
    FileLevelFact,
    iter_file_level_facts,
    known_fact_descriptor_ids,
)
from poison_check.output.report_metadata import report_metadata_sarif_properties
from poison_check.output.severity_threshold import (
    below_threshold_justification,
    is_below_threshold,
    threshold_of,
)

#: Дефолтное значение informationUri для SARIF tool.driver (аудит #29).
#: Заменяется на актуальный URL после публикации репозитория.
#: Переопределяется через переменную окружения ``POISON_CHECK_INFORMATION_URI``
#: или конструктор ``SarifFormatter(information_uri=...)``.
_DEFAULT_INFORMATION_URI: str = "https://github.com/poison-check/poison-check"

# ---------------------------------------------------------------------------
# Маппинг severity → SARIF level
# ---------------------------------------------------------------------------

_SEVERITY_TO_SARIF: dict[Severity, str] = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "warning",
    Severity.INFO: "note",
}


class SarifFormatter:
    """Форматтер результатов сканирования в SARIF 2.1.0 JSON.

    Совместим с GitHub Code Scanning: загружается через
    github/codeql-action/upload-sarif@v3.

    Структура вывода:
        runs[0].tool.driver.rules — дедуплицированные правила (по issue.code)
        runs[0].results          — все найденные проблемы
        runs[0].artifacts        — список просканированных файлов

    Пример использования:
        formatter = SarifFormatter()
        sarif_json = formatter.format(scan_result)
        Path("results.sarif").write_text(sarif_json, encoding="utf-8")
    """

    SARIF_VERSION: str = "2.1.0"
    SARIF_SCHEMA: str = (
        "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json"
    )
    TOOL_NAME: str = "poison-check"
    TOOL_INFORMATION_URI: str = _DEFAULT_INFORMATION_URI

    def __init__(self, information_uri: str | None = None) -> None:
        """Создаёт форматтер.

        :param information_uri: URI домашней страницы инструмента
            (аудит #29). По умолчанию используется ``POISON_CHECK_INFORMATION_URI``
            из окружения, иначе хардкод. SARIF-валидаторы (GitHub Code Scanning,
            Azure DevOps) проверяют, что URI существует, поэтому в production
            нужно либо настроить окружение, либо передать актуальный адрес явно.
        """
        if information_uri is not None:
            self._information_uri = information_uri
        else:
            self._information_uri = os.environ.get(
                "POISON_CHECK_INFORMATION_URI", self.TOOL_INFORMATION_URI
            )

    def format(self, result: ScanResult) -> str:
        """Сериализует ScanResult в SARIF 2.1.0 JSON.

        :param result: Результат сканирования.
        :return: Отформатированная JSON-строка, совместимая с GitHub Code Scanning.
        """
        # Собираем все issues из всех файлов
        all_issues_with_path: list[tuple[Path, Issue]] = [
            (file_path, issue)
            for file_path, file_result in result.results_per_file.items()
            for issue in file_result.issues
        ]

        # Дедуплицированные правила по issue.code
        rules = self._build_rules(all_issues_with_path)

        # Артефакты — все просканированные файлы
        artifacts = self._build_artifacts(result.results_per_file)

        # Результаты — каждый Issue → SARIF result. Порог внимания политики не
        # удаляет результаты, а помечает подпороговые (см. _issue_to_sarif_result).
        sarif_results = self._build_results(all_issues_with_path, threshold_of(result))

        # Факты уровня файла (FileResult.error и т.п.) → notifications, а не
        # results (см. _build_invocations). Не зависит от числа issues.
        invocations = self._build_invocations(result.results_per_file)

        run: dict[str, Any] = {
            "tool": {
                "driver": {
                    "name": self.TOOL_NAME,
                    "version": result.tool_version,
                    "informationUri": self._information_uri,
                    "rules": rules,
                    "notifications": self._build_notification_descriptors(),
                }
            },
            "invocations": invocations,
            "artifacts": artifacts,
            "results": sarif_results,
        }

        # Шапка отчёта (--client / --auditor) — в property bag прогона
        # (SARIF 2.1.0 §3.8 propertyBag). Это сведения об аудите, а не о
        # находке, поэтому им не место ни в results, ни в rules. Ключ
        # отсутствует целиком, если метаданные не заданы.
        run_properties = report_metadata_sarif_properties(result)
        if run_properties:
            run["properties"] = run_properties

        doc: dict[str, Any] = {
            "$schema": self.SARIF_SCHEMA,
            "version": self.SARIF_VERSION,
            "runs": [run],
        }

        return json.dumps(doc, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Внутренние методы построения структур
    # ------------------------------------------------------------------

    def _build_rules(
        self, issues: list[tuple[Path, Issue]]
    ) -> list[dict[str, Any]]:
        """Строит список дедуплицированных правил SARIF из issue.code.

        Каждый уникальный issue.code становится одним правилом.
        Для правила берём данные из первого встреченного Issue с этим кодом.
        """
        seen: dict[str, Issue] = {}
        for _path, issue in issues:
            if issue.code not in seen:
                seen[issue.code] = issue

        rules: list[dict[str, Any]] = []
        for code, issue in seen.items():
            rule: dict[str, Any] = {
                "id": code,
                "name": code,
                "shortDescription": {
                    "text": issue.message,
                },
                "defaultConfiguration": {
                    "level": _SEVERITY_TO_SARIF[issue.severity],
                },
                "properties": {
                    "severity": issue.severity.value,
                    "confidence": issue.confidence.value,
                },
            }
            # Добавляем ссылки если есть
            if issue.references:
                rule["helpUri"] = self._first_reference_url(issue)
            if issue.why:
                rule["fullDescription"] = {"text": issue.why}
            if issue.remediation:
                rule["help"] = {"text": issue.remediation}
            rules.append(rule)

        return rules

    def _build_artifacts(
        self, results_per_file: dict[Path, FileResult]
    ) -> list[dict[str, Any]]:
        """Строит список артефактов (просканированных файлов)."""
        artifacts: list[dict[str, Any]] = []
        for file_path in results_per_file:
            artifact: dict[str, Any] = {
                "location": {
                    "uri": self._path_to_uri(file_path),
                },
            }
            artifacts.append(artifact)
        return artifacts

    def _build_results(
        self,
        issues: list[tuple[Path, Issue]],
        threshold: Severity | None = None,
    ) -> list[dict[str, Any]]:
        """Строит список SARIF results из всех найденных Issues.

        Число result'ов не зависит от порога внимания: подпороговая находка
        остаётся в выводе и лишь помечается (``suppressions`` + properties).
        """
        sarif_results: list[dict[str, Any]] = []
        for file_path, issue in issues:
            sarif_result = self._issue_to_sarif_result(file_path, issue, threshold)
            sarif_results.append(sarif_result)
        return sarif_results

    # ------------------------------------------------------------------
    # Факты уровня файла → notifications (не results)
    # ------------------------------------------------------------------

    def _build_invocations(
        self, results_per_file: dict[Path, FileResult]
    ) -> list[dict[str, Any]]:
        """Строит ``runs[].invocations`` с фактами уровня файла как notifications.

        **Почему notifications, а не синтетический result (осознанный выбор).**
        По SARIF 2.1.0 ``result`` описывает НАХОДКУ по содержимому артефакта
        (уязвимость). Факт уровня файла — «файл не распарсился / повреждён / у
        него неверный magic» — это НЕ уязвимость, а состояние процесса
        сканирования над артефактом. Спецификация выделяет для этого отдельный
        канал: ``invocations[].toolExecutionNotifications`` (§3.58 notification,
        §3.20 invocation) с полями ``level`` / ``message`` / ``descriptor`` и
        привязкой к артефакту через ``locations``. Синтетический result был бы
        хуже: он засорил бы метрики находок в CI/SIEM (файловая ошибка ≠
        уязвимость) и исказил бы подсчёты по severity. Поэтому число result'ов в
        SARIF остаётся равным числу реальных Issue, а файловые факты идут
        отдельным каналом.

        **executionSuccessful=True — намеренно.** Сканер по инварианту не падает
        на пользовательском файле (повреждённый файл → ``RawScanData.error``, а
        не исключение), поэтому сам прогон инструмента всегда завершается
        успешно. Ошибка уровня ФАЙЛА — состояние артефакта, а не сбой
        инструмента; она выражается notification'ом уровня ``error``, но НЕ
        обнуляет ``executionSuccessful``. Иначе один битый входной файл заставил
        бы GitHub Code Scanning пометить весь анализ как проваленный и отклонить
        загрузку — это неверно семантически и вредно операционно.

        **Как потребитель это увидит.** GitHub Code Scanning показывает
        ``toolExecutionNotifications`` в статусе/деталях анализа (не как alert —
        alert'ы это results, поэтому alert-метрики не искажаются). SIEM,
        разбирающий SARIF, читает эти факты по пути
        ``runs[].invocations[].toolExecutionNotifications[]`` независимо от
        наличия находок — файл с ошибкой и нулём issues тоже виден.

        Факты берутся из :func:`iter_file_level_facts` — единой точки сбора
        файловых фактов. Новое поле-факт в ``FileResult`` доедет сюда без правки
        форматтера. Текст факта — сырой (форензик-инвариант).

        :param results_per_file: Результаты по файлам.
        :return: Список из одного invocation (per-run) с notifications.
        """
        notifications: list[dict[str, Any]] = []
        for file_path, file_result in results_per_file.items():
            for fact in iter_file_level_facts(file_result):
                notifications.append(
                    self._fact_to_notification(file_path, fact)
                )

        invocation: dict[str, Any] = {
            "executionSuccessful": True,
            "toolExecutionNotifications": notifications,
        }
        return [invocation]

    def _fact_to_notification(
        self, file_path: Path, fact: FileLevelFact
    ) -> dict[str, Any]:
        """Преобразует один :class:`FileLevelFact` в SARIF notification.

        Структура (§3.58):
            descriptor.id  = стабильный id класса факта
            level          = note/warning/error
            message.text   = СЫРОЙ текст факта (форензик — без нормализации)
            locations[0]   = привязка к артефакту (файлу)
        """
        return {
            "descriptor": {"id": fact.descriptor_id},
            "level": fact.level.value,
            "message": {"text": fact.message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self._path_to_uri(file_path),
                        },
                    },
                }
            ],
        }

    @staticmethod
    def _build_notification_descriptors() -> list[dict[str, Any]]:
        """Строит ``tool.driver.notifications`` — дескрипторы классов фактов.

        Объявляет все известные дескрипторы фактов уровня файла (стабильный
        набор из реестра :mod:`poison_check.output.file_level_facts`), чтобы
        ``notification.descriptor.id`` резолвился у потребителя. Список не
        зависит от конкретного прогона.
        """
        return [{"id": descriptor_id} for descriptor_id in known_fact_descriptor_ids()]

    def _issue_to_sarif_result(
        self,
        file_path: Path,
        issue: Issue,
        threshold: Severity | None = None,
    ) -> dict[str, Any]:
        """Преобразует одну Issue в SARIF result.

        Структура:
            ruleId       = issue.code
            level        = error/warning/note (маппинг из Severity)
            message.text = issue.message
            locations[0].physicalLocation.artifactLocation.uri = путь к файлу

        **Порог внимания в SARIF выражается ``suppressions`` (§3.27.23), а не
        удалением result'а.** Подпороговая находка остаётся в отчёте с исходным
        ``level`` и получает ``suppressions[0]`` с ``kind: "external"`` и
        обоснованием: порог задан внешней конфигурацией (политикой), а не
        комментарием в файле модели. Потребитель (GitHub Code Scanning)
        показывает такие результаты как dismissed — видно, но не блокирует; SIEM
        читает их как обычные результаты с признаком подавления. Полное
        удаление result'а было бы скрытием находки и нарушало бы инвариант
        «сигнал не теряется». ``level`` намеренно НЕ понижается: подпороговость
        — свойство подачи отчёта, а не самой угрозы.
        """
        # Извлекаем смещение из строки location если оно там есть.
        # Формат: "file.pkl (offset 1234)" или просто путь.
        region = self._parse_region(issue.location)

        physical_location: dict[str, Any] = {
            "artifactLocation": {
                "uri": self._path_to_uri(file_path),
            },
        }
        if region is not None:
            physical_location["region"] = region

        result: dict[str, Any] = {
            "ruleId": issue.code,
            "level": _SEVERITY_TO_SARIF[issue.severity],
            "message": {
                "text": issue.message,
            },
            "locations": [
                {
                    "physicalLocation": physical_location,
                }
            ],
        }

        # Дополнительные свойства для удобства
        below = is_below_threshold(issue, threshold)
        properties: dict[str, Any] = {
            "severity": issue.severity.value,
            "confidence": issue.confidence.value,
            "below_threshold": below,
        }
        if threshold is not None:
            properties["severity_threshold"] = threshold.value
        if below:
            result["suppressions"] = [
                {
                    "kind": "external",
                    "justification": below_threshold_justification(threshold),
                }
            ]
        if issue.details:
            properties["details"] = issue.details
        if issue.compliance_tags:
            properties["compliance_tags"] = issue.compliance_tags
        if issue.decompiled_code:
            properties["decompiled_code"] = issue.decompiled_code
        result["properties"] = properties

        # Ссылки (relatedLocations / help)
        if issue.references:
            result["relatedLocations"] = [
                {
                    "message": {"text": f"{ref.type.upper()}: {ref.id}"},
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self._reference_to_uri(ref.type, ref.id),
                        }
                    },
                }
                for ref in issue.references
            ]

        return result

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _path_to_uri(path: Path) -> str:
        """Преобразует Path в URI-формат для SARIF (RFC 3986).

        GitHub Code Scanning ожидает относительные URI без схемы file://.
        Абсолютные пути конвертируются в относительные от текущей директории.
        """
        try:
            rel = path.resolve().relative_to(Path.cwd().resolve())
            # Кодируем каждый сегмент пути
            return "/".join(quote(part, safe="") for part in rel.parts)
        except ValueError:
            # Путь не относительный — возвращаем как есть с прямыми слешами
            return str(path).replace("\\", "/")

    @staticmethod
    def _parse_region(location_str: str) -> dict[str, Any] | None:
        """Извлекает offset из строки location вида 'file.pkl (offset 1234)'.

        Возвращает SARIF region dict или None если offset не найден.
        """
        import re

        match = re.search(r"\(offset\s+(\d+)\)", location_str)
        if match:
            offset = int(match.group(1))
            # SARIF region: byteOffset (0-based)
            return {"byteOffset": offset}
        return None

    @staticmethod
    def _first_reference_url(issue: Issue) -> str:
        """Возвращает URL первой ссылки для helpUri правила."""
        ref = issue.references[0]
        return SarifFormatter._reference_to_uri(ref.type, ref.id)

    @staticmethod
    def _reference_to_uri(ref_type: str, ref_id: str) -> str:
        """Строит URL для CVE/CWE/БДУ ссылки."""
        ref_type_lower = ref_type.lower()
        if ref_type_lower == "cve":
            return f"https://nvd.nist.gov/vuln/detail/{ref_id}"
        if ref_type_lower == "cwe":
            cwe_num = ref_id.replace("CWE-", "").replace("cwe-", "")
            return f"https://cwe.mitre.org/data/definitions/{cwe_num}.html"
        if ref_type_lower == "bdu":
            return f"https://bdu.fstec.ru/threat/{quote(ref_id)}"
        # Общий случай
        return f"https://poison-check.readthedocs.io/refs/{quote(ref_id)}"

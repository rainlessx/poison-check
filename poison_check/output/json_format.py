"""JSON-форматтер для результатов сканирования.

Сериализует ScanResult в JSON строго по схеме v1.0.

Правила сериализации:
- schema_version: "1.0"
- datetime → ISO 8601 строка (timezone-aware, иначе добавляем 'Z')
- Path → str
- Enum → .value (str)
- dataclass-поля → dict
- None-значения → null (включены в вывод для явности схемы)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poison_check.core.result import (
    ComplianceReport,
    FileResult,
    Issue,
    MLContext,
    Reference,
    ScanResult,
    Severity,
    Summary,
)
from poison_check.output.report_metadata import report_metadata_json
from poison_check.output.severity_threshold import (
    count_below_threshold,
    is_below_threshold,
    threshold_of,
)


def _ser_datetime(dt: datetime) -> str:
    """Сериализует datetime в ISO 8601 строку.

    Если объект naive (без tzinfo) — добавляет суффикс 'Z' для обозначения UTC.
    Timezone-aware объекты — конвертирует в UTC и форматирует с 'Z'.
    """
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ser_reference(ref: Reference) -> dict[str, str]:
    """Сериализует ссылку CVE/CWE/БДУ."""
    return {"type": ref.type, "id": ref.id}


def _ser_issue(issue: Issue, threshold: Severity | None = None) -> dict[str, Any]:
    """Сериализует одну найденную проблему безопасности.

    Поле ``below_threshold`` — отметка порога ВНИМАНИЯ политики: находка ниже
    порога остаётся в отчёте (инвариант «сигнал не теряется»), но потребитель
    видит, что она не претендует на немедленное действие. При отсутствии порога
    поле всегда ``false``.
    """
    return {
        "code": issue.code,
        "severity": issue.severity.value,
        "confidence": issue.confidence.value,
        "message": issue.message,
        "location": issue.location,
        "details": issue.details,
        "why": issue.why,
        "remediation": issue.remediation,
        "decompiled_code": issue.decompiled_code,
        "references": [_ser_reference(r) for r in issue.references],
        "compliance_tags": issue.compliance_tags,
        "below_threshold": is_below_threshold(issue, threshold),
    }


def _ser_file_result(
    file_result: FileResult,
    ml_context: MLContext | None = None,
    threshold: Severity | None = None,
) -> dict[str, Any]:
    """Сериализует результат сканирования одного файла."""
    # Хеши берём из первого issues если есть, иначе пустые
    hashes: dict[str, str] = {}

    result: dict[str, Any] = {
        "file": {
            "path": str(file_result.file_path),
            "hashes": hashes,
            "size": None,
            "format": None,
        },
        "scanner": file_result.scanner_name,
        "ml_context": _ser_ml_context(ml_context) if ml_context else None,
        "duration_ms": file_result.duration_ms,
        "error": file_result.error,
        "issues": [_ser_issue(i, threshold) for i in file_result.issues],
    }
    return result


def _ser_ml_context(ctx: MLContext) -> dict[str, Any]:
    """Сериализует ML-контекст (фреймворк, уверенность, паттерны)."""
    return {
        "framework": ctx.framework,
        "confidence": ctx.confidence,
        "detected_patterns": ctx.detected_patterns,
    }


def _ser_summary(
    summary: Summary,
    worst_severity: Severity | None,
    below_threshold: int = 0,
) -> dict[str, Any]:
    """Сериализует агрегированные счётчики по severity.

    ``below_threshold`` — сколько находок помечено подпороговыми. Счётчики по
    severity при этом полные: порог внимания не вычитает находки из статистики,
    иначе метрики CI/SIEM зависели бы от настройки подачи отчёта.
    """
    return {
        "critical": summary.critical,
        "high": summary.high,
        "medium": summary.medium,
        "low": summary.low,
        "info": summary.info,
        "worst_severity": worst_severity.value if worst_severity else None,
        "blocked_by_policy": summary.blocked_by_policy,
        "below_threshold": below_threshold,
    }


def _ser_compliance(report: ComplianceReport) -> dict[str, Any]:
    """Сериализует compliance-маппинг.

    Поле ``disclaimer`` (аудит #30) включается всегда: автоматический маппинг
    УБИ ФСТЭК / OWASP ML / ГОСТ не верифицирован ИБ-аудитором, и заказчик
    должен это видеть явно.
    """
    return {
        "owasp_ml_top_10": report.owasp_ml_top_10,
        "fstec_ubi": report.fstec_ubi,
        "gost_references": report.gost_references,
        "disclaimer": report.disclaimer,
    }


class JsonFormatter:
    """Форматтер для вывода результатов сканирования в JSON.

    Сериализует ScanResult строго по схеме v1.0.
    Все datetime → ISO 8601, Path → str, Enum → .value, indent=2.
    """

    def format(
        self,
        result: ScanResult,
        ml_contexts: dict[Path, MLContext] | None = None,
    ) -> str:
        """Сериализует ScanResult в JSON-строку по схеме v1.0.

        :param result: Результат сканирования.
        :param ml_contexts: Опциональный словарь path → MLContext для включения
            ML-контекста в каждый файловый результат.
        :return: Отформатированная JSON-строка с indent=2.
        """
        contexts = ml_contexts or {}

        # Считаем общее количество файлов
        files_scanned = len(result.results_per_file)
        # total_bytes в schema — сумма размеров файлов.
        # FileResult не хранит размер явно, поэтому ставим None как честное значение.

        # Пересчитываем summary из реальных issues
        computed_summary = _compute_summary(result)

        # Пороги политики, с которыми выполнен прогон. Выводятся явно, чтобы по
        # отчёту было видно, ПОЧЕМУ находка помечена подпороговой и какой ключ
        # отвечал за exit code.
        threshold = threshold_of(result)
        thresholds = result.policy_thresholds
        fail_on = thresholds.fail_on_severity if thresholds is not None else None

        doc: dict[str, Any] = {
            "schema_version": "1.0",
            "tool": {
                "name": "poison-check",
                "version": result.tool_version,
                "policy": result.policy,
                "thresholds": {
                    "severity_threshold": threshold.value if threshold else None,
                    "fail_on_severity": fail_on.value if fail_on else None,
                },
            },
            "scan_info": {
                "timestamp": _ser_datetime(result.timestamp),
                "duration_ms": result.duration_ms,
                "files_scanned": files_scanned,
                "total_bytes": None,  # заполняется если доступно
            },
            "summary": _ser_summary(
                computed_summary,
                result.worst_severity,
                below_threshold=count_below_threshold(result, threshold),
            ),
            "results": [
                _ser_file_result(
                    fr,
                    ml_context=contexts.get(path),
                    threshold=threshold,
                )
                for path, fr in result.results_per_file.items()
            ],
            "compliance_report": (
                _ser_compliance(result.compliance_report)
                if result.compliance_report
                else None
            ),
        }

        # Шапка отчёта (--client / --auditor). Секция ОПУСКАЕТСЯ целиком, если
        # метаданные не заданы: пустые поля в отчёте выглядят как заполненные
        # пустотой реквизиты, а это хуже их отсутствия.
        metadata = report_metadata_json(result)
        if metadata:
            doc["report_metadata"] = metadata

        return json.dumps(doc, ensure_ascii=False, indent=2)


def _compute_summary(result: ScanResult) -> Summary:
    """Пересчитывает счётчики summary из реальных issues результата."""
    counts: dict[str, int] = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
    }
    for file_result in result.results_per_file.values():
        for issue in file_result.issues:
            key = issue.severity.value
            if key in counts:
                counts[key] += 1

    return Summary(
        critical=counts["critical"],
        high=counts["high"],
        medium=counts["medium"],
        low=counts["low"],
        info=counts["info"],
        blocked_by_policy=result.summary.blocked_by_policy,
    )

"""poison-check — ML Supply Chain Security Scanner.

Публичный API (Python API первого класса):

Базовое использование::

    from poison_check import Scanner

    result = Scanner().scan("model.pt")
    result = Scanner().scan("models/", recursive=True)
    file_result = Scanner().scan_bytes(raw_bytes, filename="model.pkl")

Forensics-режим (анализ конкретных issues, фильтрация по severity)::

    from poison_check import Scanner, Severity, Confidence

    result = Scanner(policy="banking").scan("suspect.pkl")
    for file_result in result.results_per_file.values():
        for issue in file_result.issues:
            if issue.severity >= Severity.HIGH and issue.confidence == Confidence.CERTAIN:
                print(f"{issue.code}: {issue.message}")
                if issue.decompiled_code:
                    print(issue.decompiled_code)

Программное создание Issue (для тестов / собственных детекторов)::

    from poison_check import Issue, Severity, Confidence, Reference

    custom = Issue(
        code="CUSTOM-001",
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        message="...",
        location="model.pkl",
    )
"""

from poison_check.core.result import (
    ComplianceReport,
    Confidence,
    FileResult,
    Issue,
    MLContext,
    Reference,
    ScanResult,
    Severity,
    Summary,
    dedupe_issues,
)
from poison_check.scanner import Scanner

__version__ = "0.1.0"

__all__ = [
    # Главный фасад
    "Scanner",
    # Версия
    "__version__",
    # Результаты сканирования
    "ScanResult",
    "FileResult",
    "Summary",
    "ComplianceReport",
    # Issues и их атрибуты
    "Issue",
    "Severity",
    "Confidence",
    "Reference",
    # ML-контекст
    "MLContext",
    # Утилиты
    "dedupe_issues",
]

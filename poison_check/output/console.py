"""Цветной консольный форматтер для результатов сканирования.

Использует библиотеку ``rich`` для цветного вывода в терминал.
Цвета severity подобраны под стандарты security-инструментов:

* CRITICAL → красный жирный + иконка 🚨 / [!]
* HIGH     → красный + иконка ⚠️ / [!]
* MEDIUM   → жёлтый + иконка ⚡ / [*]
* LOW      → синий + иконка ℹ️ / [i]
* INFO     → серый (dim) без иконки

Все строки берутся из I18n — это часть архитектурного принципа
"локализация как first-class feature".

ASCII-режим (аудит #26): на Windows cmd, в логах CI/CD и в SIEM-лентах
emoji часто превращаются в ``?``. Опция ``no_emoji=True`` (или CLI
``--no-emoji`` / переменная окружения ``POISON_CHECK_NO_EMOJI=1``)
переключает иконки на ASCII-fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from poison_check.core.result import (
    FileResult,
    Issue,
    ReportMetadata,
    ScanResult,
    Severity,
)
from poison_check.i18n.loader import I18n
from poison_check.output.report_metadata import iter_metadata_entries
from poison_check.output.severity_threshold import (
    count_below_threshold,
    partition_issues,
    threshold_of,
)

# Цвета по severity. Соответствуют негласному стандарту security-инструментов
# (Trivy, Grype, OSV) — критичное красное, среднее жёлтое, низкое синее.
_SEVERITY_STYLE: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "blue",
    Severity.INFO: "dim",
}

# Эмодзи-иконки для интерактивного терминала (default для UTF-8 TTY).
_SEVERITY_ICON_EMOJI: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "🚨",
    Severity.HIGH: "⚠️ ",
    Severity.MEDIUM: "⚡",
    Severity.LOW: "ℹ️ ",
    Severity.INFO: "  ",
}

# ASCII-fallback для Windows cmd, CI-логов, SIEM-лент.
# Использует двухсимвольные маркеры — выровнены по ширине с эмодзи (≈2 cells).
_SEVERITY_ICON_ASCII: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "[!]",
    Severity.HIGH: "[!]",
    Severity.MEDIUM: "[*]",
    Severity.LOW: "[i]",
    Severity.INFO: "[ ]",
}


def _should_use_emoji() -> bool:
    """Авто-детект: можно ли использовать emoji?

    Логика:

    1. ``POISON_CHECK_NO_EMOJI=1`` → нет (явный override).
    2. Не TTY (pipe / редирект / CI) → нет (логи и так не для людей).
    3. ``LANG``/``LC_ALL`` не содержит UTF-8 → нет.
    4. Windows без Windows Terminal (нет ``WT_SESSION``) → нет.
    5. Иначе → да.
    """
    if os.environ.get("POISON_CHECK_NO_EMOJI", "").strip() in ("1", "true", "yes"):
        return False
    if not sys.stdout.isatty():
        return False
    locale_env = (
        os.environ.get("LC_ALL")
        or os.environ.get("LC_CTYPE")
        or os.environ.get("LANG")
        or ""
    ).lower()
    if locale_env and "utf" not in locale_env:
        return False
    if sys.platform == "win32" and not os.environ.get("WT_SESSION"):
        return False
    return True


class ConsoleFormatter:
    """Форматтер для вывода результатов сканирования в консоль.

    Принимает опциональный ``Console`` из rich — это позволяет в тестах
    использовать ``Console(file=io.StringIO(), record=True)`` и проверять
    вывод без реального терминала.
    """

    def __init__(
        self,
        console: Console | None = None,
        i18n: I18n | None = None,
        no_emoji: bool | None = None,
        severity_threshold: Severity | None = None,
    ) -> None:
        """Создаёт форматтер.

        :param console: Опциональный экземпляр ``rich.console.Console``.
            Если не передан — создаётся стандартный.
        :param i18n: Опциональный экземпляр I18n. Если не передан —
            используется singleton ``I18n.get()``.
        :param no_emoji: ``True`` — использовать ASCII-иконки ``[!]/[*]/[i]``
            вместо эмодзи. ``None`` — авто-детект через ``_should_use_emoji()``
            (учитывает TTY, локаль, CI, Windows). ``False`` — принудительно эмодзи.
        :param severity_threshold: Порог ВНИМАНИЯ из политики. Находки ниже него
            не скрываются, а печатаются отдельным блоком с пометкой (см.
            :mod:`poison_check.output.severity_threshold`). ``None`` — порога
            нет, все находки выводятся одним списком.
        """
        self.console = console or Console()
        self.i18n = i18n or I18n.get()
        self._severity_threshold = severity_threshold
        if no_emoji is None:
            self._use_emoji = _should_use_emoji()
        else:
            self._use_emoji = not no_emoji
        self._severity_icons: dict[Severity, str] = (
            _SEVERITY_ICON_EMOJI if self._use_emoji else _SEVERITY_ICON_ASCII
        )

    # --- публичные методы вывода ---

    def format_scan_start(self, path: Path) -> None:
        """Выводит заголовок начала сканирования."""
        msg = self.i18n.t("cli.scan_start", path=str(path))
        self.console.print(Panel(Text(msg, style="bold cyan"), expand=False))

    def format_report_header(self, metadata: ReportMetadata | None) -> None:
        """Выводит шапку отчёта: заказчик и аудитор (``--client``/``--auditor``).

        Печатается только то, что задано: незаполненное поле не выводится
        вовсе (см. :mod:`poison_check.output.report_metadata`). Если не задано
        ничего — метод не печатает ни строки, и вывод не отличается от прогона
        без этих опций.

        :param metadata: Метаданные шапки отчёта или ``None``.
        """
        for entry in iter_metadata_entries(metadata):
            label = self.i18n.t(entry.field.i18n_key)
            self.console.print(f"{label}: {entry.value}", style="dim")

    def format_issue(self, issue: Issue) -> None:
        """Выводит одну проблему с цветом и иконкой по severity."""
        style = _SEVERITY_STYLE.get(issue.severity, "white")
        icon = self._severity_icons.get(issue.severity, "  ")
        sev_label = self.i18n.t(f"severity.{issue.severity.value}")

        # Шапка: иконка [SEVERITY] CODE — message
        header = Text()
        header.append(f"{icon} ", style=style)
        header.append(f"[{sev_label}] ", style=style)
        header.append(f"{issue.code} ", style="bold")
        header.append(f"— {issue.message}")
        self.console.print(header)

        # Расположение
        if issue.location:
            loc_label = self.i18n.t("issue.location")
            self.console.print(f"  {loc_label}: {issue.location}", style="dim")

        # Почему опасно
        if issue.why:
            why_label = self.i18n.t("issue.why")
            self.console.print(f"  {why_label}: {issue.why}")

        # Что делать
        if issue.remediation:
            rem_label = self.i18n.t("issue.remediation")
            self.console.print(f"  {rem_label}: {issue.remediation}", style="green")

        # Декомпилированный код — отдельной панелью
        if issue.decompiled_code:
            dec_label = self.i18n.t("issue.decompiled")
            self.console.print(
                Panel(issue.decompiled_code, title=dec_label, expand=False)
            )

        # Ссылки
        if issue.references:
            refs_label = self.i18n.t("issue.references")
            refs_text = ", ".join(f"{r.type.upper()}:{r.id}" for r in issue.references)
            self.console.print(f"  {refs_label}: {refs_text}", style="dim")

    def format_below_threshold_issue(self, issue: Issue) -> None:
        """Выводит подпороговую находку одной компактной строкой.

        Подпороговая находка не скрывается — она печатается кратко (уровень,
        код, сообщение, расположение), чтобы не конкурировать за внимание с
        основными находками, но остаться в отчёте и в логах CI.
        """
        style = _SEVERITY_STYLE.get(issue.severity, "white")
        sev_label = self.i18n.t(f"severity.{issue.severity.value}")

        line = Text("    ")
        line.append(f"[{sev_label}] ", style=style)
        line.append(f"{issue.code} ", style="bold dim")
        line.append(f"— {issue.message}", style="dim")
        if issue.location:
            line.append(f"  ({issue.location})", style="dim")
        self.console.print(line)

    def format_file_result(
        self,
        file_result: FileResult,
        threshold: Severity | None = None,
    ) -> None:
        """Выводит все issues для одного файла (и ошибку разбора, если была).

        Важно: наличие ``error`` больше НЕ прячет issues. Раньше ранний ``return``
        на ошибке подавлял вывод находок — упавший при разборе файл давал в
        консоль только строку-ошибку и ни одной Issue, хотя MLS-PARSE-001 /
        MLS-BOMB-001 по нему уже сформированы. Теперь печатаем и ошибку, и
        находки.

        Порог внимания политики (``severity_threshold``) НЕ убирает находки из
        вывода — он лишь выносит подпороговые в отдельный компактный блок под
        основными. Файловые факты и CRITICAL/HIGH туда не попадают никогда.

        :param file_result: Результат сканирования файла.
        :param threshold: Порог внимания; ``None`` — берётся из конструктора.
        """
        effective = threshold if threshold is not None else self._severity_threshold

        # Заголовок файла
        self.console.print()
        self.console.rule(str(file_result.file_path), style="cyan")

        if file_result.error is not None:
            err_msg = self.i18n.t(
                "errors.scan_error",
                path=str(file_result.file_path),
                error=file_result.error,
            )
            self.console.print(err_msg, style="bold red")

        if not file_result.issues:
            if file_result.error is None:
                self.console.print(self.i18n.t("cli.no_issues"), style="green")
            return

        primary, below = partition_issues(file_result.issues, effective)

        # Сортируем по убыванию severity, чтобы критичное было сверху.
        for issue in sorted(primary, key=lambda i: i.severity.level, reverse=True):
            self.format_issue(issue)

        if below:
            assert effective is not None  # непусто только при заданном пороге
            self.console.print(
                self.i18n.t("threshold.below_section", threshold=effective.value),
                style="dim",
            )
            for issue in sorted(below, key=lambda i: i.severity.level, reverse=True):
                self.format_below_threshold_issue(issue)

    def format_summary(self, result: ScanResult) -> None:
        """Выводит итоговую таблицу с распределением по severity."""
        # Пересчитываем счётчики по severity, не полагаемся на result.summary
        # (он мог быть не заполнен).
        groups = result.issues_by_severity()

        title = self.i18n.t("summary.title")
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column(self.i18n.t("table.severity"))
        table.add_column(self.i18n.t("table.count"), justify="right")

        # Идём от CRITICAL к INFO — порядок важен для восприятия.
        # В итоговой таблице НЕ используем emoji-иконки: у ⚠️/ℹ️ есть
        # variation selector U+FE0F, из-за которого Rich считает их шириной
        # 1 cell, а большинство терминалов рисуют 2 cells. В результате
        # столбец разъезжается. Иконки остаются в подробном выводе issue.
        for sev in (
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ):
            label = self.i18n.t(f"severity.{sev.value}")
            count = len(groups.get(sev, []))
            style = _SEVERITY_STYLE.get(sev, "white")
            table.add_row(
                Text(label, style=style),
                Text(str(count), style=style),
            )

        self.console.print()
        self.console.print(table)

        # Дополнительные строки итога
        files_count = len(result.results_per_file)
        self.console.print(
            self.i18n.t("cli.files_scanned", count=files_count), style="bold"
        )
        self.console.print(
            self.i18n.t("cli.scan_complete", duration_ms=result.duration_ms),
            style="bold",
        )

        # Порог внимания: сколько находок ушло в подпороговый блок. Строка
        # печатается только при заданном пороге и ненулевом счётчике — иначе
        # это шум. Здесь же явно сказано, что порог не влияет на гейт.
        threshold = threshold_of(result)
        if threshold is None:
            threshold = self._severity_threshold
        below_count = count_below_threshold(result, threshold)
        if threshold is not None and below_count:
            self.console.print(
                self.i18n.t(
                    "threshold.below_count",
                    threshold=threshold.value,
                    count=below_count,
                ),
                style="dim",
            )
            self.console.print(self.i18n.t("threshold.gate_note"), style="dim")

        total_issues = sum(len(g) for g in groups.values())
        critical_count = len(groups.get(Severity.CRITICAL, []))
        if total_issues == 0:
            self.console.print(self.i18n.t("cli.no_issues"), style="bold green")
        else:
            self.console.print(
                self.i18n.t("cli.issues_found", count=total_issues),
                style="bold yellow",
            )
            if critical_count > 0:
                self.console.print(
                    self.i18n.t("cli.critical_found", count=critical_count),
                    style="bold red",
                )

        # Дисклеймер о статусе верификации compliance-маппинга (аудит #30).
        # Печатается только если compliance_report присутствует, чтобы не
        # засорять вывод при отсутствии compliance-режима.
        if (
            result.compliance_report is not None
            and result.compliance_report.disclaimer
        ):
            self.console.print()
            self.console.print(
                Panel(
                    Text(result.compliance_report.disclaimer, style="dim"),
                    title="Compliance disclaimer",
                    border_style="yellow",
                    expand=True,
                )
            )

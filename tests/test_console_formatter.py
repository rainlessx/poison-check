"""Тесты для ConsoleFormatter и I18n."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pytest
from rich.console import Console

from poison_check.core.result import (
    Confidence,
    FileResult,
    Issue,
    Reference,
    ScanResult,
    Severity,
)
from poison_check.i18n.loader import I18n
from poison_check.output.console import ConsoleFormatter


# ---------- Хелперы ----------


def _make_console() -> Console:
    """Создаёт Console, пишущий в StringIO — чтобы вывод не уходил в TTY."""
    return Console(file=io.StringIO(), width=120, record=True, force_terminal=False)


def _make_formatter() -> ConsoleFormatter:
    return ConsoleFormatter(console=_make_console(), i18n=I18n("ru"))


def _make_issue(severity: Severity, code: str = "MLS-PKL-001") -> Issue:
    return Issue(
        code=code,
        severity=severity,
        confidence=Confidence.HIGH,
        message=f"Тестовое сообщение для {severity.value}",
        location="test.pkl (offset 0)",
        why="Опасный паттерн",
        remediation="Не использовать",
        references=[Reference(type="cve", id="CVE-2025-32434")],
    )


def _empty_scan_result() -> ScanResult:
    return ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(tz=timezone.utc),
        duration_ms=12.34,
    )


# ---------- I18n ----------


@pytest.fixture(autouse=True)
def _reset_i18n() -> None:
    """Сбрасываем singleton перед каждым тестом, чтобы локали не утекали."""
    I18n.reset()


def test_i18n_no_issues_returns_nonempty_string() -> None:
    """I18n.t для известного ключа возвращает непустую локализованную строку."""
    i18n = I18n("ru")
    msg = i18n.t("cli.no_issues")
    assert msg
    assert isinstance(msg, str)
    # На русском должно содержать "обнаружено" или иконку успеха.
    assert "обнаружено" in msg.lower() or "✅" in msg


def test_i18n_unknown_key_returns_key_itself() -> None:
    """Неизвестный ключ возвращается as-is, без исключения."""
    i18n = I18n("ru")
    key = "несуществующий.ключ"
    # Не должно бросать исключение.
    result = i18n.t(key)
    assert result == key


def test_i18n_format_kwargs_substituted() -> None:
    """Параметры форматирования подставляются через .format()."""
    i18n = I18n("ru")
    msg = i18n.t("cli.scan_start", path="/tmp/model.pt")
    assert "/tmp/model.pt" in msg


def test_i18n_fallback_to_english() -> None:
    """Если ключа нет в локали, но он есть в en.yaml — используется en."""
    # Создаём I18n с не существующей локалью — он отвалится в fallback.
    i18n = I18n("xx_nonexistent")
    msg = i18n.t("cli.no_issues")
    # Должны получить английскую строку, не сам ключ.
    assert msg != "cli.no_issues"
    assert msg


def test_i18n_singleton() -> None:
    """I18n.get() возвращает один и тот же инстанс."""
    a = I18n.get()
    b = I18n.get()
    assert a is b


def test_i18n_missing_format_arg_does_not_raise() -> None:
    """Если в шаблон не передан нужный kwarg — возвращается шаблон, не исключение."""
    i18n = I18n("ru")
    # cli.scan_start ждёт {path}; не передаём — не должно упасть.
    result = i18n.t("cli.scan_start")
    assert isinstance(result, str)
    assert result  # не пусто


# ---------- ConsoleFormatter ----------


def test_format_summary_empty_result_does_not_raise() -> None:
    """format_summary с пустым ScanResult не падает."""
    formatter = _make_formatter()
    result = _empty_scan_result()
    formatter.format_summary(result)  # не должно бросать


@pytest.mark.parametrize(
    "severity",
    [
        Severity.CRITICAL,
        Severity.HIGH,
        Severity.MEDIUM,
        Severity.LOW,
        Severity.INFO,
    ],
)
def test_format_issue_each_severity_does_not_raise(severity: Severity) -> None:
    """format_issue не падает ни для одного уровня severity."""
    formatter = _make_formatter()
    issue = _make_issue(severity)
    formatter.format_issue(issue)


def test_format_scan_start_does_not_raise() -> None:
    """format_scan_start не падает на обычном пути."""
    formatter = _make_formatter()
    formatter.format_scan_start(Path("/tmp/some_model.pt"))


def test_format_file_result_with_issues_does_not_raise() -> None:
    """format_file_result не падает при наличии issues."""
    formatter = _make_formatter()
    fr = FileResult(
        file_path=Path("/tmp/x.pkl"),
        scanner_name="pickle",
        issues=[
            _make_issue(Severity.CRITICAL, code="MLS-PKL-001"),
            _make_issue(Severity.LOW, code="MLS-ALW-001"),
        ],
        duration_ms=5.0,
    )
    formatter.format_file_result(fr)


def test_format_file_result_no_issues() -> None:
    """format_file_result без issues выводит сообщение об отсутствии угроз."""
    console = _make_console()
    formatter = ConsoleFormatter(console=console, i18n=I18n("ru"))
    fr = FileResult(file_path=Path("/tmp/clean.pkl"), scanner_name="pickle")
    formatter.format_file_result(fr)
    output = console.export_text()
    # На русском должна быть либо иконка ✅, либо слово "обнаружено".
    assert "✅" in output or "обнаружено" in output.lower()


def test_format_file_result_with_error() -> None:
    """format_file_result корректно отображает ошибку парсинга файла."""
    console = _make_console()
    formatter = ConsoleFormatter(console=console, i18n=I18n("ru"))
    fr = FileResult(
        file_path=Path("/tmp/broken.pkl"),
        scanner_name="pickle",
        error="malformed pickle stream",
    )
    formatter.format_file_result(fr)
    output = console.export_text()
    assert "broken.pkl" in output


def test_format_summary_with_issues_includes_counts() -> None:
    """format_summary с реальными issues показывает счётчики."""
    console = _make_console()
    formatter = ConsoleFormatter(console=console, i18n=I18n("ru"))

    file_path = Path("/tmp/x.pkl")
    fr = FileResult(
        file_path=file_path,
        scanner_name="pickle",
        issues=[
            _make_issue(Severity.CRITICAL),
            _make_issue(Severity.CRITICAL),
            _make_issue(Severity.MEDIUM),
        ],
    )
    result = _empty_scan_result()
    result.results_per_file[file_path] = fr

    formatter.format_summary(result)
    output = console.export_text()
    # Должны увидеть слова уровней и хотя бы цифру 2 (две критики).
    assert "КРИТИЧНО" in output
    assert "2" in output


# ---------------------------------------------------------------------------
# Регрессия аудита #26: ASCII-режим (--no-emoji)
# ---------------------------------------------------------------------------


class TestNoEmojiMode:
    """ConsoleFormatter поддерживает ASCII-fallback для иконок."""

    def test_no_emoji_true_uses_ascii(self) -> None:
        """no_emoji=True → ASCII-иконки [!]/[*]/[i]."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from poison_check.output.console import ConsoleFormatter

        buf = StringIO()
        rich_console = RichConsole(file=buf, force_terminal=False, width=120)
        formatter = ConsoleFormatter(console=rich_console, no_emoji=True)

        # Проверяем выбранный набор иконок
        assert formatter._severity_icons[Severity.CRITICAL] == "[!]"
        assert formatter._severity_icons[Severity.MEDIUM] == "[*]"
        assert formatter._severity_icons[Severity.LOW] == "[i]"

    def test_no_emoji_false_uses_emoji(self) -> None:
        """no_emoji=False → принудительно эмодзи."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from poison_check.output.console import ConsoleFormatter

        buf = StringIO()
        rich_console = RichConsole(file=buf, force_terminal=False, width=120)
        formatter = ConsoleFormatter(console=rich_console, no_emoji=False)
        # CRITICAL emoji содержит \U0001f6a8
        assert "🚨" in formatter._severity_icons[Severity.CRITICAL]

    def test_should_use_emoji_respects_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POISON_CHECK_NO_EMOJI=1 в env → эмодзи отключены."""
        from poison_check.output.console import _should_use_emoji  # noqa: PLC0415

        monkeypatch.setenv("POISON_CHECK_NO_EMOJI", "1")
        assert _should_use_emoji() is False

    def test_should_use_emoji_off_for_non_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Если stdout не TTY (pipe/CI), эмодзи отключены автоматически."""
        import sys as _sys

        from poison_check.output.console import _should_use_emoji  # noqa: PLC0415

        monkeypatch.delenv("POISON_CHECK_NO_EMOJI", raising=False)
        monkeypatch.setattr(_sys.stdout, "isatty", lambda: False)
        assert _should_use_emoji() is False

    def test_format_issue_no_emoji_does_not_raise(self) -> None:
        """ASCII-режим: format_issue не падает на CRITICAL issue."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from poison_check.core.result import Confidence, Issue
        from poison_check.output.console import ConsoleFormatter

        buf = StringIO()
        rich_console = RichConsole(file=buf, force_terminal=False, width=120)
        formatter = ConsoleFormatter(console=rich_console, no_emoji=True)

        issue = Issue(
            code="X-1", severity=Severity.CRITICAL, confidence=Confidence.HIGH,
            message="test", location="x.pkl",
        )
        formatter.format_issue(issue)
        output = buf.getvalue()
        assert "[!]" in output
        # Никаких эмодзи в ASCII-режиме
        assert "🚨" not in output

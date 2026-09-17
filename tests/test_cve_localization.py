"""Локализация текста CVE/PATTERN-находок по локали (ru по умолчанию, en по запросу).

Обёртка CLI локализуется через i18n; тексты правил (title/description/remediation)
живут в rules/cve/ml_cves.yaml в полях *_ru и *_en, а cve_detector выбирает язык
по текущей локали I18n.
"""

from __future__ import annotations

from pathlib import Path

from poison_check import Scanner
from poison_check.core.result import Issue
from poison_check.i18n.loader import I18n

FIX = Path(__file__).parent / "fixtures" / "malicious" / "payload_01_os_system.pkl"


def _os_system_issue(locale: str) -> Issue:
    """Сканирует известный os.system-pickle с заданной локалью и возвращает находку."""
    data = FIX.read_bytes()
    file_result = Scanner(locale=locale).scan_bytes(data, filename="payload.pkl")
    for issue in file_result.issues:
        if issue.code == "MLS-PATTERN-OS-SYSTEM":
            return issue
    raise AssertionError("MLS-PATTERN-OS-SYSTEM не найден в находках")


def test_cve_finding_english_under_locale_en() -> None:
    """При locale=en заголовок, «почему» и «что делать» — на английском."""
    issue = _os_system_issue("en")
    assert issue.message == "os.system/os.popen call detected in pickle stream"
    assert issue.why is not None
    assert issue.why.startswith("The pickle stream contains a call to os.system")
    assert issue.remediation is not None
    assert "Do not load this file" in issue.remediation


def test_cve_finding_russian_by_default() -> None:
    """Локаль по умолчанию (ru) — текст находки на русском, регрессии нет."""
    issue = _os_system_issue("ru")
    assert "Обнаружен вызов os.system/os.popen" in issue.message
    assert issue.why is not None
    assert "Pickle-файл содержит" in issue.why


def teardown_module(module: object) -> None:
    """Возвращает локаль по умолчанию, чтобы не влиять на другие тест-модули."""
    I18n.set_locale("ru")

"""Регрессионные тесты для bypass-техник обхода детекторов.

Каждый тест проверяет конкретную bypass-технику из security-research Opus:
- bypass_02: Cyrillic homoglyph → MLS-PKL-002 HIGH
- bypass_03: trusted_prefix + dangerous name → HIGH (не INFO)
- bypass_04: builtins.__import__ + getattr → HIGH
- bypass_09: PERSID opcode → MLS-PKL-003 HIGH
- bypass_10: torch impersonation + dangerous name → HIGH
"""
from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.core.result import Severity
from poison_check.scanner import Scanner

FIXTURES = Path(__file__).parent / "fixtures" / "malicious"


def _scan_severity(filename: str) -> Severity | None:
    """Сканирует фикстуру и возвращает наихудший severity."""
    path = FIXTURES / filename
    if not path.exists():
        pytest.skip(f"Фикстура не найдена: {path}")
    result = Scanner().scan(path)
    file_result = result.results_per_file.get(path)
    if file_result is None:
        return None
    if not file_result.issues:
        return None
    return max(i.severity for i in file_result.issues)


def _scan_codes(filename: str) -> set[str]:
    """Возвращает множество issue-кодов для файла."""
    path = FIXTURES / filename
    if not path.exists():
        pytest.skip(f"Фикстура не найдена: {path}")
    result = Scanner().scan(path)
    file_result = result.results_per_file.get(path)
    if file_result is None:
        return set()
    return {i.code for i in file_result.issues}


def test_bypass_02_homoglyph_detected() -> None:
    """Cyrillic homoglyph в GLOBAL opcode: 'оs' (U+043E + s) вместо 'os'."""
    worst = _scan_severity("bypass_02_homoglyph.pkl")
    assert worst is not None, "bypass_02: ни одного issue не выдано (MISSED)"
    assert worst >= Severity.HIGH, f"bypass_02: severity={worst}, ожидался HIGH+"
    codes = _scan_codes("bypass_02_homoglyph.pkl")
    assert "MLS-PKL-002" in codes, f"bypass_02: ожидался код MLS-PKL-002, получены {codes}"


def test_bypass_03_trusted_prefix_dangerous_name() -> None:
    """trusted_prefix 'torch.' + опасное имя 'system' → HIGH, не INFO."""
    worst = _scan_severity("bypass_03_trusted_prefix.pkl")
    assert worst is not None, "bypass_03: ни одного issue не выдано (MISSED)"
    assert worst >= Severity.HIGH, (
        f"bypass_03: severity={worst}, ожидался HIGH+ (trusted_prefix не должен "
        "защищать опасные имена функций)"
    )


def test_bypass_04_builtins_import_getattr() -> None:
    """builtins.__import__ + builtins.getattr → HIGH (indirect RCE chain)."""
    worst = _scan_severity("bypass_04_getattr_chain.pkl")
    assert worst is not None, "bypass_04: ни одного issue не выдано (MISSED)"
    assert worst >= Severity.HIGH, (
        f"bypass_04: severity={worst}, ожидался HIGH+ (indirect import chain)"
    )


def test_bypass_09_persid_detected() -> None:
    """PERSID opcode → MLS-PKL-003 HIGH."""
    worst = _scan_severity("bypass_09_persid_opcode.pkl")
    assert worst is not None, "bypass_09: ни одного issue не выдано (MISSED)"
    assert worst >= Severity.HIGH, f"bypass_09: severity={worst}, ожидался HIGH+"
    codes = _scan_codes("bypass_09_persid_opcode.pkl")
    assert "MLS-PKL-003" in codes, f"bypass_09: ожидался код MLS-PKL-003, получены {codes}"


def test_bypass_10_torch_impersonation() -> None:
    """torch.serialization.system: trusted_prefix 'torch.' + имя 'system' → HIGH."""
    worst = _scan_severity("bypass_10_torch_impersonation.pkl")
    assert worst is not None, "bypass_10: ни одного issue не выдано (MISSED)"
    assert worst >= Severity.HIGH, (
        f"bypass_10: severity={worst}, ожидался HIGH+ (torch impersonation)"
    )

"""Регрессионные тесты для четырёх багов, вскрытых аудитом заказчика.

Каждый тест закрепляет один из инвариантов, установленных при рефакторинге:

  1. ``test_dedupe_merges_across_severity`` — dedupe в ``result.dedupe_issues``
     сливает generic и specific находки в одну запись с максимальным
     severity и текстом от более специфичного правила. Раньше выбирался
     один «победитель» и вторая находка терялась либо оставалась дублем.

  2. ``test_protocol0_malicious_is_critical`` — pickle protocol 0 (текстовый,
     первый байт ASCII-опкод) распознаётся сканером как pickle и детекторы
     ловят ``os.system`` как CRITICAL. Раньше ``can_handle`` требовал magic
     ``\\x80\\x02..05`` и отбрасывал текстовый pickle с сообщением
     «неподдерживаемый формат».

  3. ``test_can_handle_rejects_random_binary`` — файл с pickle-расширением,
     но случайными байтами не проходит ``can_handle`` (не заваливает scan()
     ошибкой позже). Гибридная проверка «первый байт + genops первого
     опкода» рушится вместо шумного падения ниже по стеку.

  4. ``test_py2_alias_raw_in_location_normalized_for_lookup`` — pickle
     с ``__builtin__.__import__`` даёт ровно 1 issue: severity как у
     канонического ``builtins.__import__`` (HIGH), код ``MLS-PKL-001``,
     а ``location``/``message`` — сырые (Py2 форма) плюс подсказка
     «(Python 2 alias для builtins.__import__)». AllowlistDetector НЕ
     эмитит вторую запись — это ключевой инвариант: нормализация только
     на lookup, форензик-сырое в отчёте.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from poison_check.core.result import (
    Confidence,
    Issue,
    Reference,
    Severity,
    dedupe_issues,
)
from poison_check.scanner import Scanner
from poison_check.scanners.pickle_scanner import PickleScanner


# ---------------------------------------------------------------------------
# Тест 1: merge dedupe — разные severity, одна точка
# ---------------------------------------------------------------------------

def test_dedupe_merges_across_severity() -> None:
    """HIGH generic + CRITICAL specific на одном location → 1 issue CRITICAL,
    с текстом от specific и объединёнными ссылками.
    """
    same_location = "/tmp/mal.pkl:offset 42"

    generic = Issue(
        code="MLS-PKL-001",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message="Обнаружен запрещённый глобал builtins.__import__",
        location=same_location,
        details={"module": "builtins", "name": "__import__"},
        why="Generic reason",
        remediation="Generic remediation",
        references=[Reference(type="cwe", id="CWE-502")],
        compliance_tags=["fstec:ubi-067"],
    )
    specific = Issue(
        code="MLS-PATTERN-BUILTINS-INDIRECT-IMPORT",
        severity=Severity.CRITICAL,
        confidence=Confidence.CERTAIN,
        message="Indirect RCE via __import__",
        location=same_location,
        details={"cve_id": "PATTERN-BUILTINS-INDIRECT-IMPORT"},
        why="Specific detailed reason",
        remediation="Specific remediation",
        references=[
            Reference(type="cve", id="CVE-2024-XXXX"),
            Reference(type="cwe", id="CWE-502"),
        ],
        compliance_tags=["owasp-ml:ml03", "fstec:ubi-067"],
    )

    deduped = dedupe_issues([generic, specific])

    assert len(deduped) == 1
    result = deduped[0]
    # severity ушло вверх
    assert result.severity == Severity.CRITICAL
    # текст и код — от specific-записи
    assert result.code == "MLS-PATTERN-BUILTINS-INDIRECT-IMPORT"
    assert "Indirect RCE" in result.message
    # confidence — максимум по группе
    assert result.confidence == Confidence.CERTAIN
    # references объединены (уникальные)
    ref_pairs = {(r.type, r.id) for r in result.references}
    assert ("cve", "CVE-2024-XXXX") in ref_pairs
    assert ("cwe", "CWE-502") in ref_pairs
    # compliance-теги объединены (уникальные)
    assert "fstec:ubi-067" in result.compliance_tags
    assert "owasp-ml:ml03" in result.compliance_tags


def test_dedupe_reverse_case_critical_generic_high_specific() -> None:
    """CRITICAL generic + HIGH specific → CRITICAL specific.

    Проверяет, что severity не даунгрейдится ниже максимума в группе,
    а текст всё равно берётся от specific — обе оси мержа независимы.
    """
    loc = "/tmp/x.pkl:offset 10"
    generic = Issue(
        code="MLS-PKL-001",
        severity=Severity.CRITICAL,
        confidence=Confidence.CERTAIN,
        message="generic critical",
        location=loc,
        details={"module": "os", "name": "system"},
    )
    specific = Issue(
        code="MLS-PATTERN-OS-SYSTEM",
        severity=Severity.HIGH,  # маловероятно, но проверяем инвариант
        confidence=Confidence.HIGH,
        message="specific pattern",
        location=loc,
        details={"cve_id": "PATTERN-OS-SYSTEM"},
    )
    deduped = dedupe_issues([generic, specific])
    assert len(deduped) == 1
    assert deduped[0].severity == Severity.CRITICAL          # макс severity
    assert deduped[0].code == "MLS-PATTERN-OS-SYSTEM"        # текст от specific


# ---------------------------------------------------------------------------
# Тест 2: pickle protocol 0 — malicious CRITICAL
# ---------------------------------------------------------------------------

def test_protocol0_malicious_is_critical(tmp_path: Path) -> None:
    """Protocol 0 (текстовый pickle) с ``os.system`` даёт CRITICAL.

    Раньше ``can_handle`` требовал magic-байты protocol 2-5 и отбрасывал
    текстовый pickle с ``unsupported format``. Malware любит protocol 0
    именно потому, что он выглядит как ASCII-мусор — как раз то, что
    сканер обязан ловить.
    """
    import os as _os_module

    class _Payload:
        def __reduce__(self) -> tuple[object, tuple[str, ...]]:
            return (_os_module.system, ("echo pwn",))

    p = tmp_path / "proto0.pkl"
    with p.open("wb") as fh:
        pickle.dump(_Payload(), fh, protocol=0)

    # Убеждаемся, что первый байт — ASCII-опкод (не PROTO 0x80).
    with p.open("rb") as fh:
        first_byte = fh.read(1)
    assert first_byte != b"\x80", "фикстура должна быть protocol 0"

    # Сканер должен воспринять файл (иначе scanner_name будет 'unknown').
    assert PickleScanner.can_handle(p), (
        "PickleScanner.can_handle отбрасывает protocol 0 pickle — регрессия Fix 2 v2"
    )

    scanner = Scanner()
    result = scanner.scan(p)
    file_result = next(iter(result.results_per_file.values()))
    codes = {issue.code for issue in file_result.issues}
    severities = {issue.severity for issue in file_result.issues}

    assert Severity.CRITICAL in severities, (
        f"protocol 0 os.system должен быть CRITICAL, получены severities {severities}"
    )
    # Хотя бы один код от pattern-паттерна или generic-блоклиста должен быть
    assert any(
        c.startswith("MLS-PATTERN") or c == "MLS-PKL-001" for c in codes
    ), f"неожиданные коды: {codes}"


# ---------------------------------------------------------------------------
# Тест 3: can_handle отбрасывает случайный бинарник
# ---------------------------------------------------------------------------

def test_can_handle_rejects_random_binary(tmp_path: Path) -> None:
    """Файл с расширением ``.pkl`` но случайными байтами → ``can_handle`` = False.

    Регрессия: ранее ``can_handle`` фильтровал только по magic-байтам, что
    было слишком строгим для protocol 0. Fix 2 v1 ослабил проверку до
    whitelist первых байтов, но пропускал файл с валидным первым байтом
    и мусором дальше — ``scan()`` падал шумно на невалидном pickle-потоке.
    Fix 2 v2 добавил genops-валидацию первого опкода: whitelist + разбор.
    """
    bogus = tmp_path / "not_a_pickle.pkl"
    # Первый байт '#' (0x23) — не pickle-опкод. Fallback whitelist это ловит.
    bogus.write_bytes(b"#!/usr/bin/env python\nprint('hi')\n")
    assert not PickleScanner.can_handle(bogus)

    # Первый байт 'c' — валидный pickle-опкод (GLOBAL), но за ним нет
    # синтаксически корректного продолжения. Genops-этап должен это поймать.
    almost = tmp_path / "almost.pkl"
    almost.write_bytes(b"c")  # только один байт, opcode GLOBAL требует данных
    assert not PickleScanner.can_handle(almost)

    # Пустой файл — заведомо не pickle.
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(b"")
    assert not PickleScanner.can_handle(empty)


# ---------------------------------------------------------------------------
# Тест 4: Py2 alias — сырое в location, канон только для lookup
# ---------------------------------------------------------------------------

def test_py2_alias_raw_in_location_normalized_for_lookup(tmp_path: Path) -> None:
    """``__builtin__.__import__`` в pickle → 1 issue, severity как у канона,
    сырая пара в location/message, подсказка Py2 в message.

    Инвариант: нормализация применяется ТОЛЬКО при поиске в blocklist.
    Всё, что видит пользователь (location, message, details.module), —
    сырое имя, чтобы аналитик мог сразу опознать legacy-паттерн атаки.
    """
    # Строим pickle с GLOBAL __builtin__ __import__, TUPLE1("os"), REDUCE, STOP
    payload = b"\x80\x04"  # PROTO 4
    payload += b"c__builtin__\n__import__\n"  # GLOBAL __builtin__.__import__
    payload += b"\x8c\x02os"                   # SHORT_BINUNICODE 'os'
    payload += b"\x85"                          # TUPLE1
    payload += b"R"                             # REDUCE
    payload += b"."                             # STOP

    p = tmp_path / "py2.pkl"
    p.write_bytes(payload)

    scanner = Scanner()
    result = scanner.scan(p)
    file_result = next(iter(result.results_per_file.values()))
    issues = file_result.issues

    # Ровно один issue — нет дублей allowlist+blocklist.
    assert len(issues) == 1, (
        f"ожидался 1 issue, получено {len(issues)}: "
        f"{[(i.code, i.severity.value, i.message) for i in issues]}"
    )
    issue = issues[0]

    # severity как у канонического builtins.__import__ (HIGH).
    assert issue.severity == Severity.HIGH
    # generic blocklist-код (не allowlist-код).
    assert issue.code == "MLS-PKL-001"

    # location — файл+offset (без module.name); нормализация туда не утекает.
    assert issue.location.startswith(str(p)), (
        f"location должен ссылаться на файл: {issue.location}"
    )
    assert "offset" in issue.location, "location должен содержать offset"

    # message содержит СЫРОЕ имя и подсказку Py2 + канон в скобках.
    assert "__builtin__.__import__" in issue.message, (
        f"message потерял сырое имя: {issue.message}"
    )
    assert "Python 2 alias" in issue.message, (
        f"нет подсказки Py2 в message: {issue.message}"
    )
    assert "builtins.__import__" in issue.message, (
        "канон должен быть указан в скобках"
    )

    # details.module хранит СЫРОЕ имя — форензик-инвариант.
    assert issue.details.get("module") == "__builtin__"
    assert issue.details.get("name") == "__import__"


def test_py2_eval_normalized_to_builtins() -> None:
    """Быстрый smoke: normalize_global возвращает канон для eval-семейства."""
    from poison_check.core.known_dangerous import (
        KNOWN_DANGEROUS_GLOBALS,
        is_py2_alias,
        normalize_global,
    )

    # Пары канонизируются корректно
    assert normalize_global("__builtin__", "eval") == ("builtins", "eval")
    assert normalize_global("copy_reg", "_reconstructor") == ("copyreg", "_reconstructor")
    assert normalize_global("cStringIO", "StringIO") == ("io", "StringIO")

    # Каноничная пара присутствует в KNOWN_DANGEROUS_GLOBALS для критичных
    # функций — значит нормализованный lookup будет успешен.
    assert normalize_global("__builtin__", "eval") in KNOWN_DANGEROUS_GLOBALS
    assert normalize_global("__builtin__", "exec") in KNOWN_DANGEROUS_GLOBALS
    assert normalize_global("__builtin__", "__import__") in KNOWN_DANGEROUS_GLOBALS

    # is_py2_alias отвечает True на алиасы, False на канон
    assert is_py2_alias("__builtin__")
    assert is_py2_alias("copy_reg")
    assert not is_py2_alias("builtins")
    assert not is_py2_alias("os")


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_detector_cache() -> None:
    """Сбрасывает кэш детекторов между тестами.

    CLI и Scanner-фасад кэшируют детекторы по id(policy). При parametrize
    или последовательных тестах это может создать серию нюансов; здесь
    просто гарантируем, что каждый тест видит свежие инстансы.
    """
    from poison_check.cli import _DETECTOR_CACHE

    _DETECTOR_CACHE.clear()

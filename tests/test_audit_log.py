"""Тесты audit-лога (аудит #31).

Проверяет:
- запись JSON Lines с правильной структурой;
- no-op если ни флаг, ни env не заданы;
- безопасное проглатывание OSError при отсутствии прав;
- интеграция с CLI через --audit-log;
- интеграция через переменную окружения POISON_CHECK_AUDIT_LOG.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from poison_check.audit import write_audit_record
from poison_check.cli import app
from poison_check.core.result import (
    ComplianceReport,
    Confidence,
    FileResult,
    Issue,
    ScanResult,
    Severity,
    Summary,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_SAFE = _FIXTURES / "safe"
_MALICIOUS = _FIXTURES / "malicious"

runner = CliRunner()


def _make_scan_result(*, with_critical: bool = False) -> ScanResult:
    """Минимальный ScanResult для тестов audit."""
    issues: list[Issue] = []
    if with_critical:
        issues.append(
            Issue(
                code="MLS-PKL-001",
                severity=Severity.CRITICAL,
                confidence=Confidence.HIGH,
                message="test",
                location="x.pkl:0",
            )
        )
    return ScanResult(
        tool_version="0.1.0-test",
        timestamp=datetime.now(timezone.utc),
        duration_ms=42.5,
        scanned_paths=[Path("x.pkl")],
        policy="banking",
        results_per_file={
            Path("x.pkl"): FileResult(
                file_path=Path("x.pkl"),
                scanner_name="pickle",
                issues=issues,
            )
        },
        summary=Summary(critical=1 if with_critical else 0),
        compliance_report=ComplianceReport(disclaimer="test disclaimer"),
    )


# ---------------------------------------------------------------------------
# Базовое поведение write_audit_record
# ---------------------------------------------------------------------------


def test_audit_writes_jsonl_record(tmp_path: Path) -> None:
    """Запись добавляет одну строку с валидным JSON."""
    log_path = tmp_path / "audit.jsonl"
    result = _make_scan_result(with_critical=True)
    write_audit_record(result, Path("/tmp/x.pkl"), exit_code=1, audit_log_path=log_path)

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["tool_version"] == "0.1.0-test"
    assert record["scanned_path"] == "/tmp/x.pkl"
    assert record["policy"] == "banking"
    assert record["exit_code"] == 1
    assert record["issues_total"] == 1
    assert record["issues_by_severity"] == {"critical": 1}
    assert record["worst_severity"] == "critical"
    assert record["compliance_disclaimer_shown"] is True
    assert "user" in record
    assert "host" in record
    assert "pid" in record


def test_audit_appends_multiple_records(tmp_path: Path) -> None:
    """Несколько вызовов append'ятся в один файл."""
    log_path = tmp_path / "audit.jsonl"
    result = _make_scan_result()
    for _ in range(3):
        write_audit_record(result, Path("x.pkl"), exit_code=0, audit_log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # каждая строка — валидный JSON


def test_audit_noop_when_no_path_and_no_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Без --audit-log и POISON_CHECK_AUDIT_LOG ничего не пишется."""
    monkeypatch.delenv("POISON_CHECK_AUDIT_LOG", raising=False)
    result = _make_scan_result()
    # Не должно бросать
    write_audit_record(result, Path("x.pkl"), exit_code=0, audit_log_path=None)
    # И никакого файла не появилось в tmp_path
    assert list(tmp_path.iterdir()) == []


def test_audit_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POISON_CHECK_AUDIT_LOG=/path/to/log.jsonl активирует логирование."""
    log_path = tmp_path / "env.jsonl"
    monkeypatch.setenv("POISON_CHECK_AUDIT_LOG", str(log_path))
    result = _make_scan_result()
    write_audit_record(result, Path("x.pkl"), exit_code=0, audit_log_path=None)
    assert log_path.exists()


def test_audit_creates_parent_directory(tmp_path: Path) -> None:
    """Если родительской директории нет — она создаётся."""
    log_path = tmp_path / "deep" / "nested" / "audit.jsonl"
    result = _make_scan_result()
    write_audit_record(result, Path("x.pkl"), exit_code=0, audit_log_path=log_path)
    assert log_path.exists()


def test_audit_swallows_oserror(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Ошибка записи не валит сканирование, только warning в логе."""
    # Путь, в который точно нельзя писать на любой системе:
    # /proc на Linux — read-only filesystem для большинства путей.
    bad_path = Path("/proc/1/poison-check-audit-cant-write.jsonl")
    result = _make_scan_result()
    # Не должно бросить исключение
    write_audit_record(result, Path("x.pkl"), exit_code=0, audit_log_path=bad_path)


def test_audit_user_override(tmp_path: Path) -> None:
    """Параметр user= переопределяет getpass.getuser()."""
    log_path = tmp_path / "audit.jsonl"
    result = _make_scan_result()
    write_audit_record(
        result, Path("x.pkl"), exit_code=0,
        audit_log_path=log_path, user="alice",
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["user"] == "alice"


# ---------------------------------------------------------------------------
# Интеграция с CLI
# ---------------------------------------------------------------------------


def test_cli_audit_log_flag_writes_record(tmp_path: Path) -> None:
    """`scan --audit-log /path` создаёт запись после CLI-сканирования."""
    log_path = tmp_path / "cli-audit.jsonl"
    safe_file = _SAFE / "simple_list.pkl"

    result = runner.invoke(
        app,
        ["scan", str(safe_file), "--audit-log", str(log_path), "--format", "json"],
    )
    # exit code 0 для безопасного файла
    assert result.exit_code == 0, f"Output: {result.output}"
    assert log_path.exists()

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["scanned_path"] == str(safe_file)
    assert record["exit_code"] == 0
    assert record["policy"] == "default"


def test_cli_audit_log_env_var_writes_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POISON_CHECK_AUDIT_LOG в env активирует логирование без флага."""
    log_path = tmp_path / "env-audit.jsonl"
    monkeypatch.setenv("POISON_CHECK_AUDIT_LOG", str(log_path))
    safe_file = _SAFE / "simple_list.pkl"

    result = runner.invoke(app, ["scan", str(safe_file), "--format", "json"])
    assert result.exit_code == 0, f"Output: {result.output}"
    assert log_path.exists()


def test_cli_no_audit_flag_no_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без флага и env переменной audit-лог не создаётся."""
    monkeypatch.delenv("POISON_CHECK_AUDIT_LOG", raising=False)
    monkeypatch.chdir(tmp_path)
    safe_file = _SAFE / "simple_list.pkl"

    result = runner.invoke(app, ["scan", str(safe_file), "--format", "json"])
    assert result.exit_code == 0
    # tmp_path должен остаться пустым (никаких неожиданных файлов)
    assert list(tmp_path.iterdir()) == []


def test_cli_audit_log_records_critical_finding(tmp_path: Path) -> None:
    """Audit-лог отражает CRITICAL на вредоносном файле."""
    log_path = tmp_path / "audit.jsonl"
    malicious = _MALICIOUS / "payload_01_os_system.pkl"

    result = runner.invoke(
        app,
        ["scan", str(malicious), "--audit-log", str(log_path), "--format", "json"],
    )
    # exit code 1 = найдены issues, 2 = ошибка
    assert result.exit_code in (1, 2)
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["exit_code"] == result.exit_code
    if result.exit_code == 1:
        assert record["worst_severity"] == "critical"
        assert record["issues_total"] >= 1

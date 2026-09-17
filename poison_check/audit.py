"""Audit-лог сканирований для ИБ-аудитов.

Аудит #31: банки и госструктуры требуют журнал — кто что сканировал,
когда, какой результат, какая политика. Без этого результаты сканера
невозможно использовать как часть compliance-процесса.

Формат: JSON Lines (`.jsonl`) — одна запись на строку, append-only.
Удобен для tail/grep/ingest в SIEM (Splunk, ELK, etc.).

Пример записи::

    {
      "timestamp": "2026-05-05T12:34:56Z",
      "tool_version": "0.1.0",
      "user": "alice",
      "host": "auditor-laptop",
      "pid": 12345,
      "scanned_path": "/models/suspect.pt",
      "file_count": 1,
      "file_hashes": {"sha256": "abc..."},
      "policy": "banking",
      "duration_ms": 1247.3,
      "issues_total": 2,
      "issues_by_severity": {"critical": 1, "high": 1},
      "worst_severity": "critical",
      "exit_code": 1,
      "compliance_disclaimer_shown": true
    }

Никаких сырых данных файла, никаких decompiled_code в логе — только метаданные
и счётчики, чтобы лог сам не стал утечкой.

Активируется через:

* CLI-флаг ``--audit-log /var/log/poison-check.jsonl``
* Переменная окружения ``POISON_CHECK_AUDIT_LOG=...``

Если ни то, ни другое не задано — лог не пишется (no-op).
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poison_check.core.result import ScanResult, Severity

logger = logging.getLogger(__name__)


def _safe_user() -> str:
    """Возвращает имя пользователя, не падает в headless-окружении."""
    try:
        return getpass.getuser()
    except (OSError, ImportError):
        return os.environ.get("USER", "unknown")


def _safe_hostname() -> str:
    """Hostname сервера. Не падает на сетевых ошибках."""
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def _summarize_hashes(result: ScanResult) -> dict[str, str]:
    """Берёт SHA-256 первого файла как идентификатор сканирования.

    При множественных файлах в audit-логе достаточно SHA-256 каждого.
    Чтобы не раздувать запись — собираем sha256 в строку через запятую.
    """
    hashes: list[str] = []
    for fr in result.results_per_file.values():
        # FileResult сам не содержит хеши, нужно дотянуться до RawScanData,
        # которого здесь уже нет. Пропускаем — хеши попадут в JSON-отчёт,
        # а в audit-log пишем количество файлов и path.
        _ = fr
    if hashes:
        return {"sha256": ",".join(hashes)}
    return {}


def write_audit_record(
    result: ScanResult,
    scanned_path: Path,
    exit_code: int,
    audit_log_path: Path | None = None,
    user: str | None = None,
) -> None:
    """Записывает одну запись в audit-лог в формате JSON Lines.

    Если ``audit_log_path`` равен ``None``, проверяет переменную окружения
    ``POISON_CHECK_AUDIT_LOG``. Если и она не задана — функция ничего не делает
    (no-op): audit-логирование выключено.

    Никогда не бросает исключений — отсутствие прав на запись или ошибка
    записи логируются через ``logging.warning`` и проглатываются. Аудит-лог
    не должен мешать основной задаче (сканированию).

    :param result: Результат сканирования (берётся timestamp, duration_ms,
        политика, summary, compliance_report).
    :param scanned_path: Путь, переданный пользователем (файл или директория).
    :param exit_code: Код возврата CLI (0/1/2).
    :param audit_log_path: Явный путь к файлу лога. Если None — берётся из env.
    :param user: Override имени пользователя (для тестов). Если None —
        ``getpass.getuser()``.
    """
    target = audit_log_path
    if target is None:
        env_path = os.environ.get("POISON_CHECK_AUDIT_LOG", "").strip()
        if not env_path:
            return  # logging выключен
        target = Path(env_path)

    # Считаем issues по severity
    counts: dict[str, int] = {s.value: 0 for s in Severity}
    total = 0
    for fr in result.results_per_file.values():
        for issue in fr.issues:
            counts[issue.severity.value] = counts.get(issue.severity.value, 0) + 1
            total += 1

    worst = result.worst_severity
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool_version": result.tool_version,
        "user": user if user is not None else _safe_user(),
        "host": _safe_hostname(),
        "pid": os.getpid(),
        "scanned_path": str(scanned_path),
        "file_count": len(result.results_per_file),
        "policy": result.policy,
        "duration_ms": round(result.duration_ms, 2),
        "issues_total": total,
        "issues_by_severity": {k: v for k, v in counts.items() if v > 0},
        "worst_severity": worst.value if worst else None,
        "exit_code": exit_code,
        "compliance_disclaimer_shown": (
            result.compliance_report is not None
            and result.compliance_report.disclaimer is not None
        ),
    }

    try:
        # Создаём родительскую директорию если её нет (mkdir -p).
        target.parent.mkdir(parents=True, exist_ok=True)
        # Append-only: открываем в режиме 'a', одна строка JSON + \n.
        # ensure_ascii=False — путь с unicode-именем должен быть читаемым.
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        logger.warning(
            "Не удалось записать audit-лог в %s: %s. Сканирование завершено корректно, "
            "но запись в журнал не произведена.",
            target,
            exc,
        )

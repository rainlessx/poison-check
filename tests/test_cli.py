"""Тесты CLI-команд poison-check через typer.testing.CliRunner.

Проверяет три команды: doctor, list-scanners, scan.
Тесты не зависят от сети и внешних сервисов.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from poison_check.cli import app
from poison_check.core.registry import DetectorRegistry, ScannerRegistry
from poison_check.i18n.loader import I18n

_FIXTURES = Path(__file__).parent / "fixtures"
_SAFE = _FIXTURES / "safe"
_MALICIOUS = _FIXTURES / "malicious"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_i18n() -> None:
    """Сбрасывает singleton I18n перед каждым тестом."""
    I18n.reset()


# ---------------------------------------------------------------------------
# Вспомогательная функция
# ---------------------------------------------------------------------------


def _registries_populated() -> bool:
    """Проверяет, что реестры не пусты после вызова CLI."""
    return bool(ScannerRegistry.all_scanners()) and bool(DetectorRegistry.all_detectors())


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_exit_code_zero() -> None:
    """poison-check doctor завершается с кодом 0 при корректной установке."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, f"Ожидался exit code 0, получен {result.exit_code}.\n{result.output}"


def test_doctor_contains_python() -> None:
    """Вывод poison-check doctor содержит строку 'Python'."""
    result = runner.invoke(app, ["doctor"])
    assert "Python" in result.output, f"'Python' не найден в выводе:\n{result.output}"


def test_doctor_contains_poison_check_version() -> None:
    """Вывод poison-check doctor содержит версию poison-check."""
    result = runner.invoke(app, ["doctor"])
    assert "poison-check" in result.output.lower()
    assert "0.1.0" in result.output


def test_doctor_shows_registered_scanners() -> None:
    """doctor показывает список зарегистрированных сканеров."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    # После вызова CLI реестры должны быть заполнены
    assert _registries_populated(), "Реестры не заполнены после запуска CLI"


# ---------------------------------------------------------------------------
# list-scanners
# ---------------------------------------------------------------------------


def test_list_scanners_contains_pickle() -> None:
    """poison-check list-scanners выводит 'pickle' в таблице."""
    result = runner.invoke(app, ["list-scanners"])
    assert result.exit_code == 0, f"exit code: {result.exit_code}\n{result.output}"
    assert "pickle" in result.output.lower(), (
        f"'pickle' не найден в выводе list-scanners:\n{result.output}"
    )


def test_list_scanners_shows_extensions() -> None:
    """list-scanners выводит расширения файлов."""
    result = runner.invoke(app, ["list-scanners"])
    assert result.exit_code == 0
    # PickleScanner поддерживает .pkl
    assert ".pkl" in result.output


def test_list_scanners_table_structure() -> None:
    """list-scanners выводит таблицу с форматами."""
    result = runner.invoke(app, ["list-scanners"])
    assert result.exit_code == 0
    # Таблица должна содержать заголовки (Сканер / Расширения / Описание)
    output_lower = result.output.lower()
    assert "сканер" in output_lower or "scanner" in output_lower or "pickle" in output_lower


# ---------------------------------------------------------------------------
# scan — безопасный файл
# ---------------------------------------------------------------------------


def test_scan_safe_pkl_exit_code_zero() -> None:
    """poison-check scan safe/simple_list.pkl → exit code 0 (нет угроз)."""
    safe_file = _SAFE / "simple_list.pkl"
    assert safe_file.exists(), f"Тестовый файл не найден: {safe_file}"

    result = runner.invoke(app, ["scan", str(safe_file)])
    assert result.exit_code == 0, (
        f"Ожидался exit code 0 для безопасного файла, получен {result.exit_code}.\n{result.output}"
    )


def test_scan_safe_shows_no_issues() -> None:
    """Вывод для безопасного файла содержит сообщение об отсутствии угроз."""
    safe_file = _SAFE / "simple_list.pkl"
    result = runner.invoke(app, ["scan", str(safe_file)])
    assert result.exit_code == 0
    # Должно быть сообщение "нет угроз" или аналог
    # ConsoleFormatter выводит "✅ Угроз не обнаружено"
    output_lower = result.output.lower()
    assert any(
        marker in output_lower
        for marker in ["угроз не обнаружено", "no_issues", "✅", "нет", "clean"]
    ), f"Не найдено сообщение об отсутствии угроз:\n{result.output}"


# ---------------------------------------------------------------------------
# scan — вредоносный файл
# ---------------------------------------------------------------------------


def test_scan_malicious_pkl_exit_code_one() -> None:
    """poison-check scan malicious/os_system.pkl → exit code 1 (найдены угрозы)."""
    malicious_file = _MALICIOUS / "os_system.pkl"
    assert malicious_file.exists(), f"Тестовый файл не найден: {malicious_file}"

    result = runner.invoke(app, ["scan", str(malicious_file)])
    assert result.exit_code == 1, (
        f"Ожидался exit code 1 для вредоносного файла, получен {result.exit_code}.\n{result.output}"
    )


def test_scan_malicious_shows_issue() -> None:
    """Вывод для вредоносного файла содержит информацию об угрозе."""
    malicious_file = _MALICIOUS / "os_system.pkl"
    result = runner.invoke(app, ["scan", str(malicious_file)])
    assert result.exit_code == 1
    # Должно быть упоминание os.system или CRITICAL/КРИТИЧНО
    output_lower = result.output.lower()
    assert any(
        marker in output_lower
        for marker in ["critical", "критично", "os", "system", "mls001", "mls-"]
    ), f"Не найдено сообщение об угрозе:\n{result.output}"


# ---------------------------------------------------------------------------
# scan --format json
# ---------------------------------------------------------------------------


def test_scan_malicious_json_format_valid() -> None:
    """poison-check scan malicious/os_system.pkl --format json → валидный JSON."""
    malicious_file = _MALICIOUS / "os_system.pkl"
    result = runner.invoke(app, ["scan", str(malicious_file), "--format", "json"])

    # JSON-вывод может идти в stdout независимо от exit code
    assert result.exit_code in (0, 1), (
        f"Неожиданный exit code {result.exit_code}.\n{result.output}"
    )

    # Парсим JSON из вывода
    try:
        parsed = json.loads(result.output)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Вывод не является валидным JSON: {exc}\nВывод:\n{result.output}")

    # Проверяем обязательные поля схемы v1.0
    assert parsed.get("schema_version") == "1.0", "Отсутствует или неверное поле schema_version"
    assert "tool" in parsed, "Отсутствует поле tool"
    assert "scan_info" in parsed, "Отсутствует поле scan_info"
    assert "summary" in parsed, "Отсутствует поле summary"
    assert "results" in parsed, "Отсутствует поле results"


def test_scan_json_schema_fields() -> None:
    """JSON-отчёт содержит все поля схемы v1.0."""
    safe_file = _SAFE / "simple_list.pkl"
    result = runner.invoke(app, ["scan", str(safe_file), "--format", "json"])

    parsed = json.loads(result.output)

    # Проверяем tool
    tool = parsed["tool"]
    assert tool["name"] == "poison-check"
    assert "version" in tool
    assert "policy" in tool

    # Проверяем scan_info
    scan_info = parsed["scan_info"]
    assert "timestamp" in scan_info
    assert "duration_ms" in scan_info
    assert "files_scanned" in scan_info

    # Проверяем summary
    summary = parsed["summary"]
    for key in ("critical", "high", "medium", "low", "info"):
        assert key in summary, f"Отсутствует поле summary.{key}"

    # Проверяем results
    assert isinstance(parsed["results"], list)
    assert len(parsed["results"]) >= 1

    # Проверяем структуру одного файлового результата
    file_res = parsed["results"][0]
    assert "file" in file_res
    assert "path" in file_res["file"]
    assert "scanner" in file_res
    assert "issues" in file_res


def test_scan_json_issues_have_required_fields() -> None:
    """Каждый issue в JSON содержит обязательные поля."""
    malicious_file = _MALICIOUS / "os_system.pkl"
    result = runner.invoke(app, ["scan", str(malicious_file), "--format", "json"])

    parsed = json.loads(result.output)
    results = parsed.get("results", [])
    assert results, "Нет результатов в JSON"

    all_issues = [issue for r in results for issue in r.get("issues", [])]
    assert all_issues, "Нет issues в JSON для вредоносного файла"

    required_issue_fields = {"code", "severity", "confidence", "message", "location"}
    for issue in all_issues:
        missing = required_issue_fields - set(issue.keys())
        assert not missing, f"Issue отсутствуют поля: {missing}. Issue: {issue}"


def test_scan_json_severity_values_valid() -> None:
    """Значения severity в JSON соответствуют допустимым значениям Enum."""
    malicious_file = _MALICIOUS / "os_system.pkl"
    result = runner.invoke(app, ["scan", str(malicious_file), "--format", "json"])
    parsed = json.loads(result.output)

    valid_severities = {"critical", "high", "medium", "low", "info"}
    for file_res in parsed.get("results", []):
        for issue in file_res.get("issues", []):
            sev = issue.get("severity")
            assert sev in valid_severities, f"Недопустимое значение severity: {sev}"


# ---------------------------------------------------------------------------
# scan — несуществующий файл → exit code 2
# ---------------------------------------------------------------------------


def test_scan_nonexistent_file_exit_code_two() -> None:
    """poison-check scan несуществующий.pkl → exit code 2."""
    result = runner.invoke(app, ["scan", "несуществующий_файл.pkl"])
    assert result.exit_code == 2, (
        f"Ожидался exit code 2 для несуществующего файла, получен {result.exit_code}.\n{result.output}"
    )


def test_scan_nonexistent_file_error_message() -> None:
    """Для несуществующего файла выводится сообщение об ошибке."""
    result = runner.invoke(app, ["scan", "несуществующий_файл.pkl"])
    assert result.exit_code == 2
    output_lower = result.output.lower()
    assert any(
        marker in output_lower
        for marker in ["ошибка", "не найден", "error", "not found", "несуществующий"]
    ), f"Сообщение об ошибке не найдено в выводе:\n{result.output}"


# ---------------------------------------------------------------------------
# Дополнительные сценарии
# ---------------------------------------------------------------------------


def test_scan_save_json_to_file(tmp_path: Path) -> None:
    """--output сохраняет JSON-отчёт в файл."""
    safe_file = _SAFE / "simple_list.pkl"
    output_file = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["scan", str(safe_file), "--format", "json", "--output", str(output_file)],
    )
    # exit code 0 для безопасного файла
    assert result.exit_code == 0, f"exit code: {result.exit_code}\n{result.output}"
    assert output_file.exists(), "Файл отчёта не создан"

    # Файл должен содержать валидный JSON
    parsed = json.loads(output_file.read_text(encoding="utf-8"))
    assert parsed.get("schema_version") == "1.0"


def test_scan_payload_01_os_system_detected() -> None:
    """payload_01_os_system.pkl детектируется как угроза."""
    malicious_file = _MALICIOUS / "payload_01_os_system.pkl"
    result = runner.invoke(app, ["scan", str(malicious_file)])
    assert result.exit_code in (1, 2), (
        f"Ожидался exit code 1 или 2, получен {result.exit_code}\n{result.output}"
    )


# ---------------------------------------------------------------------------
# Регрессия аудита #5: фильтрация symlink / FIFO / специальных файлов
# ---------------------------------------------------------------------------


def test_collect_paths_skips_symlink_at_root(tmp_path: Path) -> None:
    """_collect_paths не следует за корневым symlink."""
    from poison_check.cli import _collect_paths

    target = tmp_path / "real.pkl"
    target.write_bytes(b"\x80\x02.")
    link = tmp_path / "link.pkl"
    link.symlink_to(target)

    assert _collect_paths(link, recursive=False) == []


def test_collect_paths_skips_symlink_in_directory(tmp_path: Path) -> None:
    """_collect_paths рекурсивно отбрасывает symlink-файлы."""
    from poison_check.cli import _collect_paths

    real = tmp_path / "real.pkl"
    real.write_bytes(b"\x80\x02.")
    link = tmp_path / "link.pkl"
    link.symlink_to(real)

    collected = _collect_paths(tmp_path, recursive=True)
    assert real in collected
    assert link not in collected, "Symlink не должен попадать в список"


def test_collect_paths_skips_fifo(tmp_path: Path) -> None:
    """_collect_paths отбрасывает FIFO (named pipe) — DoS-вектор."""
    import os

    fifo = tmp_path / "pipe"
    try:
        os.mkfifo(fifo)
    except (OSError, AttributeError):
        pytest.skip("Платформа не поддерживает mkfifo")

    from poison_check.cli import _collect_paths

    collected = _collect_paths(tmp_path, recursive=False)
    assert fifo not in collected, "FIFO не должен попадать в список"


def test_collect_paths_includes_regular_file(tmp_path: Path) -> None:
    """Обычный файл корректно попадает в список (positive test)."""
    from poison_check.cli import _collect_paths

    f = tmp_path / "model.pkl"
    f.write_bytes(b"\x80\x02.")
    assert _collect_paths(f, recursive=False) == [f]


# ---------------------------------------------------------------------------
# scan — лимит размера файла из политики (extra_rules.max_file_size_gb)
# ---------------------------------------------------------------------------


def _write_pickle(path: Path, payload_size: int) -> Path:
    """Собирает безопасный pickle заданного размера ручной opcode-конструкцией.

    Байты формируются вручную (протокол 4, BINUNICODE со строкой из 'A',
    затем STOP) — без pickle.dumps, как требует политика проекта по фикстурам.
    """
    payload = b"A" * payload_size
    data = (
        b"\x80\x04"                                   # PROTO 4
        + b"X" + len(payload).to_bytes(4, "little")   # BINUNICODE + длина
        + payload
        + b"."                                        # STOP
    )
    path.write_bytes(data)
    return path


def _write_policy(path: Path, max_file_size_gb: float) -> Path:
    """Пишет минимальную политику с extra_rules.max_file_size_gb.

    Значение форматируется без экспоненты: YAML 1.1 разбирает «1e-06»
    как строку, а не как число.
    """
    path.write_text(
        "name: tiny_limit\n"
        "description: Тестовая политика с крошечным лимитом размера\n"
        "fail_on_severity: critical\n"
        "compliance: []\n"
        "extra_rules:\n"
        f"  max_file_size_gb: {max_file_size_gb:.12f}\n",
        encoding="utf-8",
    )
    return path


def _scan_json(*args: str) -> dict:
    """Запускает scan --format json и возвращает разобранный отчёт."""
    result = runner.invoke(app, ["scan", *args, "--format", "json"])
    try:
        return json.loads(result.output)
    except json.JSONDecodeError as exc:  # pragma: no cover — диагностика
        pytest.fail(f"Вывод не JSON: {exc}\n{result.output}")


def test_scan_policy_max_file_size_blocks_large_file(tmp_path: Path) -> None:
    """extra_rules.max_file_size_gb применяется: файл сверх лимита → error."""
    model = _write_pickle(tmp_path / "big.pkl", payload_size=4096)
    policy = _write_policy(tmp_path / "tiny.yaml", max_file_size_gb=0.000001)  # 1000 байт

    parsed = _scan_json(str(model), "--policy", str(policy))

    error = parsed["results"][0]["error"]
    assert error is not None, "Ожидалась ошибка про превышение лимита размера"
    assert "слишком большой" in error, f"Неожиданный текст ошибки: {error}"


def test_scan_policy_max_file_size_exit_code_two(tmp_path: Path) -> None:
    """Файл сверх лимита политики → exit code 2 (ошибка сканирования)."""
    model = _write_pickle(tmp_path / "big.pkl", payload_size=4096)
    policy = _write_policy(tmp_path / "tiny.yaml", max_file_size_gb=0.000001)

    result = runner.invoke(app, ["scan", str(model), "--policy", str(policy)])
    assert result.exit_code == 2, f"Ожидался exit code 2, получен {result.exit_code}"


def test_scan_cli_flag_overrides_policy_limit(tmp_path: Path) -> None:
    """--max-file-size имеет приоритет над extra_rules.max_file_size_gb."""
    model = _write_pickle(tmp_path / "big.pkl", payload_size=4096)
    policy = _write_policy(tmp_path / "tiny.yaml", max_file_size_gb=0.000001)

    parsed = _scan_json(
        str(model), "--policy", str(policy), "--max-file-size", "1"
    )

    assert parsed["results"][0]["error"] is None, (
        "CLI-флаг должен переопределять лимит политики"
    )


def test_scan_without_policy_limit_scans_file(tmp_path: Path) -> None:
    """Политика без extra_rules → работает дефолтный лимит сканера (10 ГБ)."""
    model = _write_pickle(tmp_path / "big.pkl", payload_size=4096)

    parsed = _scan_json(str(model), "--policy", "default")

    assert parsed["results"][0]["error"] is None


def test_scan_policy_limit_allows_small_file(tmp_path: Path) -> None:
    """Файл в пределах лимита политики сканируется без ошибок."""
    model = _write_pickle(tmp_path / "small.pkl", payload_size=16)
    policy = _write_policy(tmp_path / "tiny.yaml", max_file_size_gb=0.000001)

    parsed = _scan_json(str(model), "--policy", str(policy))

    assert parsed["results"][0]["error"] is None


def test_scan_banking_policy_escalates_external_url(tmp_path: Path) -> None:
    """--policy banking: неизвестный домен в модели → HIGH (MLS-NET-002).

    Через default та же находка остаётся MEDIUM — регрессия задачи 3.
    """
    payload = "https://attacker.example/exfil".encode()
    data = (
        b"\x80\x04"
        + b"X" + len(payload).to_bytes(4, "little")
        + payload
        + b"."
    )
    model = tmp_path / "c2.pkl"
    model.write_bytes(data)

    def _net_severity(policy_name: str) -> str:
        parsed = _scan_json(str(model), "--policy", policy_name)
        issues = [
            i
            for r in parsed["results"]
            for i in r["issues"]
            if i["code"] == "MLS-NET-002"
        ]
        assert len(issues) == 1, f"Ожидался один MLS-NET-002, получено: {issues}"
        return str(issues[0]["severity"])

    assert _net_severity("banking") == "high"
    assert _net_severity("default") == "medium"

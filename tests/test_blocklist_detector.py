"""Тесты для BlocklistDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.core.result import Confidence, Issue, MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.blocklist_detector import BlocklistDetector

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

_DUMMY_PATH = Path("test.pkl")


def _make_raw(
    globals_set: set[tuple[str, str]] | None = None,
    file_path: Path = _DUMMY_PATH,
) -> RawScanData:
    """Минимальный RawScanData для тестов BlocklistDetector."""
    return RawScanData(
        file_path=file_path,
        file_hash={},
        file_size=100,
        scanner_name="pickle",
        globals=globals_set,
    )


@pytest.fixture
def context() -> MLContext:
    """Нейтральный MLContext для передачи в analyze()."""
    return MLContext(framework="unknown", confidence=1.0)


@pytest.fixture
def detector() -> BlocklistDetector:
    """BlocklistDetector с дефолтным YAML (rules/cve/ml_cves.yaml)."""
    return BlocklistDetector()


# ---------------------------------------------------------------------------
# Тест 1: os.system в globals → Issue с severity=CRITICAL
# ---------------------------------------------------------------------------


def test_os_system_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """os.system в globals → Issue с severity=CRITICAL."""
    raw = _make_raw(globals_set={("os", "system")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == Severity.CRITICAL
    assert "os" in issue.details["module"]
    assert "system" in issue.details["name"]


def test_subprocess_popen_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """subprocess.Popen в globals → Issue с severity=CRITICAL."""
    raw = _make_raw(globals_set={("subprocess", "Popen")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL


def test_builtins_eval_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """builtins.eval в globals → Issue с severity=CRITICAL."""
    raw = _make_raw(globals_set={("builtins", "eval")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].confidence == Confidence.CERTAIN


# ---------------------------------------------------------------------------
# Тест 2: чистые globals → пустой список issues
# ---------------------------------------------------------------------------


def test_clean_pytorch_globals_produce_no_issues(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """Стандартные globals легитимной PyTorch-модели → пустой список."""
    clean_globals: set[tuple[str, str]] = {
        ("torch", "Tensor"),
        ("torch", "FloatTensor"),
        ("torch", "LongTensor"),
        ("collections", "OrderedDict"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }
    raw = _make_raw(globals_set=clean_globals)
    issues = detector.analyze(raw, context)

    assert issues == []


def test_none_globals_produce_no_issues(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """globals=None (файл без pickle-globals) → пустой список."""
    raw = _make_raw(globals_set=None)
    issues = detector.analyze(raw, context)

    assert issues == []


def test_empty_globals_set_produce_no_issues(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """Пустое множество globals → пустой список."""
    raw = _make_raw(globals_set=set())
    issues = detector.analyze(raw, context)

    assert issues == []


# ---------------------------------------------------------------------------
# Тест 3: CVE-паттерны из YAML в BlocklistDetector НЕ обрабатываются
# ---------------------------------------------------------------------------
# Ранее BlocklistDetector дублировал функциональность CVEDetector: оба читали
# ml_cves.yaml, оба эмитили Issue на одни и те же (module, name). В выводе
# появлялись пары "PATTERN-OS-SYSTEM" (raw id из YAML) и "MLS-PATTERN-OS-SYSTEM"
# (из CVEDetector), сбивая с толку заказчика. YAML теперь целиком в CVEDetector.


def test_yaml_only_pattern_not_detected_by_blocklist(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """Глобал, покрытый только YAML-паттерном (не hardcoded) → 0 Issue.

    ``dill._dill._create_code`` есть в PATTERN-DILL-BYPASS, но НЕ в hardcoded
    blocklist. BlocklistDetector его не ловит — этим занимается CVEDetector.
    """
    raw = _make_raw(globals_set={("dill._dill", "_create_code")})
    issues = detector.analyze(raw, context)
    assert issues == []


# ---------------------------------------------------------------------------
# Тест 4: graceful degradation при отсутствии YAML
# ---------------------------------------------------------------------------


def test_yaml_not_found_no_exception(tmp_path: Path, context: MLContext) -> None:
    """BlocklistDetector не падает если YAML файл не найден."""
    non_existent = tmp_path / "missing.yaml"
    detector = BlocklistDetector(rules_path=non_existent)

    raw = _make_raw(globals_set={("os", "system")})
    issues = detector.analyze(raw, context)

    # Hardcoded blocklist всё равно работает
    assert len(issues) == 1
    assert issues[0].code == "MLS-PKL-001"
    assert issues[0].severity == Severity.CRITICAL


def test_yaml_not_found_clean_globals_still_empty(
    tmp_path: Path, context: MLContext
) -> None:
    """При отсутствии YAML чистые globals дают пустой список."""
    non_existent = tmp_path / "missing.yaml"
    detector = BlocklistDetector(rules_path=non_existent)

    raw = _make_raw(globals_set={("torch", "Tensor"), ("numpy", "ndarray")})
    issues = detector.analyze(raw, context)

    assert issues == []


def test_yaml_invalid_content_graceful(tmp_path: Path, context: MLContext) -> None:
    """BlocklistDetector не падает при некорректном YAML (не список)."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("this is not a list: just a string", encoding="utf-8")

    detector = BlocklistDetector(rules_path=bad_yaml)
    raw = _make_raw(globals_set={("os", "system")})
    issues = detector.analyze(raw, context)

    # Graceful degradation: hardcoded blocklist работает
    assert len(issues) == 1
    assert issues[0].code == "MLS-PKL-001"


# ---------------------------------------------------------------------------
# Тест 5: множественные hardcoded глобалы, отсутствие дубликатов
# ---------------------------------------------------------------------------


def test_multiple_dangerous_globals_produce_multiple_issues(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """Несколько опасных globals → отдельный Issue для каждого."""
    raw = _make_raw(
        globals_set={
            ("os", "system"),
            ("builtins", "exec"),
        }
    )
    issues = detector.analyze(raw, context)

    assert len(issues) == 2
    for issue in issues:
        assert issue.code == "MLS-PKL-001"
        assert issue.severity == Severity.CRITICAL


def test_no_duplicate_issues_per_global(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """Один опасный глобал → ровно один Issue, не два.

    Регрессионный тест: раньше BlocklistDetector читал YAML И hardcoded,
    поэтому один ``os.system`` порождал ДВА Issue с кодами
    ``PATTERN-OS-SYSTEM`` и ``MLS-PKL-001``. Теперь YAML в CVEDetector,
    BlocklistDetector — только hardcoded, ровно один Issue.
    """
    raw = _make_raw(globals_set={("os", "system")})
    issues = detector.analyze(raw, context)
    assert len(issues) == 1
    assert issues[0].code == "MLS-PKL-001"


def test_blocklist_only_global_gets_mls001(
    tmp_path: Path, context: MLContext
) -> None:
    """Глобал только в hardcoded blocklist (без YAML) → code=MLS001."""
    non_existent = tmp_path / "no.yaml"
    detector = BlocklistDetector(rules_path=non_existent)

    # importlib.import_module — в hardcoded blocklist, но не добавлен в тестовый YAML
    raw = _make_raw(globals_set={("importlib", "import_module")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].code == "MLS-PKL-001"
    assert issues[0].severity == Severity.CRITICAL
    cwe_refs = [r for r in issues[0].references if r.type == "cwe"]
    assert cwe_refs


def test_location_with_reduce_call() -> None:
    """Если есть matching reduce_call — location содержит offset."""
    from poison_check.core.result import ReduceCall

    detector = BlocklistDetector()
    context = MLContext(framework="unknown", confidence=1.0)
    raw = RawScanData(
        file_path=Path("model.pkl"),
        file_hash={},
        file_size=200,
        scanner_name="pickle",
        globals={("os", "system")},
        reduce_calls=[ReduceCall(module="os", name="system", position=42)],
    )
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert "offset 42" in issues[0].location


# ---------------------------------------------------------------------------
# Тесты на новые globals — POSIX / Windows os.system через нативные модули
# ---------------------------------------------------------------------------


def test_posix_system_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """posix.system (Linux-эквивалент os.system) → CRITICAL issue."""
    raw = _make_raw(globals_set={("posix", "system")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == Severity.CRITICAL
    assert issue.details["module"] == "posix"
    assert issue.details["name"] == "system"


def test_nt_system_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """nt.system (Windows-эквивалент os.system) → CRITICAL issue."""
    raw = _make_raw(globals_set={("nt", "system")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == Severity.CRITICAL
    assert issue.details["module"] == "nt"
    assert issue.details["name"] == "system"


# ---------------------------------------------------------------------------
# Тесты на os.exec-семейство
# ---------------------------------------------------------------------------


def test_os_execv_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """os.execv → CRITICAL issue (замена процесса произвольным исполняемым)."""
    raw = _make_raw(globals_set={("os", "execv")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "execv"


def test_os_execve_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """os.execve → CRITICAL issue."""
    raw = _make_raw(globals_set={("os", "execve")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "execve"


def test_os_execvp_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """os.execvp → CRITICAL issue."""
    raw = _make_raw(globals_set={("os", "execvp")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "execvp"


# ---------------------------------------------------------------------------
# Тесты на os.spawn-семейство
# ---------------------------------------------------------------------------


def test_os_spawnl_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """os.spawnl → CRITICAL issue."""
    raw = _make_raw(globals_set={("os", "spawnl")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "spawnl"


def test_os_spawnve_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """os.spawnve → CRITICAL issue."""
    raw = _make_raw(globals_set={("os", "spawnve")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "spawnve"


# ---------------------------------------------------------------------------
# Тест: псевдотерминал
# ---------------------------------------------------------------------------


def test_pty_spawn_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """pty.spawn → CRITICAL issue (RCE через псевдотерминал)."""
    raw = _make_raw(globals_set={("pty", "spawn")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["module"] == "pty"


# ---------------------------------------------------------------------------
# Тесты на вторичную десериализацию (pickle-in-pickle)
# ---------------------------------------------------------------------------


def test_marshal_loads_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """marshal.loads → CRITICAL issue (RCE через marshal внутри pickle)."""
    raw = _make_raw(globals_set={("marshal", "loads")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["module"] == "marshal"
    assert issues[0].details["name"] == "loads"


def test_pickle_loads_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """pickle.loads → CRITICAL issue (pickle внутри pickle)."""
    raw = _make_raw(globals_set={("pickle", "loads")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["module"] == "pickle"
    assert issues[0].details["name"] == "loads"


def test_dill_loads_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """dill.loads → CRITICAL issue (dill внутри pickle)."""
    raw = _make_raw(globals_set={("dill", "loads")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["module"] == "dill"
    assert issues[0].details["name"] == "loads"


# ---------------------------------------------------------------------------
# Тесты на сетевые подключения
# ---------------------------------------------------------------------------


def test_socket_socket_produces_high_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """socket.socket → HIGH issue (сетевое подключение — C2 или data exfiltration)."""
    raw = _make_raw(globals_set={("socket", "socket")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    issue = issues[0]
    # socket.socket — HIGH (не CRITICAL: не прямое RCE, но серьёзная угроза)
    assert issue.severity == Severity.HIGH
    assert issue.details["module"] == "socket"
    assert issue.details["name"] == "socket"


# ---------------------------------------------------------------------------
# Тесты на subprocess расширения
# ---------------------------------------------------------------------------


def test_subprocess_check_output_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """subprocess.check_output → CRITICAL issue."""
    raw = _make_raw(globals_set={("subprocess", "check_output")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "check_output"


def test_subprocess_check_call_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """subprocess.check_call → CRITICAL issue."""
    raw = _make_raw(globals_set={("subprocess", "check_call")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "check_call"


def test_subprocess_run_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """subprocess.run → CRITICAL issue."""
    raw = _make_raw(globals_set={("subprocess", "run")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "run"


# ---------------------------------------------------------------------------
# Тест: blocklist_issue использует severity из _BlocklistRule
# ---------------------------------------------------------------------------


def test_blocklist_rule_severity_used_in_issue(
    tmp_path: Path, context: MLContext
) -> None:
    """socket.socket получает HIGH (из _BlocklistRule), а не CRITICAL (дефолт)."""
    non_existent = tmp_path / "no.yaml"
    detector = BlocklistDetector(rules_path=non_existent)

    raw = _make_raw(globals_set={("socket", "socket")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# Тест: known_dangerous.py синхронизирован с _HARDCODED_BLOCKLIST
# ---------------------------------------------------------------------------


def test_cve_detector_triggers_on_inst_opcode() -> None:
    """Регрессия аудита #14: CVEDetector срабатывает на INST opcode (Python 2 legacy).

    До фикса все CVE-правила имели trigger_opcodes = [REDUCE, GLOBAL, STACK_GLOBAL].
    Pickle proto 0/1 могут использовать только INST для вызова callable; такие
    payload обходили CVEDetector. После фикса в trigger_opcodes добавлены
    INST, NEWOBJ, NEWOBJ_EX.
    """
    from poison_check.core.result import OpcodeInfo
    from poison_check.detectors.cve_detector import CVEDetector

    cve = CVEDetector()
    raw = RawScanData(
        file_path=_DUMMY_PATH,
        file_hash={},
        file_size=42,
        scanner_name="pickle",
        opcodes=[OpcodeInfo(position=0, opcode="INST", arg="os system")],
        globals={("os", "system")},
    )
    ctx = MLContext(framework="unknown", confidence=0.5)
    issues = cve.analyze(raw, ctx)
    # Должна сработать как минимум PATTERN-OS-SYSTEM
    pattern_codes = {i.code for i in issues}
    assert any("OS-SYSTEM" in c for c in pattern_codes), (
        f"CVEDetector не сработал на INST + os.system: {pattern_codes}"
    )


def test_cve_detector_triggers_on_newobj_opcode() -> None:
    """Регрессия аудита #14: NEWOBJ как trigger opcode."""
    from poison_check.core.result import OpcodeInfo
    from poison_check.detectors.cve_detector import CVEDetector

    cve = CVEDetector()
    raw = RawScanData(
        file_path=_DUMMY_PATH,
        file_hash={},
        file_size=42,
        scanner_name="pickle",
        opcodes=[OpcodeInfo(position=0, opcode="NEWOBJ", arg=None)],
        globals={("subprocess", "Popen")},
    )
    ctx = MLContext(framework="unknown", confidence=0.5)
    issues = cve.analyze(raw, ctx)
    assert any("SUBPROCESS" in i.code for i in issues), (
        f"CVEDetector не сработал на NEWOBJ + subprocess.Popen: "
        f"{[i.code for i in issues]}"
    )


def test_known_dangerous_globals_matches_hardcoded_blocklist() -> None:
    """KNOWN_DANGEROUS_GLOBALS точно совпадает с BlocklistDetector._HARDCODED_BLOCKLIST."""
    from poison_check.core.known_dangerous import KNOWN_DANGEROUS_GLOBALS

    assert BlocklistDetector._HARDCODED_BLOCKLIST == KNOWN_DANGEROUS_GLOBALS


# ---------------------------------------------------------------------------
# Задача 4: расширенный blocklist — Py2 commands, exec/spawn-семейство,
#           shell-обёртки subprocess, интерактивный интерпретатор
# ---------------------------------------------------------------------------
# Все pickle-фикстуры ниже собираются ВРУЧНУЮ из опкодов (GLOBAL + аргумент +
# REDUCE + STOP). pickle.dumps на злонамеренном объекте не используется —
# см. CLAUDE.md: вредоносные фикстуры только opcode-конструированием.


def _build_global_reduce_pickle(module: str, name: str, arg: str) -> bytes:
    """Собирает pickle proto 4: ``module.name(arg)`` через GLOBAL + REDUCE.

    Байты формируются вручную:
      ``\\x80\\x04`` PROTO 4, ``c<module>\\n<name>\\n`` GLOBAL,
      ``\\x8c<len><arg>`` SHORT_BINUNICODE, ``\\x85`` TUPLE1,
      ``R`` REDUCE, ``.`` STOP.
    """
    arg_bytes = arg.encode("utf-8")
    assert len(arg_bytes) < 256, "аргумент не влезает в SHORT_BINUNICODE"
    return (
        b"\x80\x04"
        + b"c" + module.encode("ascii") + b"\n" + name.encode("ascii") + b"\n"
        + b"\x8c" + bytes([len(arg_bytes)]) + arg_bytes
        + b"\x85"
        + b"R"
        + b"."
    )


def _scan_issues(path: Path) -> list[Issue]:
    """Полный проход Scanner по файлу → список issues после dedupe."""
    from poison_check.scanner import Scanner

    result = Scanner().scan(path)
    file_result = next(iter(result.results_per_file.values()))
    return list(file_result.issues)


# --- Py2-модуль commands ----------------------------------------------------


def test_py2_commands_getoutput_is_critical_with_raw_name(tmp_path: Path) -> None:
    """Py2 pickle ``commands.getoutput('id')`` → ровно 1 CRITICAL с сырым именем.

    ``commands`` — Python 2 модуль, прямой аналог ``os.system``. Он намеренно
    НЕ добавлен в ``_MODULE_ONLY_ALIASES`` (его ``getstatus``/``mkarg`` не
    имеют Py3-аналога), поэтому пара попадает в blocklist напрямую.

    Форензик-инвариант: в message/details лежит СЫРОЕ ``commands.getoutput``,
    а не канонизированное ``subprocess.getoutput``.
    """
    p = tmp_path / "py2_commands.pkl"
    p.write_bytes(_build_global_reduce_pickle("commands", "getoutput", "id"))

    issues = _scan_issues(p)

    assert len(issues) == 1, (
        f"ожидался 1 issue, получено {len(issues)}: "
        f"{[(i.code, i.severity.value) for i in issues]}"
    )
    issue = issues[0]
    assert issue.severity == Severity.CRITICAL
    assert issue.code == "MLS-PKL-001"
    # Сырое имя в message и details — форензик-инвариант.
    assert "commands.getoutput" in issue.message
    assert issue.details["module"] == "commands"
    assert issue.details["name"] == "getoutput"
    # Канонизации к subprocess не произошло — модуль не алиас.
    assert "subprocess" not in issue.message


def test_commands_getstatusoutput_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """commands.getstatusoutput → CRITICAL issue."""
    raw = _make_raw(globals_set={("commands", "getstatusoutput")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["module"] == "commands"


# --- subprocess: shell-обёртки ---------------------------------------------


@pytest.mark.parametrize("func", ["getoutput", "getstatusoutput"])
def test_subprocess_shell_wrappers_are_critical(
    detector: BlocklistDetector, context: MLContext, func: str
) -> None:
    """subprocess.getoutput / getstatusoutput → CRITICAL, не MEDIUM.

    Обе функции выполняют строку через shell (``/bin/sh -c``), то есть дают
    полноценный RCE, а раньше попадали только в MEDIUM allowlist-miss.
    """
    raw = _make_raw(globals_set={("subprocess", func)})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].code == "MLS-PKL-001"
    assert issues[0].details["name"] == func


# --- os.exec* / os.spawn* / posix_spawn* ------------------------------------


@pytest.mark.parametrize(
    "func",
    [
        "execl", "execle", "execlp", "execvpe",
        "spawnv", "spawnvp", "spawnlp", "spawnvpe",
        "posix_spawn", "posix_spawnp",
    ],
)
def test_os_exec_and_spawn_family_are_critical(
    detector: BlocklistDetector, context: MLContext, func: str
) -> None:
    """Полное exec/spawn-семейство os → CRITICAL issue (MLS-PKL-001).

    Раньше в blocklist были только execv/execve/execvp/spawnl/spawnve —
    остальные варианты той же семантики давали лишь MEDIUM allowlist-miss.
    """
    raw = _make_raw(globals_set={("os", func)})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].code == "MLS-PKL-001"
    assert issues[0].details["name"] == func


def test_code_interact_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """code.interact (интерактивный REPL внутри процесса) → CRITICAL issue."""
    raw = _make_raw(globals_set={("code", "interact")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["module"] == "code"


def test_builtins_breakpoint_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """builtins.breakpoint → CRITICAL issue (подмена hook через PYTHONBREAKPOINT)."""
    raw = _make_raw(globals_set={("builtins", "breakpoint")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["name"] == "breakpoint"


# --- Интеграция: CRITICAL вместо MEDIUM allowlist-miss ----------------------


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("os", "execlp"),
        ("subprocess", "getoutput"),
        ("os", "spawnv"),
    ],
)
def test_new_blocklist_pairs_end_to_end_critical_not_medium(
    tmp_path: Path, module: str, name: str
) -> None:
    """Полный проход Scanner: новые пары дают CRITICAL, а не MEDIUM.

    Регрессия задачи 4: до расширения blocklist эти вызовы проходили мимо
    BlocklistDetector и оседали в AllowlistDetector как MLS-ALW-001 (MEDIUM),
    хотя каждый из них — прямой RCE.
    """
    p = tmp_path / f"{module}_{name}.pkl"
    p.write_bytes(_build_global_reduce_pickle(module, name, "/bin/sh"))

    issues = _scan_issues(p)

    assert len(issues) == 1, (
        f"ожидался 1 issue, получено {len(issues)}: "
        f"{[(i.code, i.severity.value) for i in issues]}"
    )
    issue = issues[0]
    assert issue.severity == Severity.CRITICAL, (
        f"{module}.{name} должен быть CRITICAL, получено "
        f"{issue.severity.value} ({issue.code})"
    )
    assert not issue.code.startswith("MLS-ALW-")
    assert issue.details["module"] == module
    assert issue.details["name"] == name


def test_new_blocklist_pairs_skipped_by_allowlist_detector() -> None:
    """AllowlistDetector не дублирует новые blocklist-пары.

    Инвариант: пара в ``KNOWN_DANGEROUS_GLOBALS`` пропускается allowlist'ом —
    пользователь получает один CRITICAL, а не CRITICAL + MEDIUM.
    """
    from poison_check.detectors.allowlist_detector import AllowlistDetector

    pairs = {
        ("commands", "getoutput"),
        ("subprocess", "getstatusoutput"),
        ("os", "posix_spawnp"),
        ("code", "interact"),
        ("builtins", "breakpoint"),
        ("posix", "popen"),
    }
    raw = _make_raw(globals_set=pairs)
    ctx = MLContext(framework="unknown", confidence=1.0)

    assert AllowlistDetector().analyze(raw, ctx) == []


# --- posix.popen: рассинхрон blocklist ↔ ml_cves.yaml -----------------------


def test_posix_popen_produces_critical_issue(
    detector: BlocklistDetector, context: MLContext
) -> None:
    """posix.popen → CRITICAL issue.

    Пара перечислена в ``PATTERN-OS-SYSTEM`` (rules/cve/ml_cves.yaml), но
    отсутствовала в ``KNOWN_DANGEROUS_GLOBALS`` — детекторы расходились.
    """
    raw = _make_raw(globals_set={("posix", "popen")})
    issues = detector.analyze(raw, context)

    assert len(issues) == 1
    assert issues[0].severity == Severity.CRITICAL
    assert issues[0].details["module"] == "posix"
    assert issues[0].details["name"] == "popen"


def test_posix_popen_blocklist_and_cve_dedupe_to_single_issue(
    tmp_path: Path, context: MLContext
) -> None:
    """posix.popen ловится и blocklist, и CVE — после dedupe остаётся 1 issue.

    До синхронизации BlocklistDetector молчал (пары не было в blocklist),
    и находку выдавал только CVEDetector. Теперь срабатывают оба, но обе
    записи указывают на один offset и сливаются в ``dedupe_issues``.
    """
    from poison_check.core.result import dedupe_issues
    from poison_check.detectors.cve_detector import CVEDetector
    from poison_check.scanners.pickle_scanner import PickleScanner

    p = tmp_path / "posix_popen.pkl"
    p.write_bytes(_build_global_reduce_pickle("posix", "popen", "id"))

    raw = PickleScanner().scan(p)
    assert raw.error is None

    blocklist_issues = BlocklistDetector().analyze(raw, context)
    cve_issues = CVEDetector().analyze(raw, context)

    # Оба детектора видят угрозу — рассинхрона больше нет.
    assert len(blocklist_issues) == 1, "BlocklistDetector не поймал posix.popen"
    assert any("OS-SYSTEM" in i.code for i in cve_issues), (
        f"CVEDetector не поймал posix.popen: {[i.code for i in cve_issues]}"
    )

    # Обе записи привязаны к одному offset → dedupe сливает их в одну.
    merged = dedupe_issues(blocklist_issues + cve_issues)
    assert len(merged) == 1, (
        f"dedupe должен оставить 1 issue, получено {len(merged)}: "
        f"{[(i.code, i.location) for i in merged]}"
    )
    assert merged[0].severity == Severity.CRITICAL


# --- Py2/legacy алиасы ------------------------------------------------------


def test_thread_module_alias_normalized_to_underscore_thread() -> None:
    """Py2 ``thread`` нормализуется в ``_thread``; имена внутри совпадают."""
    from poison_check.core.known_dangerous import is_py2_alias, normalize_global

    assert normalize_global("thread", "start_new_thread") == (
        "_thread",
        "start_new_thread",
    )
    assert is_py2_alias("thread")
    assert not is_py2_alias("_thread")


def test_commands_module_is_not_an_alias() -> None:
    """``commands`` НЕ алиас: normalize_global оставляет пару как есть.

    Модульный алиас ``commands``→``subprocess`` сломал бы normalize_global на
    функциях без Py3-аналога (getstatus, mkarg, mk2arg). Вместо этого опасные
    пары занесены в KNOWN_DANGEROUS_GLOBALS напрямую.
    """
    from poison_check.core.known_dangerous import (
        KNOWN_DANGEROUS_GLOBALS,
        is_py2_alias,
        normalize_global,
    )

    assert not is_py2_alias("commands")
    assert normalize_global("commands", "getoutput") == ("commands", "getoutput")
    assert ("commands", "getoutput") in KNOWN_DANGEROUS_GLOBALS
    assert ("commands", "getstatusoutput") in KNOWN_DANGEROUS_GLOBALS


def test_dangerous_function_names_cover_new_blocklist_pairs() -> None:
    """Имена всех новых blocklist-пар присутствуют в DANGEROUS_FUNCTION_NAMES.

    Это нужно AllowlistDetector: глобал из trusted_prefix с опасным именем
    (``torch.utils.evil.spawnv``) поднимается до HIGH, а не остаётся INFO.
    """
    from poison_check.core.known_dangerous import DANGEROUS_FUNCTION_NAMES

    new_names = {
        "getoutput", "getstatusoutput",
        "execl", "execle", "execlp", "execvpe",
        "spawnv", "spawnvp", "spawnlp", "spawnvpe",
        "posix_spawn", "posix_spawnp",
        "interact", "breakpoint",
    }
    assert new_names <= DANGEROUS_FUNCTION_NAMES


class TestGetattrNarrowing:
    """Калибровка правки 2: MLS-PKL-001 для getattr сужен по 2-му аргументу.

    getattr(obj, "<безопасное_имя>") (легит-реконструкция, ultralytics/YOLO) не
    флагается; getattr к опасному имени / от не-литерала — остаётся HIGH.
    Опирается на факты сканера в metadata (getattr_literal_attrs / getattr_dynamic).
    """

    def _codes(self, meta: dict[str, str] | None) -> set[str]:
        raw = RawScanData(
            file_path=Path("m.pkl"), file_hash={}, file_size=100,
            scanner_name="pickle", globals={("builtins", "getattr")}, metadata=meta,
        )
        ctx = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])
        return {i.code for i in BlocklistDetector().analyze(raw, ctx)}

    def test_safe_literal_suppressed(self) -> None:
        codes = self._codes({"getattr_present": "true",
                             "getattr_literal_attrs": "DetectionModel,fromkeys"})
        assert "MLS-PKL-001" not in codes

    def test_dangerous_literal_kept(self) -> None:
        codes = self._codes({"getattr_present": "true", "getattr_literal_attrs": "system"})
        assert "MLS-PKL-001" in codes

    def test_dynamic_arg_kept(self) -> None:
        codes = self._codes({"getattr_present": "true", "getattr_dynamic": "true"})
        assert "MLS-PKL-001" in codes

    def test_no_metadata_kept_conservative(self) -> None:
        # Нет данных о литералах → консервативно оставляем HIGH.
        assert "MLS-PKL-001" in self._codes(None)

    def test_mixed_safe_and_dangerous_kept(self) -> None:
        # Хоть одно опасное имя среди литералов → HIGH сохраняется.
        codes = self._codes({"getattr_present": "true",
                             "getattr_literal_attrs": "Conv,system,BatchNorm2d"})
        assert "MLS-PKL-001" in codes

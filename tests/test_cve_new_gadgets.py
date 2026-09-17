"""Регрессионные тесты для новых CVE-гаджетов 2024–2025 в ml_cves.yaml.

Проверяется, что CVEDetector ловит каждый добавленный gadget-модуль:
- pandas.read_pickle (вложенная десериализация)
- bdb/pdb (Bdb.run) — исполнение кода через отладчик
- asteval.Interpreter
- runpy._run_code / _run_module_code
- torch.hub.load

Инвариант проекта: сначала YAML-правило, потом тест. Все вредоносные фикстуры
строятся ВРУЧНУЮ через opcode-конструкцию (не через pickle.dumps злонамеренного
объекта). pickle.load здесь не вызывается ни на одном байте.

Образец подхода — tests/test_bypass_detections.py и
tests/fixtures/malicious/_build_bypass_fixtures.py.
"""

from __future__ import annotations

from pathlib import Path

from poison_check.core.result import Confidence, MLContext, Severity
from poison_check.detectors.cve_detector import CVEDetector
from poison_check.scanners.pickle_scanner import PickleScanner

# ---------------------------------------------------------------------------
# Низкоуровневые помощники opcode-конструкции (подмножество _build_bypass_fixtures)
# ---------------------------------------------------------------------------

PROTO = b"\x80"
GLOBAL = b"c"
EMPTY_TUPLE = b")"
REDUCE = b"R"
STOP = b"."


def _proto(v: int) -> bytes:
    """PROTO opcode + версия протокола."""
    return PROTO + bytes([v])


def _global_textmode(module: str, name: str) -> bytes:
    """GLOBAL opcode с текстовыми аргументами (module\\nname\\n).

    pickletools.genops вернёт arg = "module name"; сканер разложит его в
    (module, name) через rsplit(" ", 1) — поэтому модуль может содержать точки
    (``torch.hub``), а имя — точки для dotted-qualname (``Bdb.run``).
    """
    return GLOBAL + module.encode("utf-8") + b"\n" + name.encode("utf-8") + b"\n"


def _reduce_call_pickle(module: str, name: str) -> bytes:
    """Минимальный pickle: GLOBAL module.name + () + REDUCE + STOP.

    Достаточен, чтобы сканер собрал (module, name) в globals и опкоды
    GLOBAL/REDUCE, а CVEDetector сопоставил паттерн.
    """
    return _proto(2) + _global_textmode(module, name) + EMPTY_TUPLE + REDUCE + STOP


_CTX = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])


def _scan_and_detect(payload: bytes) -> list:
    """Прогоняет payload через PickleScanner + CVEDetector, возвращает issues."""
    raw = PickleScanner().scan_bytes(payload, source_path=Path("<gadget>"))
    return CVEDetector().analyze(raw, _CTX)


def _codes(issues: list) -> set[str]:
    """Множество кодов issue."""
    return {i.code for i in issues}


# ---------------------------------------------------------------------------
# Утверждение: сканер действительно извлёк ожидаемый global
# ---------------------------------------------------------------------------


def test_scanner_extracts_dotted_module_and_name() -> None:
    """GLOBAL 'torch.hub load' → globals содержит ('torch.hub', 'load')."""
    raw = PickleScanner().scan_bytes(
        _reduce_call_pickle("torch.hub", "load"), source_path=Path("<g>")
    )
    assert raw.globals is not None
    assert ("torch.hub", "load") in raw.globals


def test_scanner_extracts_dotted_qualname() -> None:
    """GLOBAL 'bdb Bdb.run' → globals содержит ('bdb', 'Bdb.run')."""
    raw = PickleScanner().scan_bytes(
        _reduce_call_pickle("bdb", "Bdb.run"), source_path=Path("<g>")
    )
    assert raw.globals is not None
    assert ("bdb", "Bdb.run") in raw.globals


# ---------------------------------------------------------------------------
# По одному тесту на каждое новое правило
# ---------------------------------------------------------------------------


def test_pandas_read_pickle_detected() -> None:
    """pandas.read_pickle → MLS-PATTERN-PANDAS-READ-PICKLE (HIGH)."""
    issues = _scan_and_detect(_reduce_call_pickle("pandas", "read_pickle"))
    assert "MLS-PATTERN-PANDAS-READ-PICKLE" in _codes(issues)
    issue = next(i for i in issues if i.code == "MLS-PATTERN-PANDAS-READ-PICKLE")
    assert issue.severity is Severity.HIGH
    assert issue.confidence is Confidence.HIGH


def test_pandas_read_pickle_submodule_detected() -> None:
    """pandas.io.pickle.read_pickle тоже ловится (второй affected_global)."""
    issues = _scan_and_detect(_reduce_call_pickle("pandas.io.pickle", "read_pickle"))
    assert "MLS-PATTERN-PANDAS-READ-PICKLE" in _codes(issues)


def test_bdb_run_detected() -> None:
    """bdb.Bdb.run → MLS-PATTERN-BDB-PDB-RUN (HIGH)."""
    issues = _scan_and_detect(_reduce_call_pickle("bdb", "Bdb.run"))
    assert "MLS-PATTERN-BDB-PDB-RUN" in _codes(issues)
    issue = next(i for i in issues if i.code == "MLS-PATTERN-BDB-PDB-RUN")
    assert issue.severity is Severity.HIGH


def test_pdb_run_detected() -> None:
    """pdb.run (module-level) тоже ловится тем же правилом."""
    issues = _scan_and_detect(_reduce_call_pickle("pdb", "run"))
    assert "MLS-PATTERN-BDB-PDB-RUN" in _codes(issues)


def test_asteval_interpreter_detected() -> None:
    """asteval.Interpreter → MLS-PATTERN-ASTEVAL-INTERP (HIGH/MEDIUM)."""
    issues = _scan_and_detect(_reduce_call_pickle("asteval", "Interpreter"))
    assert "MLS-PATTERN-ASTEVAL-INTERP" in _codes(issues)
    issue = next(i for i in issues if i.code == "MLS-PATTERN-ASTEVAL-INTERP")
    assert issue.severity is Severity.HIGH
    assert issue.confidence is Confidence.MEDIUM


def test_runpy_run_code_detected() -> None:
    """runpy._run_code → MLS-PATTERN-RUNPY-EXEC (HIGH)."""
    issues = _scan_and_detect(_reduce_call_pickle("runpy", "_run_code"))
    assert "MLS-PATTERN-RUNPY-EXEC" in _codes(issues)


def test_runpy_run_module_code_detected() -> None:
    """runpy._run_module_code — второй affected_global того же правила."""
    issues = _scan_and_detect(_reduce_call_pickle("runpy", "_run_module_code"))
    assert "MLS-PATTERN-RUNPY-EXEC" in _codes(issues)


def test_torch_hub_load_detected() -> None:
    """torch.hub.load → MLS-PATTERN-TORCH-HUB (HIGH/MEDIUM)."""
    issues = _scan_and_detect(_reduce_call_pickle("torch.hub", "load"))
    assert "MLS-PATTERN-TORCH-HUB" in _codes(issues)
    issue = next(i for i in issues if i.code == "MLS-PATTERN-TORCH-HUB")
    assert issue.severity is Severity.HIGH
    assert issue.confidence is Confidence.MEDIUM


def test_torch_hub_state_dict_from_url_detected() -> None:
    """torch.hub.load_state_dict_from_url тоже ловится."""
    issues = _scan_and_detect(
        _reduce_call_pickle("torch.hub", "load_state_dict_from_url")
    )
    assert "MLS-PATTERN-TORCH-HUB" in _codes(issues)


# ---------------------------------------------------------------------------
# Отрицательные тесты: новые правила не дают ложных срабатываний
# ---------------------------------------------------------------------------


def test_clean_pandas_core_not_flagged_by_new_rules() -> None:
    """pandas.core.frame.DataFrame (легитимный global) не триггерит новые правила.

    pandas.read_pickle и pandas.core.* — разные globals; правило PANDAS-READ-PICKLE
    не должно срабатывать на обычной сериализации DataFrame.
    """
    issues = _scan_and_detect(
        _reduce_call_pickle("pandas.core.frame", "DataFrame")
    )
    assert "MLS-PATTERN-PANDAS-READ-PICKLE" not in _codes(issues)


def test_new_gadget_rules_have_russian_texts() -> None:
    """Каждое новое правило несёт русские title_ru/description_ru/remediation_ru."""
    detector = CVEDetector()
    new_ids = {
        "PATTERN-PANDAS-READ-PICKLE",
        "PATTERN-BDB-PDB-RUN",
        "PATTERN-ASTEVAL-INTERP",
        "PATTERN-RUNPY-EXEC",
        "PATTERN-TORCH-HUB",
    }
    by_id = {p.id: p for p in detector._patterns}
    for rule_id in new_ids:
        assert rule_id in by_id, f"Правило {rule_id} не загружено"
        pattern = by_id[rule_id]
        assert pattern.title_ru.strip()
        assert pattern.description_ru.strip()
        assert pattern.remediation_ru.strip()
        assert pattern.references, f"{rule_id}: нет references"

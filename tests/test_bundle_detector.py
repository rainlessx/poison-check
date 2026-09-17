"""Тесты BundleCodeDetector — интерпретация фактов комплекта в Issue.

Проверяется граница уровней: Уровень 1 (INFO, MLS-BUNDLE-001/003) vs Уровень 2
(HIGH, MLS-BUNDLE-002), и ключевое различие «код исполнится при загрузке» vs
«просто лежит .py» (опасный паттерн в НЕссылочном .py остаётся INFO, не HIGH).
"""

from __future__ import annotations

from pathlib import Path

from poison_check.core.result import MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.bundle_detector import METADATA_BUNDLE, BundleCodeDetector
from poison_check.scanners.bundle_scanner import BundleScanner


def _raw(py_files: dict, root: str = "/bundle") -> RawScanData:
    return RawScanData(
        file_path=Path(root),
        file_hash={},
        file_size=0,
        scanner_name="bundle",
        metadata={METADATA_BUNDLE: {"root": root, "config_present": True,
                                    "config_error": None, "code_refs": {},
                                    "py_files": py_files}},
    )


def _py(referenced: bool, names: list[tuple[str, int, str]]) -> dict:
    return {
        "referenced": referenced,
        "names": [{"name": n, "line": ln, "kind": k, "pos": "load"} for n, ln, k in names],
        "obfuscations": [],
        "parse_error": None,
    }


def _pyp(
    referenced: bool,
    names: list[tuple[str, int, str, str]],
    obfs: list[tuple[int, str]] | None = None,
) -> dict:
    """py-факты с явной позицией (n, line, kind, pos) и обфускациями (line, kind)."""
    return {
        "referenced": referenced,
        "names": [{"name": n, "line": ln, "kind": k, "pos": p} for n, ln, k, p in names],
        "obfuscations": [{"line": ln, "pos": "load", "kind": k} for ln, k in (obfs or [])],
        "parse_error": None,
    }


_CTX = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])


def _analyze(py_files: dict) -> list:
    return BundleCodeDetector().analyze(_raw(py_files), _CTX)


class TestLevels:
    """Уровни серьёзности и их коды."""

    def test_referenced_dangerous_gives_high(self) -> None:
        issues = _analyze({"modeling.py": _py(True, [("exec", 5, "call")])})
        codes = {i.code: i.severity for i in issues}
        assert codes["MLS-BUNDLE-001"] == Severity.INFO   # факт наличия кода
        assert codes["MLS-BUNDLE-002"] == Severity.HIGH    # опасный паттерн по ссылке

    def test_referenced_clean_gives_info_only(self) -> None:
        # Реалистичный безобидный код: только INFO (Уровень 1), без HIGH.
        issues = _analyze({"modeling.py": _py(True, [
            ("numpy", 1, "import"), ("numpy.matmul", 10, "call"),
        ])})
        codes = {i.code for i in issues}
        assert codes == {"MLS-BUNDLE-001"}
        assert all(i.severity == Severity.INFO for i in issues)

    def test_unreferenced_dangerous_stays_info_not_high(self) -> None:
        # Опасный паттерн, но .py НЕ ссылается из конфига → Уровень 1, НЕ HIGH.
        issues = _analyze({"evil.py": _py(False, [("exec", 3, "call")])})
        codes = {i.code: i.severity for i in issues}
        assert codes == {"MLS-BUNDLE-003": Severity.INFO}
        assert "MLS-BUNDLE-002" not in codes
        # Паттерн зафиксирован в details, но не эскалирован.
        noref = next(i for i in issues if i.code == "MLS-BUNDLE-003")
        assert noref.details["patterns"], "паттерн должен быть отмечен в details"

    def test_weights_only_no_py_no_issues(self) -> None:
        assert _analyze({}) == []


class TestMatching:
    """Сопоставление имён с паттернами Уровня 2."""

    def test_subprocess_prefix_matches(self) -> None:
        issues = _analyze({"m.py": _py(True, [("subprocess.run", 2, "call")])})
        assert any(i.code == "MLS-BUNDLE-002" for i in issues)

    def test_os_path_does_not_match_os_system(self) -> None:
        # os.path.join не должен считаться опасным (точный матч os.system).
        issues = _analyze({"m.py": _py(True, [("os.path.join", 2, "call")])})
        assert not any(i.code == "MLS-BUNDLE-002" for i in issues)

    def test_os_environ_is_silent(self) -> None:
        # os.environ НАМЕРЕННО убран из паттернов (рутинно в modeling.py —
        # выбор устройства). Политика «молчание для рутинного»: ноль срабатываний.
        issues = _analyze({"m.py": _py(True, [("os.environ", 2, "attr"),
                                              ("os.environ.get", 2, "call")])})
        assert not any(i.code == "MLS-BUNDLE-002" for i in issues)
        assert {i.code for i in issues} == {"MLS-BUNDLE-001"}


class TestPositionSeverity:
    """Детект 3: позиция узла определяет severity (HIGH load vs INFO method)."""

    def test_dangerous_at_load_is_high(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [("exec", 3, "call", "load")])})
        b002 = [i for i in issues if i.code == "MLS-BUNDLE-002"]
        assert b002 and b002[0].severity == Severity.HIGH

    def test_dangerous_in_method_is_info(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [("exec", 9, "call", "method")])})
        b002 = [i for i in issues if i.code == "MLS-BUNDLE-002"]
        assert b002 and b002[0].severity == Severity.INFO

    def test_both_positions_emit_high_and_info(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [
            ("exec", 3, "call", "load"), ("os.system", 9, "call", "method")])})
        sevs = sorted(i.severity for i in issues if i.code == "MLS-BUNDLE-002")
        assert Severity.HIGH in sevs and Severity.INFO in sevs


class TestCallClassSeverity:
    """Калибровка правки 1: severity MLS-BUNDLE-002 зависит от КЛАССА вызова.

    code-exec (exec/subprocess/…) в загрузочной точке → HIGH; side-effect
    (сеть/ФС/native: requests/socket/ctypes) → LOW (факт, виден в отчёте, но
    ниже порога любой политики — не гейтит); любой класс в обычном методе → INFO.
    """

    def _sev(self, issues: list, code: str = "MLS-BUNDLE-002") -> set:
        return {i.severity for i in issues if i.code == code}

    def test_side_effect_at_load_is_low_not_high(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [("requests.get", 6, "call", "load")])})
        assert self._sev(issues) == {Severity.LOW}

    def test_socket_at_load_is_low(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [("socket.socket", 4, "call", "load")])})
        assert self._sev(issues) == {Severity.LOW}

    def test_code_exec_at_load_stays_high(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [("subprocess.run", 5, "call", "load")])})
        assert self._sev(issues) == {Severity.HIGH}

    def test_side_effect_in_method_is_info(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [("requests.get", 20, "call", "method")])})
        assert self._sev(issues) == {Severity.INFO}

    def test_mixed_classes_at_load_split_high_and_low(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [
            ("exec", 3, "call", "load"), ("requests.get", 4, "call", "load")])})
        assert self._sev(issues) == {Severity.HIGH, Severity.LOW}


class TestObfuscation:
    """Детект 4: MLS-BUNDLE-004 (INFO) по фактам обфускации, содержимое не раскрыто."""

    def test_obfuscation_emits_info_004(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [], obfs=[(2, "dynamic-exec")])})
        b004 = [i for i in issues if i.code == "MLS-BUNDLE-004"]
        assert b004 and b004[0].severity == Severity.INFO

    def test_obfuscation_never_high(self) -> None:
        issues = _analyze({"m.py": _pyp(True, [],
                                        obfs=[(2, "encoded-literal"), (3, "chr-assembly")])})
        assert all(i.severity != Severity.HIGH for i in issues)


class TestFilters:
    """Детектор реагирует только на факты BundleScanner."""

    def test_non_bundle_scanner_ignored(self) -> None:
        raw = RawScanData(
            file_path=Path("x.pkl"), file_hash={}, file_size=0,
            scanner_name="pickle", metadata={METADATA_BUNDLE: {"py_files": {}}},
        )
        assert BundleCodeDetector().analyze(raw, _CTX) == []

    def test_no_bundle_metadata_ignored(self) -> None:
        raw = RawScanData(
            file_path=Path("/b"), file_hash={}, file_size=0,
            scanner_name="bundle", metadata={},
        )
        assert BundleCodeDetector().analyze(raw, _CTX) == []


class TestIntegration:
    """Сквозной путь BundleScanner → BundleCodeDetector на реальных директориях."""

    def _run(self, root: Path) -> dict:
        raw = BundleScanner().scan(root)
        issues = BundleCodeDetector().analyze(raw, _CTX)
        return {i.code: i.severity for i in issues}

    def test_referenced_exec_end_to_end(self, tmp_path: Path) -> None:
        import json
        (tmp_path / "config.json").write_text(
            json.dumps({"auto_map": {"AutoModel": "modeling.M"}}))
        # Безобидный маркер внутри exec — реальная нагрузка не используется.
        (tmp_path / "modeling.py").write_text(
            "exec(\"open('proof_x.txt','w').write('triggered')\")\n")
        codes = self._run(tmp_path)
        assert codes.get("MLS-BUNDLE-002") == Severity.HIGH

    def test_realistic_clean_no_high(self, tmp_path: Path) -> None:
        import json
        (tmp_path / "config.json").write_text(
            json.dumps({"auto_map": {"AutoModel": "modeling.M"}}))
        (tmp_path / "modeling.py").write_text(
            "import numpy as np\n"
            "class M:\n"
            "    def forward(self, x):\n"
            "        return np.matmul(x, x)\n")
        codes = self._run(tmp_path)
        assert "MLS-BUNDLE-002" not in codes
        assert codes.get("MLS-BUNDLE-001") == Severity.INFO

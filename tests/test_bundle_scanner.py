"""Тесты BundleScanner — извлечение фактов из директории-комплекта.

Проверяется НЕЙТРАЛЬНОЕ извлечение фактов (config.json → ссылки на .py;
AST-имена вызовов/импортов/атрибутов), без интерпретации угрозы. Код в
фикстурах НЕ исполняется — только ast.parse. Опасные паттерны в фикстурах несут
безобидный маркер, а не реальную нагрузку.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from poison_check.scanners.bundle_scanner import (
    METADATA_BUNDLE,
    BundleScanner,
    _extract_names,
    _FactExtractor,
)


def _write(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")


def _facts(root: Path) -> dict:
    raw = BundleScanner().scan(root)
    assert raw.scanner_name == "bundle"
    assert raw.error is None
    assert raw.metadata is not None
    return raw.metadata[METADATA_BUNDLE]


class TestConfigRefs:
    """Разбор ссылок config.json на локальный код."""

    def test_auto_map_reference_resolved_and_marked(self, tmp_path: Path) -> None:
        _write(tmp_path, "config.json",
               json.dumps({"auto_map": {"AutoModel": "modeling_x.MyModel"}}))
        _write(tmp_path, "modeling_x.py", "class MyModel:\n    pass\n")
        facts = _facts(tmp_path)
        assert facts["config_present"] is True
        assert facts["code_refs"] == {"auto_map.AutoModel": "modeling_x.MyModel"}
        assert facts["py_files"]["modeling_x.py"]["referenced"] is True

    def test_custom_pipeline_string_reference(self, tmp_path: Path) -> None:
        _write(tmp_path, "config.json",
               json.dumps({"custom_pipeline": "pipeline"}))
        _write(tmp_path, "pipeline.py", "def run():\n    return 1\n")
        facts = _facts(tmp_path)
        assert facts["py_files"]["pipeline.py"]["referenced"] is True

    def test_unreferenced_py_marked_not_referenced(self, tmp_path: Path) -> None:
        _write(tmp_path, "config.json", json.dumps({"model_type": "bert"}))
        _write(tmp_path, "helper.py", "x = 1\n")
        facts = _facts(tmp_path)
        assert facts["py_files"]["helper.py"]["referenced"] is False

    def test_no_config_no_error(self, tmp_path: Path) -> None:
        _write(tmp_path, "helper.py", "x = 1\n")
        facts = _facts(tmp_path)
        assert facts["config_present"] is False
        assert facts["config_error"] is None
        assert facts["py_files"]["helper.py"]["referenced"] is False

    def test_broken_config_recorded_not_raised(self, tmp_path: Path) -> None:
        _write(tmp_path, "config.json", "{ это не json ")
        facts = _facts(tmp_path)
        assert facts["config_present"] is True
        assert facts["config_error"] is not None

    def test_remote_ref_with_double_dash_ignored(self, tmp_path: Path) -> None:
        # HF-форма "repo--module.Cls" — не локальная ссылка, .py не помечается.
        _write(tmp_path, "config.json",
               json.dumps({"auto_map": {"AutoModel": "some-repo--modeling.Model"}}))
        _write(tmp_path, "modeling.py", "class Model:\n    pass\n")
        facts = _facts(tmp_path)
        assert facts["py_files"]["modeling.py"]["referenced"] is False

    def test_path_traversal_target_not_resolved(self, tmp_path: Path) -> None:
        # Цель с ../ не должна разрешаться в файл вне комплекта.
        _write(tmp_path, "config.json",
               json.dumps({"auto_map": {"AutoModel": "../evil.Model"}}))
        facts = _facts(tmp_path)
        # ../evil.py вне корня — среди referenced его быть не должно.
        assert all(not f["referenced"] for f in facts["py_files"].values())


class TestNameExtraction:
    """AST-извлечение имён вызовов/импортов/атрибутов."""

    def _names(self, source: str) -> set[str]:
        return {n["name"] for n in _extract_names(ast.parse(source))}

    def test_bare_builtin_call(self) -> None:
        assert "exec" in self._names("exec('x')\n")

    def test_dotted_call(self) -> None:
        names = self._names("import os\nos.system('ls')\n")
        assert "os.system" in names

    def test_attribute_access_without_call(self) -> None:
        # os.environ используется как подписка — это attr, не call.
        names = self._names("import os\nk = os.environ['X']\n")
        assert "os.environ" in names

    def test_import_alias_resolved(self) -> None:
        # import os as o; o.system(...) → нормализуется к os.system.
        names = self._names("import os as o\no.system('ls')\n")
        assert "os.system" in names

    def test_from_import_recorded(self) -> None:
        names = self._names("from subprocess import run\nrun(['ls'])\n")
        assert "subprocess.run" in names

    def test_benign_names_not_dangerous(self) -> None:
        names = self._names("import os\np = os.path.join('a', 'b')\n")
        # os.path.join присутствует, но os.system/os.popen/os.environ — нет.
        assert "os.path.join" in names
        assert "os.system" not in names


class TestIndirectAndAssembly:
    """Детект 1/2: разрешение косвенного имени через ЛИТЕРАЛ (один узел + литералы)."""

    def _facts(self, src: str, cls: frozenset[str] = frozenset()) -> _FactExtractor:
        e = _FactExtractor(cls)
        e.run(ast.parse(src))
        return e

    def test_getattr_literal_resolved(self) -> None:
        e = self._facts("getattr(__import__('builtins'),'exec')('x')")
        indirect = {n["name"] for n in e.names if n["kind"] == "indirect"}
        assert "builtins.exec" in indirect

    def test_name_assembled_from_literals(self) -> None:
        e = self._facts("getattr(__import__('buil'+'tins'),'ex'+'ec')('x')")
        indirect = {n["name"] for n in e.names if n["kind"] == "indirect"}
        assert "builtins.exec" in indirect

    def test_non_literal_arg_not_resolved(self) -> None:
        # Аргумент-переменная НЕ разрешается в конкретный глобал (не data-flow).
        e = self._facts("getattr(os, name)")
        indirect = {n["name"] for n in e.names if n["kind"] == "indirect"}
        assert indirect == set()


class TestPosition:
    """Детект 3: позиция узла (load vs method), dunder auto_map-класса."""

    def _facts(self, src: str, cls: frozenset[str]) -> _FactExtractor:
        e = _FactExtractor(cls)
        e.run(ast.parse(src))
        return e

    def test_module_level_is_load(self) -> None:
        e = self._facts("exec('x')\n", frozenset())
        assert [n["pos"] for n in e.names if n["name"] == "exec"] == ["load"]

    def test_init_of_auto_map_class_is_load(self) -> None:
        e = self._facts("class M:\n def __init__(self):\n  exec('x')\n", frozenset({"M"}))
        assert [n["pos"] for n in e.names if n["name"] == "exec"] == ["load"]

    def test_ordinary_method_is_method(self) -> None:
        e = self._facts("class M:\n def train(self):\n  exec('x')\n", frozenset({"M"}))
        assert [n["pos"] for n in e.names if n["name"] == "exec"] == ["method"]

    def test_dunder_of_non_auto_map_class_is_method(self) -> None:
        # __init__ класса, на который auto_map НЕ ссылается → не загрузочная точка.
        e = self._facts("class Other:\n def __init__(self):\n  exec('x')\n", frozenset({"M"}))
        assert [n["pos"] for n in e.names if n["name"] == "exec"] == ["method"]


class TestObfuscationSignals:
    """Детект 4: сигналы обфускации (только позиция load), содержимое не раскрыто."""

    def _obf(self, src: str) -> set[str]:
        e = _FactExtractor(frozenset())
        e.run(ast.parse(src))
        return {o["kind"] for o in e.obfuscations}

    def test_getattr_non_literal(self) -> None:
        assert "dynamic-getattr" in self._obf("getattr(o, name)\n")

    def test_exec_non_literal(self) -> None:
        assert "dynamic-exec" in self._obf("exec(data)\n")

    def test_base64_decode(self) -> None:
        assert "encoded-literal" in self._obf("import base64\nbase64.b64decode(b'x')\n")

    def test_chr_assembly(self) -> None:
        assert "chr-assembly" in self._obf("n = chr(111) + chr(115)\n")

    def test_literal_exec_is_not_obfuscation(self) -> None:
        # exec от ЛИТЕРАЛА — видимый код, это не обфускация (детект прямой).
        assert self._obf("exec('open(\"p\",\"w\").write(\"x\")')\n") == set()

    def test_obfuscation_only_at_load_position(self) -> None:
        # Обфускация в обычном методе не фиксируется (вектор — загрузочный код).
        e = _FactExtractor(frozenset({"M"}))
        e.run(ast.parse("class M:\n def train(self):\n  getattr(o, name)\n"))
        assert e.obfuscations == []


class TestPyParsing:
    """Устойчивость разбора .py."""

    def test_syntax_error_recorded_not_raised(self, tmp_path: Path) -> None:
        _write(tmp_path, "broken.py", "def (:\n")
        facts = _facts(tmp_path)
        assert facts["py_files"]["broken.py"]["parse_error"] is not None

    def test_code_is_not_executed(self, tmp_path: Path) -> None:
        # Если бы сканер исполнял код, появился бы файл-маркер. Его быть не должно.
        marker = tmp_path / "proof_scanner.txt"
        _write(tmp_path, "modeling.py",
               f"open({str(marker)!r}, 'w').write('x')\n")
        _write(tmp_path, "config.json",
               json.dumps({"auto_map": {"AutoModel": "modeling.M"}}))
        _facts(tmp_path)
        assert not marker.exists(), "сканер не должен исполнять код комплекта"

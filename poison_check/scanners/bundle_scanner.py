"""Сканер комплекта модели: директория с весами + загрузочным кодом.

Отдельный вектор угрозы: исполняемый Python лежит РЯДОМ с весами (не внутри
файла весов) и исполняется при загрузке через механизм Hugging Face
``trust_remote_code`` — ``config.json`` с полями ``auto_map`` /
``custom_pipeline`` / ``auto_modelcard``, указывающими на локальный ``.py``.

Разделение слоёв: сканер знает ФОРМАТ комплекта и
извлекает НЕЙТРАЛЬНЫЕ факты (какие ``.py`` на что ссылается конфиг; какие
имена вызовов/импортов/атрибутов встречаются в каждом ``.py``), УГРОЗУ
интерпретирует :class:`BundleCodeDetector` по списку паттернов из YAML.

Анализ СТАТИЧЕСКИЙ: код разбирается через :func:`ast.parse`, НЕ исполняется.
Граница (non-goal): сканер не судит, опасен ли код, и не делает data-flow —
только извлекает синтаксические факты.

Этот сканер работает на УРОВНЕ ДИРЕКТОРИИ и не участвует в пофайловой
диспетчеризации :class:`ScannerRegistry` (он не «берёт» отдельный файл);
вызывается напрямую из обхода директории (:mod:`poison_check.analysis.bundle`).
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from poison_check.core.scanner_base import BaseScanner, RawScanData

logger = logging.getLogger(__name__)

#: Имя загрузочного конфига Hugging Face.
_CONFIG_NAME: str = "config.json"

#: Поля config.json, ссылающиеся на локальный код (механизм trust_remote_code).
_CODE_REF_FIELDS: tuple[str, ...] = ("auto_map", "custom_pipeline", "auto_modelcard")

#: Ключ в RawScanData.metadata, под которым лежат факты комплекта.
METADATA_BUNDLE: str = "bundle"

#: Максимальный размер одного .py для разбора (защита от «.py-бомбы»): код
#: загрузчика модели реально занимает килобайты; читать в память многомегабайтный
#: файл под видом .py не нужно.
_MAX_PY_SIZE: int = 2_000_000


@dataclass
class _PyFacts:
    """Нейтральные факты об одном .py-файле комплекта.

    Все поля — синтаксические: имена узлов, позиция узла в дереве, разрешённые
    ЛИТЕРАЛЫ внутри узла. Никакого data-flow (значения переменных между шагами
    не отслеживаются).
    """

    referenced: bool = False  # на файл ссылается config.json
    # [{name, line, kind, pos}] — pos: "load" (исполнится при загрузке) | "method"
    names: list[dict[str, Any]] = field(default_factory=list)
    # [{line, pos, kind}] — сигналы динамического/обфусцированного разрешения имён
    obfuscations: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None  # текст SyntaxError, если .py не разобрался


class BundleScanner(BaseScanner):
    """Сканер комплекта модели (директория): извлекает факты о коде и ссылках.

    НЕ интерпретирует угрозу — складывает факты в ``RawScanData.metadata`` под
    ключом ``bundle``; их разбирает :class:`BundleCodeDetector`.
    """

    name: ClassVar[str] = "bundle"
    description: ClassVar[str] = (
        "Сканер комплекта модели: код рядом с весами (trust_remote_code)"
    )
    supported_extensions: ClassVar[list[str]] = []  # директория, не файл
    magic_bytes: ClassVar[list[bytes]] = []

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Всегда False: комплект — это директория, пофайловой диспетчеризации нет.

        Сканер вызывается напрямую из обхода директории, а не через
        ``ScannerRegistry.find_scanner`` (тот работает с отдельными файлами).
        """
        return False

    def scan(self, path: Path) -> RawScanData:
        """Извлекает факты комплекта из директории ``path``.

        :param path: Корень директории-комплекта.
        :return: RawScanData (scanner_name='bundle') с фактами в metadata['bundle'].
            Поле ``error`` НЕ заполняется: отсутствие конфига/кода — не ошибка
            разбора, а штатная ситуация (см. :class:`BundleCodeDetector`).
        """
        config_present, code_refs, config_error = self._read_config_refs(path)
        py_facts = self._collect_py_facts(path, code_refs)

        bundle_facts: dict[str, Any] = {
            "root": str(path),
            "config_present": config_present,
            "config_error": config_error,
            # field -> target (как записано в config.json), для сообщения детектора
            "code_refs": code_refs,
            # имя .py (относительно корня) -> факты
            "py_files": {name: _facts_to_dict(f) for name, f in py_facts.items()},
        }
        return RawScanData(
            file_path=path,
            file_hash={},
            file_size=0,
            scanner_name=self.name,
            metadata={METADATA_BUNDLE: bundle_facts},
        )

    # ------------------------------------------------------------------
    # Разбор config.json
    # ------------------------------------------------------------------
    def _read_config_refs(
        self, root: Path
    ) -> tuple[bool, dict[str, str], str | None]:
        """Читает config.json и собирает ссылки на локальный код.

        :return: (config_present, {field: target}, config_error).
            ``target`` — строка из конфига (напр. ``"modeling_x.MyModel"`` или
            ``"pipeline.py"``); разрешение в .py-файл делает
            :meth:`_resolve_ref_module`.
        """
        config_path = root / _CONFIG_NAME
        if not config_path.is_file():
            return False, {}, None
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError) as exc:
            # Битый config — не наша угроза и не ошибка разбора весов; фиксируем
            # факт, чтобы детектор мог при желании о нём сообщить, и идём дальше.
            return True, {}, f"{type(exc).__name__}: {exc}"
        if not isinstance(raw, dict):
            return True, {}, None

        refs: dict[str, str] = {}
        for field_name in _CODE_REF_FIELDS:
            value = raw.get(field_name)
            for key, target in _iter_ref_targets(field_name, value):
                refs[key] = target
        return True, refs, None

    # ------------------------------------------------------------------
    # Сбор фактов .py
    # ------------------------------------------------------------------
    def _collect_py_facts(
        self, root: Path, code_refs: dict[str, str]
    ) -> dict[str, _PyFacts]:
        """Разбирает все .py комплекта и помечает, на какие из них ссылается конфиг.

        Набор .py = объединение: все ``*.py`` в корне директории + файлы,
        разрешённые из ссылок конфига (могут лежать во вложенном пакете).
        """
        referenced_files = {
            self._resolve_ref_module(root, target) for target in code_refs.values()
        }
        referenced_files.discard(None)

        py_paths: dict[str, Path] = {}
        for p in sorted(root.glob("*.py")):
            py_paths[p.name] = p
        for rp in referenced_files:
            if rp is not None and rp.is_file():
                py_paths[str(rp.relative_to(root))] = rp

        facts: dict[str, _PyFacts] = {}
        referenced_rel = {
            str(rp.relative_to(root)) for rp in referenced_files if rp is not None
        }
        # Имена классов auto_map по файлам: их dunder'ы (__init__/__new__/…) —
        # загрузочные точки (детект 3). Цель "modeling.CustomModel" → файл
        # modeling.py, класс CustomModel.
        auto_map_classes = self._auto_map_classes_by_file(root, code_refs)
        for rel_name, p in sorted(py_paths.items()):
            f = _PyFacts(referenced=rel_name in referenced_rel)
            self._extract_py(p, f, auto_map_classes.get(rel_name, frozenset()))
            facts[rel_name] = f
        return facts

    @staticmethod
    def _auto_map_classes_by_file(
        root: Path, code_refs: dict[str, str]
    ) -> dict[str, frozenset[str]]:
        """Сопоставляет .py-файл → множество имён классов, на которые ссылается auto_map.

        Только имя класса (последний компонент цели после модуля) — синтаксис,
        не анализ кода. ``"modeling.CustomModel"`` → {"modeling.py": {"CustomModel"}}.
        """
        out: dict[str, set[str]] = {}
        for target in code_refs.values():
            if not target or "--" in target or target.endswith(".py"):
                continue
            module, _, cls = target.rpartition(".")
            if not module or not cls:
                continue
            rel = module.replace(".", "/") + ".py"
            out.setdefault(rel, set()).add(cls)
        return {k: frozenset(v) for k, v in out.items()}

    @staticmethod
    def _resolve_ref_module(root: Path, target: str) -> Path | None:
        """Разрешает ссылку из config.json в путь к .py внутри комплекта.

        Примеры целей: ``"modeling_x.MyModel"`` → ``modeling_x.py``;
        ``"pkg.mod.Cls"`` → ``pkg/mod.py``; ``"pipeline.py"`` → ``pipeline.py``.
        Удалённые ссылки HF вида ``"repo--module.Cls"`` игнорируются (не локальны).
        """
        if not target or "--" in target:
            return None
        module = target[:-3] if target.endswith(".py") else target.rsplit(".", 1)[0]
        if not module:
            return None
        candidate = root / (module.replace(".", "/") + ".py")
        try:
            # Защита от побега за пределы комплекта (path traversal в target).
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def _extract_py(
        self, path: Path, facts: _PyFacts, auto_map_classes: frozenset[str]
    ) -> None:
        """Разбирает один .py через ast.parse и заполняет facts (names + obfuscations).

        Код НЕ исполняется. При SyntaxError / ошибке чтения — фиксируем в
        ``facts.parse_error`` и возвращаемся (файл остаётся в отчёте как
        «не разобран», но прогон не падает).

        :param auto_map_classes: имена классов этого файла, на которые ссылается
            auto_map — их dunder'ы считаются загрузочными точками (детект 3).
        """
        try:
            if path.stat().st_size > _MAX_PY_SIZE:
                facts.parse_error = "файл .py слишком большой для статического разбора"
                return
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            facts.parse_error = f"OSError: {exc}"
            return
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            facts.parse_error = f"SyntaxError: {exc}"
            return
        except ValueError as exc:
            # ast.parse кидает ValueError, напр. на NUL-байте в исходнике.
            facts.parse_error = f"ValueError: {exc}"
            return
        extractor = _FactExtractor(auto_map_classes)
        extractor.run(tree)
        facts.names = extractor.names
        facts.obfuscations = extractor.obfuscations


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def _iter_ref_targets(field_name: str, value: Any) -> list[tuple[str, str]]:
    """Нормализует значение поля-ссылки config.json в список (ключ, цель).

    ``auto_map`` — словарь ``{"AutoModel": "modeling_x.Model", ...}``.
    ``custom_pipeline`` / ``auto_modelcard`` — строка ``"pipeline"`` или путь.
    """
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for sub_key, target in value.items():
            if isinstance(target, str) and target:
                out.append((f"{field_name}.{sub_key}", target))
            elif isinstance(target, list):
                # Встречается форма ["config_module.Cls", "modeling_module.Cls"].
                for i, t in enumerate(target):
                    if isinstance(t, str) and t:
                        out.append((f"{field_name}.{sub_key}[{i}]", t))
    elif isinstance(value, str) and value:
        out.append((field_name, value))
    return out


def _facts_to_dict(facts: _PyFacts) -> dict[str, Any]:
    """Сериализует _PyFacts в обычный словарь для metadata."""
    return {
        "referenced": facts.referenced,
        "names": facts.names,
        "obfuscations": facts.obfuscations,
        "parse_error": facts.parse_error,
    }


def _dotted_name(node: ast.AST) -> str | None:
    """Собирает dotted-имя из ast.Name / ast.Attribute (``a.b.c``).

    Возвращает None для вычисляемых выражений (``obj[i].attr`` и т.п.) — их мы
    не нормализуем (это уже был бы data-flow, вне границы).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    return None


# Dunder-методы класса, исполняемые при загрузке/десериализации модели.
_LOAD_DUNDERS: frozenset[str] = frozenset(
    {"__init__", "__new__", "__init_subclass__", "__reduce__", "__setstate__"}
)

# Функции косвенного разрешения имени: f(base, "literal") / f("literal").
_GETATTR_NAMES: frozenset[str] = frozenset({"getattr"})
_IMPORT_NAMES: frozenset[str] = frozenset(
    {"__import__", "importlib.import_module", "importlib.__import__"}
)
# Функции исполнения кода из строки.
_CODE_EXEC_NAMES: frozenset[str] = frozenset({"exec", "eval", "compile"})
# Декодеры, из которых может собираться ИМЯ (сигнал обфускации, детект 4).
_DECODE_NAMES: frozenset[str] = frozenset(
    {"base64.b64decode", "base64.b64encode", "base64.b32decode", "base64.b16decode",
     "base64.a85decode", "base64.standard_b64decode", "base64.urlsafe_b64decode",
     "bytes.fromhex", "codecs.decode", "codecs.escape_decode", "binascii.unhexlify",
     "binascii.a2b_hex", "binascii.a2b_base64"}
)


def _const_str(node: ast.AST) -> str | None:
    """Вычисляет СТРОКОВЫЙ ЛИТЕРАЛ, собираемый на AST внутри одного выражения.

    Поддерживает: строковую константу, конкатенацию ``"o"+"s"`` константных
    строк, ``"sep".join([...const...])``. Любой НЕ-константный операнд →
    None (его мы не вычисляем — это был бы трекинг значений). Это разрешение
    литералов ВНУТРИ узла, не data-flow.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _const_str(node.left)
        right = _const_str(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):  # f-string
        parts: list[str] = []
        for v in node.values:
            p = _const_str(v)
            if p is None:
                return None
            parts.append(p)
        return "".join(parts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
    ):
        sep = _const_str(node.func.value)
        seq = node.args[0]
        if sep is not None and isinstance(seq, ast.List | ast.Tuple):
            items: list[str] = []
            for e in seq.elts:
                p = _const_str(e)
                if p is None:
                    return None
                items.append(p)
            return sep.join(items)
    return None


def _has_chr_assembly(node: ast.AST) -> bool:
    """True, если выражение собирает строку из chr(...)-последовательности.

    Узнаётся синтаксически: конкатенация/join, среди операндов которой есть
    вызовы chr(...). Содержимое не вычисляется — только факт сборки по символам.
    """
    found = {"chr": False, "assembly": False}

    def walk(n: ast.AST, in_assembly: bool) -> None:
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            walk(n.left, True)
            walk(n.right, True)
            return
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "join"
        ):
            for e in n.args:
                walk(e, True)
            return
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "chr":
            found["chr"] = True
            if in_assembly:
                found["assembly"] = True
        if isinstance(n, ast.List | ast.Tuple):
            for e in n.elts:
                walk(e, in_assembly)

    walk(node, False)
    return found["chr"] and found["assembly"]


class _FactExtractor:
    """Извлекает синтаксические факты (имена, позиция, обфускация) из AST.

    Принцип границы: каждое правило смотрит на ОДИН узел и литералы внутри него
    плюс позицию узла в дереве (стек областей видимости). Значения переменных
    между шагами НЕ отслеживаются — это был бы data-flow, вне границы.
    """

    def __init__(self, auto_map_classes: frozenset[str]) -> None:
        self._auto_map_classes = auto_map_classes
        self._alias_map: dict[str, str] = {}
        self.names: list[dict[str, Any]] = []
        self.obfuscations: list[dict[str, Any]] = []
        self._seen: set[tuple[str, int, str]] = set()
        self._seen_obf: set[tuple[int, str]] = set()
        # Стек областей: элементы ("class", name) / ("func", name) / ("decorator",)
        self._scope: list[tuple[str, str]] = []

    # --- публичный вход ---
    def run(self, tree: ast.AST) -> None:
        """Запускает два прохода: алиасы импортов, затем обход с позицией."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._alias_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self._alias_map[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        self._visit_body_of(tree)

    # --- позиция узла ---
    def _pos(self) -> str:
        """Классифицирует текущую позицию: 'load' (исполнится при загрузке) | 'method'.

        load: module-/class-level код, декораторы, dunder'ы auto_map-класса.
        method: тело обычного метода/функции (при загрузке не вызывается).
        Это позиция узла в дереве, НЕ доказательство исполнения.
        """
        # Декоратор исполняется при определении (= при импорте) → load.
        if any(kind == "decorator" for kind, _ in self._scope):
            return "load"
        # Ближайшая объемлющая функция и ближайший класс.
        enclosing_func: str | None = None
        enclosing_class: str | None = None
        for kind, name in reversed(self._scope):
            if kind == "func" and enclosing_func is None:
                enclosing_func = name
            elif kind == "class" and enclosing_class is None:
                enclosing_class = name
        if enclosing_func is None:
            return "load"  # module- или class-level statement
        if (
            enclosing_func in _LOAD_DUNDERS
            and enclosing_class is not None
            and enclosing_class in self._auto_map_classes
        ):
            return "load"  # dunder auto_map-класса
        return "method"

    # --- обход тел с отслеживанием позиции ---
    def _visit_body_of(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self._dispatch(child)

    def _dispatch(self, node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            self._visit_decorators(node.decorator_list)
            self._scope.append(("class", node.name))
            for child in node.body:
                self._dispatch(child)
            self._scope.pop()
            return
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            self._visit_decorators(node.decorator_list)
            self._scope.append(("func", node.name))
            for child in node.body:
                self._dispatch(child)
            self._scope.pop()
            return
        # Прочие узлы: осмотреть сам узел на факты и спуститься вглубь.
        self._inspect(node)
        self._visit_body_of(node)

    def _visit_decorators(self, decorators: list[ast.expr]) -> None:
        self._scope.append(("decorator", ""))
        for dec in decorators:
            self._dispatch(dec)
        self._scope.pop()

    # --- извлечение фактов из одного узла ---
    def _inspect(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self._add(alias.name, node.lineno, "import")
        elif isinstance(node, ast.ImportFrom) and node.module:
            self._add(node.module, node.lineno, "import")
            for alias in node.names:
                self._add(f"{node.module}.{alias.name}", node.lineno, "import")
        elif isinstance(node, ast.Call):
            self._inspect_call(node)
        elif isinstance(node, ast.Attribute):
            self._add(self._dotted(node), node.lineno, "attr")
        elif (
            isinstance(node, ast.BinOp)
            and self._pos() == "load"
            and _has_chr_assembly(node)
        ):
            # chr()-сборка в голом выражении (не аргумент вызова): x = chr()+chr().
            self._add_obf(node.lineno, "chr-assembly")

    def _inspect_call(self, node: ast.Call) -> None:
        func_name = self._dotted(node.func)
        line = node.lineno

        if func_name in _CODE_EXEC_NAMES:
            # exec/eval/compile: совпадение с паттерном ТОЛЬКО при ЛИТЕРАЛЬНОМ
            # аргументе (код виден → детект прямой). НЕ-литерал → имя не
            # добавляем: это сигнал обфускации (детект 4), а не доказанная опасность.
            if node.args and _const_str(node.args[0]) is not None:
                self._add(func_name, line, "call")
        else:
            # Прямое dotted-имя вызова (os.system(...), socket.socket(...)).
            self._add(func_name, line, "call")

        # Детект 1/2: косвенное разрешение через ЛИТЕРАЛ (getattr/__import__).
        resolved = self._resolve_indirect(node)
        if resolved is not None:
            self._add(resolved, line, "indirect")

        # Детект 4: сигналы динамического/обфусцированного разрешения имён.
        self._detect_obfuscation(node, func_name, line)

    def _resolve_indirect(self, node: ast.Call) -> str | None:
        """Разрешает getattr(base,"lit") / __import__("lit") в dotted-имя по литералам.

        Только если аргумент-имя — строковый литерал (в т.ч. собранный на AST из
        констант). НЕ-литеральный аргумент → None (здесь не разрешаем; см.
        _detect_obfuscation). Смотрит ОДИН узел вызова и литералы в его поддереве.
        """
        fname = self._dotted(node.func)
        if fname in _GETATTR_NAMES and len(node.args) >= 2:
            base = self._resolve_base(node.args[0])
            attr = _const_str(node.args[1])
            if base is not None and attr is not None:
                return f"{base}.{attr}"
        if fname in _IMPORT_NAMES and len(node.args) >= 1:
            mod = _const_str(node.args[0])
            return mod if mod else None
        return None

    def _resolve_base(self, node: ast.AST) -> str | None:
        """Разрешает «базу» для getattr: имя/атрибут/вложенный __import__('lit')."""
        dotted = self._dotted(node)
        if dotted is not None:
            return dotted
        if isinstance(node, ast.Call):
            fname = self._dotted(node.func)
            if fname in _IMPORT_NAMES and node.args:
                return _const_str(node.args[0])
        return None

    def _detect_obfuscation(self, node: ast.Call, func_name: str | None, line: int) -> None:
        """Фиксирует ФАКТ обфускации/динамического разрешения (детект 4). Без раскрытия.

        Только в загрузочной позиции — вектор про код, исполняемый при загрузке.
        """
        if self._pos() != "load":
            return
        # getattr / __import__ / import_module с НЕ-литеральным аргументом.
        if func_name in _GETATTR_NAMES and len(node.args) >= 2 and _const_str(node.args[1]) is None:
            self._add_obf(line, "dynamic-getattr")
        if func_name in _IMPORT_NAMES and node.args and _const_str(node.args[0]) is None:
            self._add_obf(line, "dynamic-import")
        # exec/eval/compile от НЕ-литерального аргумента.
        if func_name in _CODE_EXEC_NAMES and node.args and _const_str(node.args[0]) is None:
            self._add_obf(line, "dynamic-exec")
        # Декодеры (base64/hex/codecs) — имя может собираться из закодированного.
        if func_name in _DECODE_NAMES:
            self._add_obf(line, "encoded-literal")
        # chr()-последовательность, собирающая строку (в т.ч. ''.join([chr,...])
        # как сам вызов, или chr()+chr() как аргумент).
        if _has_chr_assembly(node) or any(_has_chr_assembly(a) for a in node.args):
            self._add_obf(line, "chr-assembly")

    # --- низкоуровневые помощники ---
    def _dotted(self, node: ast.AST) -> str | None:
        name = _dotted_name(node)
        return self._resolve_alias(name) if name is not None else None

    def _resolve_alias(self, dotted: str) -> str:
        head, _, tail = dotted.partition(".")
        real_head = self._alias_map.get(head)
        if real_head is None:
            return dotted
        return f"{real_head}.{tail}" if tail else real_head

    def _add(self, name: str | None, line: int, kind: str) -> None:
        if not name:
            return
        resolved = self._resolve_alias(name)
        pos = self._pos()
        key = (resolved, line, kind)
        if key in self._seen:
            return
        self._seen.add(key)
        self.names.append({"name": resolved, "line": line, "kind": kind, "pos": pos})

    def _add_obf(self, line: int, kind: str) -> None:
        key = (line, kind)
        if key in self._seen_obf:
            return
        self._seen_obf.add(key)
        self.obfuscations.append({"line": line, "pos": "load", "kind": kind})


def _extract_names(tree: ast.AST) -> list[dict[str, Any]]:
    """Совместимость: извлекает нейтральные имена из AST (без auto_map-контекста).

    Тонкая обёртка над :class:`_FactExtractor` — возвращает только список имён
    (``{name, line, kind, pos}``). Используется в юнит-тестах извлечения имён.
    """
    extractor = _FactExtractor(frozenset())
    extractor.run(tree)
    return extractor.names

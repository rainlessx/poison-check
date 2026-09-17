"""Декомпилятор pickle-потока в читаемый Python-код.

Базовая версия MVP (неделя 9): покрывает >90% реальных случаев.
Полная версия будет расширяться итеративно.

Подход: символическое выполнение Pickle VM.
Вместо реального выполнения строим Python AST / строки кода,
имитируя состояние стека и генерируя операторы присваивания.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

from poison_check.core.result import OpcodeInfo
from poison_check.core.scanner_base import RawScanData

logger = logging.getLogger(__name__)


def _safe_unquote(expr: str) -> str:
    """Безопасно снимает кавычки со строкового Python-литерала.

    Раньше использовался ``eval(expr)``, что технически работало (вход —
    результат ``repr()``), но было хрупким инвариантом: добавление любого
    нового источника ``_sym`` со строкой, начинающейся с кавычки, могло бы
    превратить ``eval`` в RCE-вектор внутри сканера, который призван
    защищать от RCE.

    ``ast.literal_eval`` ограничен литералами (str/bytes/int/float/None/True/
    False/tuple/list/dict/set), поэтому даже подменённый вход не сможет
    выполнить произвольный Python-код.

    При ошибке парсинга — fallback к простому strip кавычек, как было раньше.
    """
    try:
        result = ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return expr.strip("'\"")
    return str(result)

# ---------------------------------------------------------------------------
# Символические значения на стеке
# ---------------------------------------------------------------------------

_MARK_SENTINEL = object()  # метка для MARK-опкода


class _Sym:
    """Символическое значение на стеке декомпилятора.

    Хранит Python-выражение в виде строки и флаг callable.
    """

    __slots__ = ("expr", "callable")

    def __init__(self, expr: str, is_callable: bool = False) -> None:
        """Инициализирует символическое значение.

        Args:
            expr: Python-выражение (строка).
            is_callable: True если значение — вызываемый объект (функция/класс).
        """
        self.expr = expr
        self.callable = is_callable

    def __repr__(self) -> str:
        return f"_Sym({self.expr!r})"


def _sym(expr: str, is_callable: bool = False) -> _Sym:
    """Вспомогательная фабрика для _Sym."""
    return _Sym(expr, is_callable)


# ---------------------------------------------------------------------------
# Вспомогательные функции форматирования
# ---------------------------------------------------------------------------

def _repr_str(value: str) -> str:
    """Форматирует строку как Python-литерал."""
    return repr(value)


def _repr_int(value: int) -> str:
    """Форматирует целое число как Python-литерал."""
    return repr(value)


def _format_args(args: _Sym | None) -> str:
    """Форматирует аргументы вызова функции.

    Args:
        args: Символическое значение аргументов (кортеж или None).

    Returns:
        Строка аргументов без внешних скобок, готовая для подстановки в вызов.
    """
    if args is None:
        return ""
    expr = args.expr
    # Если выражение — кортеж в скобках, убираем внешние скобки
    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1]
        # trailing comma для одноэлементных кортежей: (x,) → x
        if inner.endswith(","):
            return inner[:-1]
        return inner
    return expr


# ---------------------------------------------------------------------------
# Основной класс декомпилятора
# ---------------------------------------------------------------------------

class PickleDecompiler:
    """Декомпилятор pickle opcode-потока в читаемый Python-код.

    Реализует символическое выполнение подмножества Pickle VM:
    вместо реального выполнения строит строки Python-кода,
    генерируя переменные _var0, _var1, ... для промежуточных результатов.

    Базовая версия MVP покрывает опкоды:
        PROTO, GLOBAL, STACK_GLOBAL, REDUCE, MARK,
        STRING, UNICODE, SHORT_BINUNICODE, BINUNICODE,
        INT, BININT, BININT1, BININT2, LONG, LONG1, LONG4,
        TUPLE, TUPLE1, TUPLE2, TUPLE3, EMPTY_TUPLE,
        LIST, EMPTY_LIST, APPEND, APPENDS,
        DICT, EMPTY_DICT, SETITEM, SETITEMS,
        STOP, NONE, NEWTRUE, NEWFALSE,
        BUILD, NEWOBJ
    """

    def __init__(self) -> None:
        """Инициализирует внутреннее состояние декомпилятора."""
        self._var_counter: int = 0
        self._stack: list[Any] = []
        self._memo: dict[int, _Sym] = {}
        self._output_lines: list[str] = []
        self._imports: dict[str, set[str]] = {}  # module → {name, ...}

    def _reset(self) -> None:
        """Сбрасывает состояние перед каждым вызовом decompile()."""
        self._var_counter = 0
        self._stack = []
        self._memo = {}
        self._output_lines = []
        self._imports = {}

    def _new_var(self) -> str:
        """Генерирует уникальное имя переменной: _var0, _var1, ...

        Returns:
            Имя переменной в формате _varN.
        """
        name = f"_var{self._var_counter}"
        self._var_counter += 1
        return name

    def _register_import(self, module: str, name: str) -> str:
        """Регистрирует импорт и возвращает квалифицированное имя.

        Args:
            module: Имя модуля (например, "os").
            name: Имя атрибута (например, "system").

        Returns:
            Квалифицированное Python-выражение (например, "os.system").
        """
        if module not in self._imports:
            self._imports[module] = set()
        self._imports[module].add(name)
        if module:
            return f"{module}.{name}"
        return name

    def _build_import_lines(self) -> list[str]:
        """Генерирует строки импортов в начало результирующего кода.

        Returns:
            Список строк вида 'import os', 'import subprocess', ...
        """
        lines: list[str] = []
        for module in sorted(self._imports):
            if module:
                lines.append(f"import {module}")
        return lines

    def _pop(self) -> _Sym | None:
        """Снимает верхний элемент стека.

        Returns:
            Символическое значение или None если стек пуст.
        """
        if self._stack:
            val = self._stack.pop()
            if val is _MARK_SENTINEL:
                return None
            if isinstance(val, _Sym):
                return val
            return _sym(repr(val))
        return None

    def _pop_until_mark(self) -> list[_Sym]:
        """Снимает элементы со стека до MARK-сентинела включительно.

        Returns:
            Список символических значений (в порядке от нижнего к верхнему).
        """
        items: list[_Sym] = []
        while self._stack:
            val = self._stack.pop()
            if val is _MARK_SENTINEL:
                break
            if isinstance(val, _Sym):
                items.insert(0, val)
            else:
                items.insert(0, _sym(repr(val)))
        return items

    def _pop_skipping_mark(self) -> _Sym | None:
        """Снимает верхний элемент стека, прозрачно пропуская MARK-сентинел.

        Используется в REDUCE для поиска функции, когда между функцией
        и аргументами в стеке может находиться MARK (протокол 0/1).

        Returns:
            Символическое значение или None если стек пуст.
        """
        # Сначала пытаемся обычный pop
        if not self._stack:
            return None
        val = self._stack.pop()
        if val is _MARK_SENTINEL:
            # MARK был между args и func — берём следующий элемент
            if not self._stack:
                return None
            val = self._stack.pop()
        if val is _MARK_SENTINEL:
            return None
        if isinstance(val, _Sym):
            return val
        return _sym(repr(val))

    def _peek(self) -> _Sym | None:
        """Возвращает верхний элемент стека без снятия.

        Returns:
            Символическое значение или None если стек пуст.
        """
        if self._stack:
            val = self._stack[-1]
            if val is _MARK_SENTINEL:
                return None
            if isinstance(val, _Sym):
                return val
        return None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def decompile(
        self,
        opcodes: list[OpcodeInfo],
        globals_used: set[tuple[str, str]] | None = None,
    ) -> str:
        """Принимает список опкодов и возвращает читаемый Python-код.

        Реализует символическое выполнение Pickle VM: имитирует стек,
        но вместо выполнения строит строки Python-кода.

        Args:
            opcodes: Список OpcodeInfo из RawScanData.opcodes.
            globals_used: Множество (module, name) из RawScanData.globals
                          (используется для дополнительных импортов).

        Returns:
            Python-код в виде строки. Пустая строка если opcodes пуст.
        """
        self._reset()

        if not opcodes:
            return ""

        for opinfo in opcodes:
            self._process_opcode(opinfo)

        # Финальный результат: верхушка стека
        result_var: str | None = None
        if self._stack:
            top = self._stack[-1]
            if isinstance(top, _Sym) and top.expr:
                result_var = top.expr

        # Собираем финальный код
        import_lines = self._build_import_lines()
        body_lines = list(self._output_lines)

        if result_var and (
            not body_lines or not body_lines[-1].startswith("result =")
        ):
            body_lines.append(f"result = {result_var}")

        all_lines = import_lines + ([""] if import_lines and body_lines else []) + body_lines
        return "\n".join(all_lines)

    def decompile_from_raw(self, raw_data: RawScanData) -> str | None:
        """Удобная обёртка над decompile() для работы с RawScanData.

        Args:
            raw_data: Объект RawScanData из PickleScanner.

        Returns:
            Python-код или None если opcodes недоступны.
        """
        if raw_data.opcodes is None:
            return None
        return self.decompile(
            raw_data.opcodes,
            raw_data.globals or set(),
        )

    # ------------------------------------------------------------------
    # Обработка опкодов
    # ------------------------------------------------------------------

    def _process_opcode(self, opinfo: OpcodeInfo) -> None:
        """Диспетчеризует один опкод на соответствующий обработчик.

        Args:
            opinfo: Один опкод из pickle-потока.
        """
        name = opinfo.opcode
        arg = opinfo.arg

        # --- Версия протокола ---
        if name == "PROTO":
            # Не генерирует код, только комментарий в начале
            if not self._output_lines:
                proto = int(arg) if arg is not None else "?"
                self._output_lines.append(f"# pickle protocol {proto}")

        # --- Импорт функции/класса ---
        elif name == "GLOBAL":
            self._handle_global(arg)

        elif name == "STACK_GLOBAL":
            self._handle_stack_global()

        # --- Строковые литералы ---
        elif name in (
            "STRING", "UNICODE",
            "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
        ):
            str_val: str = arg if isinstance(arg, str) else ""
            self._stack.append(_sym(_repr_str(str_val)))

        elif name in ("BINSTRING", "SHORT_BINSTRING"):
            # Bytes-литерал
            if isinstance(arg, bytes):
                self._stack.append(_sym(repr(arg)))
            elif isinstance(arg, str):
                self._stack.append(_sym(_repr_str(arg)))
            else:
                self._stack.append(_sym("b\"\""))

        # --- Числа ---
        elif name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"):
            self._stack.append(_sym(_repr_int(int(arg)) if arg is not None else "0"))

        elif name in ("FLOAT", "BINFLOAT"):
            self._stack.append(_sym(repr(float(arg)) if arg is not None else "0.0"))

        # --- Константы ---
        elif name == "NONE":
            self._stack.append(_sym("None"))

        elif name == "NEWTRUE":
            self._stack.append(_sym("True"))

        elif name == "NEWFALSE":
            self._stack.append(_sym("False"))

        # --- MARK ---
        elif name == "MARK":
            self._stack.append(_MARK_SENTINEL)

        # --- Кортежи ---
        elif name == "EMPTY_TUPLE":
            self._stack.append(_sym("()"))

        elif name == "TUPLE":
            items = self._pop_until_mark()
            if not items:
                self._stack.append(_sym("()"))
            elif len(items) == 1:
                self._stack.append(_sym(f"({items[0].expr},)"))
            else:
                inner = ", ".join(i.expr for i in items)
                self._stack.append(_sym(f"({inner})"))

        elif name == "TUPLE1":
            a = self._pop()
            expr = a.expr if a else "None"
            self._stack.append(_sym(f"({expr},)"))

        elif name == "TUPLE2":
            b = self._pop()
            a = self._pop()
            ea = a.expr if a else "None"
            eb = b.expr if b else "None"
            self._stack.append(_sym(f"({ea}, {eb})"))

        elif name == "TUPLE3":
            c = self._pop()
            b = self._pop()
            a = self._pop()
            ea = a.expr if a else "None"
            eb = b.expr if b else "None"
            ec = c.expr if c else "None"
            self._stack.append(_sym(f"({ea}, {eb}, {ec})"))

        # --- Списки ---
        elif name == "EMPTY_LIST":
            self._stack.append(_sym("[]"))

        elif name == "LIST":
            items = self._pop_until_mark()
            if not items:
                self._stack.append(_sym("[]"))
            else:
                inner = ", ".join(i.expr for i in items)
                self._stack.append(_sym(f"[{inner}]"))

        elif name == "APPEND":
            item = self._pop()
            lst = self._peek()
            if lst is not None and item is not None:
                # Генерируем .append()
                var = self._new_var()
                self._output_lines.append(f"{var} = {lst.expr}")
                self._output_lines.append(f"{var}.append({item.expr})")
                # Обновляем верхушку стека
                self._stack[-1] = _sym(var)

        elif name == "APPENDS":
            items = self._pop_until_mark()
            lst = self._peek()
            if lst is not None and items:
                var = self._new_var()
                self._output_lines.append(f"{var} = {lst.expr}")
                for it in items:
                    self._output_lines.append(f"{var}.append({it.expr})")
                self._stack[-1] = _sym(var)

        # --- Словари ---
        elif name == "EMPTY_DICT":
            self._stack.append(_sym("{}"))

        elif name == "DICT":
            items = self._pop_until_mark()
            # DICT собирает пары ключ-значение (чётное количество)
            pairs: list[str] = []
            for idx in range(0, len(items) - 1, 2):
                k = items[idx].expr
                v = items[idx + 1].expr
                pairs.append(f"{k}: {v}")
            inner = ", ".join(pairs)
            self._stack.append(_sym(f"{{{inner}}}"))

        elif name == "SETITEM":
            setitem_val = self._pop()
            setitem_key = self._pop()
            dct = self._peek()
            if dct is not None and setitem_key is not None and setitem_val is not None:
                var = self._new_var()
                self._output_lines.append(f"{var} = {dct.expr}")
                self._output_lines.append(
                    f"{var}[{setitem_key.expr}] = {setitem_val.expr}"
                )
                self._stack[-1] = _sym(var)

        elif name == "SETITEMS":
            items = self._pop_until_mark()
            dct = self._peek()
            if dct is not None and items:
                var = self._new_var()
                self._output_lines.append(f"{var} = {dct.expr}")
                for idx in range(0, len(items) - 1, 2):
                    k = items[idx].expr
                    v = items[idx + 1].expr
                    self._output_lines.append(f"{var}[{k}] = {v}")
                self._stack[-1] = _sym(var)

        # --- Вызов функции ---
        elif name == "REDUCE":
            self._handle_reduce()

        # --- Создание объекта ---
        elif name in ("NEWOBJ", "NEWOBJ_EX"):
            self._handle_newobj(name)

        # --- Инициализация объекта через __setstate__ ---
        elif name == "BUILD":
            self._handle_build()

        # --- Memo ---
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            if arg is not None:
                top = self._peek()
                if top is not None:
                    self._memo[int(arg)] = top

        elif name == "MEMOIZE":
            top = self._peek()
            if top is not None:
                idx = len(self._memo)
                self._memo[idx] = top

        elif name in ("GET", "BINGET", "LONG_BINGET"):
            if arg is not None:
                cached = self._memo.get(int(arg))
                self._stack.append(cached if cached is not None else _sym("None"))

        # --- Дубликат ---
        elif name == "DUP":
            if self._stack:
                self._stack.append(self._stack[-1])

        # --- POP ---
        elif name == "POP":
            self._pop()

        elif name == "POP_MARK":
            self._pop_until_mark()

        # --- Завершение потока ---
        elif name == "STOP":
            pass  # Завершение — финальный результат возьмём в decompile()

        # --- INST (протокол 0) ---
        elif name == "INST":
            self._handle_inst(arg)

        # --- Остальные опкоды — не генерируем код, не падаем ---
        else:
            pass

    # ------------------------------------------------------------------
    # Обработчики конкретных опкодов
    # ------------------------------------------------------------------

    def _handle_global(self, arg: Any) -> None:
        """Обрабатывает опкод GLOBAL 'module name'.

        Args:
            arg: Аргумент вида 'os system'.
        """
        if not isinstance(arg, str):
            self._stack.append(_sym("None"))
            return

        parts = arg.rsplit(" ", 1)
        if len(parts) == 2:
            module, name = parts[0], parts[1]
        else:
            module, name = arg, ""

        qualified = self._register_import(module, name)
        self._stack.append(_sym(qualified, is_callable=True))

    def _handle_stack_global(self) -> None:
        """Обрабатывает опкод STACK_GLOBAL: снимает name, module со стека."""
        name_sym = self._pop()
        module_sym = self._pop()

        name = name_sym.expr.strip("'\"") if name_sym else ""
        module = module_sym.expr.strip("'\"") if module_sym else ""

        # Снимаем кавычки для строковых литералов через ast.literal_eval —
        # безопасный аналог eval(), ограниченный литералами (см. _safe_unquote).
        if name_sym and name_sym.expr.startswith(("'", '"')):
            name = _safe_unquote(name_sym.expr)

        if module_sym and module_sym.expr.startswith(("'", '"')):
            module = _safe_unquote(module_sym.expr)

        qualified = self._register_import(module, name)
        self._stack.append(_sym(qualified, is_callable=True))

    def _handle_reduce(self) -> None:
        """Обрабатывает опкод REDUCE: func(*args).

        Снимает аргументы и функцию, генерирует вызов функции.
        В pickle VM стек может выглядеть как [func, MARK, args] —
        MARK между функцией и аргументами пропускается при поиске функции.
        """
        args = self._pop()
        # Пропускаем MARK-сентинел если он между args и func
        func = self._pop_skipping_mark()

        func_expr = func.expr if func else "unknown"
        args_str = _format_args(args)

        var = self._new_var()
        self._output_lines.append(f"{var} = {func_expr}({args_str})")
        self._stack.append(_sym(var))

    def _handle_newobj(self, opname: str) -> None:
        """Обрабатывает NEWOBJ / NEWOBJ_EX: cls(*args) или cls(*args, **kwargs).

        Args:
            opname: "NEWOBJ" или "NEWOBJ_EX".
        """
        if opname == "NEWOBJ_EX":
            kwargs = self._pop()
            args = self._pop()
            cls = self._pop()
        else:
            args = self._pop()
            cls = self._pop()
            kwargs = None

        cls_expr = cls.expr if cls else "object"
        args_str = _format_args(args)

        if kwargs is not None and kwargs.expr not in ("None", "{}"):
            call = f"{cls_expr}({args_str}, **{kwargs.expr})"
        else:
            call = f"{cls_expr}({args_str})"

        var = self._new_var()
        self._output_lines.append(f"{var} = {call}")
        self._stack.append(_sym(var))

    def _handle_build(self) -> None:
        """Обрабатывает BUILD: obj.__setstate__(state) или obj.__dict__.update(state).

        Снимает state и obj со стека, генерирует вызов __setstate__.
        """
        state = self._pop()
        obj = self._peek()  # obj остаётся на стеке

        if obj is not None and state is not None:
            self._output_lines.append(
                f"{obj.expr}.__setstate__({state.expr})"
            )

    def _handle_inst(self, arg: Any) -> None:
        """Обрабатывает опкод INST (протокол 0): cls(*args) через items до MARK.

        Args:
            arg: Строка 'module name'.
        """
        items = self._pop_until_mark()

        if isinstance(arg, str):
            parts = arg.rsplit(" ", 1)
            module = parts[0] if len(parts) == 2 else arg
            name = parts[1] if len(parts) == 2 else ""
            qualified = self._register_import(module, name)
        else:
            qualified = "unknown"

        args_str = ", ".join(i.expr for i in items)
        var = self._new_var()
        self._output_lines.append(f"{var} = {qualified}({args_str})")
        self._stack.append(_sym(var))

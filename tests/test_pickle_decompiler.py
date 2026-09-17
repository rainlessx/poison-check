"""Тесты для PickleDecompiler — базовая версия MVP.

Использует вредоносные fixtures из tests/fixtures/malicious/,
созданные через opcode-конструкцию (не pickle.dumps злонамеренных объектов).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from poison_check.analysis.pickle_decompiler import PickleDecompiler
from poison_check.core.result import OpcodeInfo
from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.pickle_scanner import PickleScanner

FIXTURES = Path(__file__).parent / "fixtures"
MALICIOUS_DIR = FIXTURES / "malicious"
SAFE_DIR = FIXTURES / "safe"

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _scan(path: Path) -> RawScanData:
    """Сканирует файл и возвращает RawScanData."""
    scanner = PickleScanner()
    return scanner.scan(path)


def _decompile(path: Path) -> str:
    """Сканирует файл и возвращает декомпилированный Python-код."""
    raw = _scan(path)
    decompiler = PickleDecompiler()
    result = decompiler.decompile_from_raw(raw)
    assert result is not None, f"decompile_from_raw вернул None для {path.name}"
    return result


# ---------------------------------------------------------------------------
# Тест 1: os.system payload — "os.system" в выводе
# ---------------------------------------------------------------------------


def test_decompile_os_system_payload_contains_os_system() -> None:
    """Декомпиляция os.system payload содержит 'os.system' в выводе."""
    code = _decompile(MALICIOUS_DIR / "payload_01_os_system.pkl")
    assert "os.system" in code, (
        f"'os.system' не найден в декомпилированном коде:\n{code}"
    )


def test_decompile_os_system_legacy_payload() -> None:
    """Декомпиляция legacy os_system.pkl (с аргументом) содержит 'os.system'."""
    code = _decompile(MALICIOUS_DIR / "os_system.pkl")
    assert "os.system" in code, (
        f"'os.system' не найден в декомпилированном коде:\n{code}"
    )


# ---------------------------------------------------------------------------
# Тест 2: subprocess payload — "subprocess" в выводе
# ---------------------------------------------------------------------------


def test_decompile_subprocess_payload_contains_subprocess() -> None:
    """Декомпиляция subprocess.Popen payload содержит 'subprocess' в выводе."""
    code = _decompile(MALICIOUS_DIR / "payload_02_subprocess_popen.pkl")
    assert "subprocess" in code, (
        f"'subprocess' не найден в декомпилированном коде:\n{code}"
    )


def test_decompile_subprocess_payload_contains_popen() -> None:
    """Декомпиляция subprocess.Popen payload содержит 'Popen' в выводе."""
    code = _decompile(MALICIOUS_DIR / "payload_02_subprocess_popen.pkl")
    assert "Popen" in code, (
        f"'Popen' не найден в декомпилированном коде:\n{code}"
    )


# ---------------------------------------------------------------------------
# Тест 3: eval payload — "eval" в выводе
# ---------------------------------------------------------------------------


def test_decompile_eval_payload_contains_eval() -> None:
    """Декомпиляция builtins.eval payload содержит 'eval' в выводе."""
    code = _decompile(MALICIOUS_DIR / "payload_03_builtins_eval.pkl")
    assert "eval" in code, (
        f"'eval' не найден в декомпилированном коде:\n{code}"
    )


# ---------------------------------------------------------------------------
# Тест 4: builtins.exec — для полноты покрытия
# ---------------------------------------------------------------------------


def test_decompile_exec_payload_contains_exec() -> None:
    """Декомпиляция builtins.exec payload содержит 'exec' в выводе."""
    code = _decompile(MALICIOUS_DIR / "payload_04_builtins_exec.pkl")
    assert "exec" in code, (
        f"'exec' не найден в декомпилированном коде:\n{code}"
    )


# ---------------------------------------------------------------------------
# Тест 5: чистый список — валидный Python без os/subprocess импортов
# ---------------------------------------------------------------------------


def test_decompile_safe_list_no_dangerous_imports() -> None:
    """Декомпиляция безопасного списка не содержит import os/subprocess."""
    code = _decompile(SAFE_DIR / "simple_list.pkl")
    assert "import os" not in code, (
        f"'import os' найден в безопасном файле:\n{code}"
    )
    assert "import subprocess" not in code, (
        f"'import subprocess' найден в безопасном файле:\n{code}"
    )


def test_decompile_safe_list_no_system_call() -> None:
    """Декомпиляция безопасного списка не содержит вызовов os.system/subprocess."""
    code = _decompile(SAFE_DIR / "simple_list.pkl")
    assert "os.system" not in code
    assert "subprocess" not in code


# ---------------------------------------------------------------------------
# Тест 6: пустой opcodes → "" или None, не Exception
# ---------------------------------------------------------------------------


def test_decompile_empty_opcodes_returns_empty_string() -> None:
    """decompile([]) возвращает пустую строку, не бросает исключение."""
    decompiler = PickleDecompiler()
    result = decompiler.decompile(opcodes=[], globals_used=set())
    assert result == "", f"Ожидалась пустая строка, получили: {result!r}"


def test_decompile_from_raw_none_opcodes_returns_none() -> None:
    """decompile_from_raw с opcodes=None возвращает None, не бросает исключение."""
    decompiler = PickleDecompiler()
    raw = RawScanData(
        file_path=Path("test.pkl"),
        file_hash={},
        file_size=0,
        scanner_name="pickle",
        opcodes=None,
    )
    result = decompiler.decompile_from_raw(raw)
    assert result is None, f"Ожидался None, получили: {result!r}"


def test_decompile_empty_opcodes_no_exception() -> None:
    """decompile с пустым списком не бросает никаких исключений."""
    decompiler = PickleDecompiler()
    try:
        decompiler.decompile([], set())
    except Exception as exc:
        pytest.fail(f"decompile([]) бросил исключение: {exc}")


# ---------------------------------------------------------------------------
# Тест 7: вывод decompile — валидный Python (ast.parse)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload_path",
    [
        MALICIOUS_DIR / "payload_01_os_system.pkl",
        MALICIOUS_DIR / "payload_02_subprocess_popen.pkl",
        MALICIOUS_DIR / "payload_03_builtins_eval.pkl",
        MALICIOUS_DIR / "payload_04_builtins_exec.pkl",
        MALICIOUS_DIR / "os_system.pkl",
        SAFE_DIR / "simple_list.pkl",
        SAFE_DIR / "simple_dict.pkl",
    ],
    ids=[
        "payload_01_os_system",
        "payload_02_subprocess_popen",
        "payload_03_builtins_eval",
        "payload_04_builtins_exec",
        "os_system_legacy",
        "safe_list",
        "safe_dict",
    ],
)
def test_decompile_output_is_valid_python(payload_path: Path) -> None:
    """Вывод decompile() является синтаксически валидным Python-кодом.

    Проверяется через ast.parse() — если парсинг не бросает исключение,
    код корректен синтаксически.
    """
    code = _decompile(payload_path)
    if not code.strip():
        return  # пустой вывод — допустим

    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(
            f"Декомпилированный код не является валидным Python "
            f"для {payload_path.name}:\n{exc}\n\nКод:\n{code}"
        )


# ---------------------------------------------------------------------------
# Тест 8: decompile_from_raw — удобная обёртка работает корректно
# ---------------------------------------------------------------------------


def test_decompile_from_raw_os_system() -> None:
    """decompile_from_raw корректно работает с RawScanData от PickleScanner."""
    raw = _scan(MALICIOUS_DIR / "payload_01_os_system.pkl")
    decompiler = PickleDecompiler()
    code = decompiler.decompile_from_raw(raw)
    assert code is not None
    assert "os.system" in code


# ---------------------------------------------------------------------------
# Тест 9: import-строки генерируются для вредоносных payload
# ---------------------------------------------------------------------------


def test_decompile_os_system_has_import_statement() -> None:
    """Декомпиляция os.system payload содержит 'import os'."""
    code = _decompile(MALICIOUS_DIR / "payload_01_os_system.pkl")
    assert "import os" in code, (
        f"'import os' не найден в декомпилированном коде:\n{code}"
    )


def test_decompile_subprocess_has_import_statement() -> None:
    """Декомпиляция subprocess.Popen payload содержит 'import subprocess'."""
    code = _decompile(MALICIOUS_DIR / "payload_02_subprocess_popen.pkl")
    assert "import subprocess" in code, (
        f"'import subprocess' не найден в декомпилированном коде:\n{code}"
    )


# ---------------------------------------------------------------------------
# Тест 10: повторное использование декомпилятора — reset работает корректно
# ---------------------------------------------------------------------------


def test_decompiler_reuse_is_clean() -> None:
    """Повторное использование одного экземпляра даёт чистые результаты."""
    decompiler = PickleDecompiler()

    raw1 = _scan(MALICIOUS_DIR / "payload_01_os_system.pkl")
    code1 = decompiler.decompile_from_raw(raw1)

    raw2 = _scan(MALICIOUS_DIR / "payload_02_subprocess_popen.pkl")
    code2 = decompiler.decompile_from_raw(raw2)

    assert code1 is not None
    assert code2 is not None

    # Результаты не должны смешиваться
    assert "subprocess" not in code1 or "os" in code1  # os.system есть в code1
    assert "os.system" not in code2 or "subprocess" in code2  # subprocess есть в code2
    assert "subprocess" in code2


# ---------------------------------------------------------------------------
# Тест 11: протокол в комментарии
# ---------------------------------------------------------------------------


def test_decompile_includes_protocol_comment() -> None:
    """Декомпилированный код начинается с комментария о версии протокола."""
    code = _decompile(MALICIOUS_DIR / "payload_01_os_system.pkl")
    assert "# pickle protocol" in code, (
        f"Комментарий о протоколе не найден:\n{code}"
    )


# ---------------------------------------------------------------------------
# Тест 12: decompile с минимальным OpcodeInfo (прямой вызов без сканера)
# ---------------------------------------------------------------------------


def test_decompile_minimal_global_reduce_stop() -> None:
    """decompile с минимальным набором опкодов: GLOBAL + EMPTY_TUPLE + REDUCE + STOP."""
    opcodes = [
        OpcodeInfo(position=0, opcode="PROTO", arg=2),
        OpcodeInfo(position=2, opcode="GLOBAL", arg="os system"),
        OpcodeInfo(position=12, opcode="EMPTY_TUPLE", arg=None),
        OpcodeInfo(position=13, opcode="REDUCE", arg=None),
        OpcodeInfo(position=14, opcode="STOP", arg=None),
    ]
    decompiler = PickleDecompiler()
    code = decompiler.decompile(opcodes, {("os", "system")})

    assert "os.system" in code
    assert "import os" in code

    # Валидный Python
    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(f"Код не является валидным Python:\n{exc}\n\nКод:\n{code}")

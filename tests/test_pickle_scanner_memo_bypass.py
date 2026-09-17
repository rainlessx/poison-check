"""Регрессионные тесты: STACK_GLOBAL bypass через memo (BINPUT/MEMOIZE + BINGET).

Атака: строки 'module' и 'name' кладутся в memo через PUT/BINPUT/MEMOIZE,
удаляются со стека через POP, затем извлекаются через GET/BINGET прямо перед
STACK_GLOBAL. До исправления scanner не вёл состояние memo и получал int
вместо str — isinstance-проверка проваливалась, глобал не регистрировался.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from poison_check.scanners.pickle_scanner import PickleScanner


@pytest.fixture()
def scanner() -> PickleScanner:
    """Экземпляр PickleScanner для тестов."""
    return PickleScanner()


# ---------------------------------------------------------------------------
# Вспомогательные построители payload-ов (без pickle.dumps на злых объектах)
# ---------------------------------------------------------------------------

def _make_binput_pop_binget_payload() -> bytes:
    """Протокол 2: BINUNICODE → BINPUT → POP → BINGET → STACK_GLOBAL."""
    return (
        b"\x80\x02"                      # PROTO 2
        b"X\x02\x00\x00\x00os"           # BINUNICODE 'os'
        b"q\x00"                         # BINPUT 0  (memo[0] = 'os')
        b"0"                             # POP
        b"X\x06\x00\x00\x00system"       # BINUNICODE 'system'
        b"q\x01"                         # BINPUT 1  (memo[1] = 'system')
        b"0"                             # POP
        b"h\x00"                         # BINGET 0  -> 'os'
        b"h\x01"                         # BINGET 1  -> 'system'
        b"\x93"                          # STACK_GLOBAL
        b")"                             # EMPTY_TUPLE
        b"R"                             # REDUCE
        b"."                             # STOP
    )


def _make_memoize_pop_binget_payload() -> bytes:
    """Протокол 4: SHORT_BINUNICODE → MEMOIZE → POP → BINGET → STACK_GLOBAL."""
    inner = (
        b"\x8c\x02os"                    # SHORT_BINUNICODE 'os'
        b"\x94"                          # MEMOIZE  (memo[0] = 'os')
        b"0"                             # POP
        b"\x8c\x06system"               # SHORT_BINUNICODE 'system'
        b"\x94"                          # MEMOIZE  (memo[1] = 'system')
        b"0"                             # POP
        b"h\x00"                         # BINGET 0  -> 'os'
        b"h\x01"                         # BINGET 1  -> 'system'
        b"\x93"                          # STACK_GLOBAL
        b")"                             # EMPTY_TUPLE
        b"R"                             # REDUCE
        b"."                             # STOP
    )
    return b"\x80\x04\x95" + struct.pack("<Q", len(inner)) + inner


def _make_long_binget_payload() -> bytes:
    """Протокол 2: BINPUT → POP → LONG_BINGET → STACK_GLOBAL.

    LONG_BINGET = j (0x6a), LONG_BINPUT = r (0x72) — разные opcodes.
    """
    return (
        b"\x80\x02"
        b"X\x02\x00\x00\x00os"
        b"q\x00"                         # BINPUT 0
        b"0"                             # POP
        b"X\x06\x00\x00\x00system"
        b"q\x01"                         # BINPUT 1
        b"0"                             # POP
        b"j\x00\x00\x00\x00"             # LONG_BINGET 0  (j = 0x6a)
        b"j\x01\x00\x00\x00"             # LONG_BINGET 1
        b"\x93"                          # STACK_GLOBAL
        b")"
        b"R"
        b"."
    )


def _make_put_get_payload() -> bytes:
    """Протокол 0: STRING → PUT → POP → GET → STACK_GLOBAL."""
    return (
        b"\x80\x00"                      # PROTO 0
        b"Vos\n"                         # UNICODE 'os'
        b"p0\n"                          # PUT '0'
        b"0"                             # POP
        b"Vsystem\n"                     # UNICODE 'system'
        b"p1\n"                          # PUT '1'
        b"0"                             # POP
        b"g0\n"                          # GET '0'
        b"g1\n"                          # GET '1'
        b"\x93"                          # STACK_GLOBAL
        b")"
        b"R"
        b"."
    )


# ---------------------------------------------------------------------------
# Тесты детекции bypass-паттернов
# ---------------------------------------------------------------------------

class TestMemoBypassDetection:
    """STACK_GLOBAL через memo должен детектироваться как (os, system)."""

    def test_fixture_file(self, scanner: PickleScanner, tmp_path: Path) -> None:
        """Fixture-файл из tests/fixtures/malicious/ детектируется корректно."""
        fixture = Path(__file__).parent / "fixtures" / "malicious" / "stack_global_via_memo.pkl"
        result = scanner.scan(fixture)
        assert result.globals is not None
        assert ("os", "system") in result.globals

    def test_binput_pop_binget(self, scanner: PickleScanner, tmp_path: Path) -> None:
        """BINPUT + POP + BINGET (протокол 2)."""
        path = tmp_path / "bypass_binput.pkl"
        path.write_bytes(_make_binput_pop_binget_payload())
        result = scanner.scan(path)
        assert result.globals is not None
        assert ("os", "system") in result.globals

    def test_memoize_pop_binget(self, scanner: PickleScanner, tmp_path: Path) -> None:
        """MEMOIZE + POP + BINGET (протокол 4)."""
        path = tmp_path / "bypass_memoize.pkl"
        path.write_bytes(_make_memoize_pop_binget_payload())
        result = scanner.scan(path)
        assert result.globals is not None
        assert ("os", "system") in result.globals

    def test_long_binget(self, scanner: PickleScanner, tmp_path: Path) -> None:
        """BINPUT + POP + LONG_BINGET (4-байтовый индекс)."""
        path = tmp_path / "bypass_long_binget.pkl"
        path.write_bytes(_make_long_binget_payload())
        result = scanner.scan(path)
        assert result.globals is not None
        assert ("os", "system") in result.globals

    def test_put_get_proto0(self, scanner: PickleScanner, tmp_path: Path) -> None:
        """PUT + POP + GET (протокол 0, текстовые индексы)."""
        path = tmp_path / "bypass_put_get.pkl"
        path.write_bytes(_make_put_get_payload())
        result = scanner.scan(path)
        assert result.globals is not None
        assert ("os", "system") in result.globals


# ---------------------------------------------------------------------------
# Регрессия: нормальные случаи не сломаны
# ---------------------------------------------------------------------------

class TestMemoNoRegression:
    """Обычный STACK_GLOBAL (строки напрямую на стеке) работает как раньше."""

    def test_stack_global_direct(self, scanner: PickleScanner, tmp_path: Path) -> None:
        """Строки на стеке без memo — детектируется."""
        payload = (
            b"\x80\x02"
            b"X\x02\x00\x00\x00os"
            b"X\x06\x00\x00\x00system"
            b"\x93"                      # STACK_GLOBAL
            b")"
            b"R"
            b"."
        )
        path = tmp_path / "direct.pkl"
        path.write_bytes(payload)
        result = scanner.scan(path)
        assert result.globals is not None
        assert ("os", "system") in result.globals

    def test_global_opcode(self, scanner: PickleScanner, tmp_path: Path) -> None:
        """Обычный GLOBAL opcode (не STACK_GLOBAL) работает как раньше."""
        payload = b"\x80\x02" b"cos\nsystem\n" b")" b"R" b"."
        path = tmp_path / "global_opcode.pkl"
        path.write_bytes(payload)
        result = scanner.scan(path)
        assert result.globals is not None
        assert ("os", "system") in result.globals

    def test_clean_model_no_false_positive(
        self, scanner: PickleScanner, tmp_path: Path
    ) -> None:
        """Pickle с memo, но без опасных globals — 0 issues после детектора."""
        # Кладём безобидную строку в memo и достаём её — не должно быть FP.
        payload = (
            b"\x80\x02"
            b"X\x05\x00\x00\x00hello"   # BINUNICODE 'hello'
            b"q\x00"                     # BINPUT 0
            b"0"                         # POP
            b"h\x00"                     # BINGET 0
            b"."                         # STOP
        )
        path = tmp_path / "clean_memo.pkl"
        path.write_bytes(payload)
        result = scanner.scan(path)
        # globals может быть None или пустым — главное что os/system нет
        if result.globals:
            assert ("os", "system") not in result.globals

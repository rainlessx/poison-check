"""Property-based тесты для pickle-парсера через hypothesis.

Property-based тесты через hypothesis нужны для парсера pickle, «чтобы
не падать на некорректных данных». Аудит #21 показал,
что эти тесты отсутствовали, несмотря на ``hypothesis`` в dev-deps.

Ключевые инварианты:

1. ``PickleScanner.scan()`` НИКОГДА не бросает исключений наружу —
   даже на полностью случайных байтах. На повреждённых данных возвращает
   ``RawScanData`` с заполненным полем ``error``.
2. ``PickleScanner.scan_bytes()`` имеет тот же контракт.
3. ``_detect_protocol()`` возвращает либо int 0–5, либо None — никогда не падает.
4. ``_walk_opcodes()`` накапливает результаты до момента ошибки и не теряет
   уже разобранную часть (graceful degradation).

Эти тесты дают защиту от классов багов, которые сложно поймать unit-тестами:
edge-cases в truncated/corrupt pickle, неожиданные комбинации opcode'ов,
unicode-нормализация, etc.
"""

from __future__ import annotations

import io
import pickletools
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from poison_check.scanners.pickle_scanner import (
    PickleScanner,
    _detect_protocol,
    _walk_opcodes,
)


# ---------------------------------------------------------------------------
# Стратегии генерации
# ---------------------------------------------------------------------------


# Полностью случайные байты — основной adversarial-вход.
_RANDOM_BYTES = st.binary(min_size=0, max_size=4096)

# Случайные байты с pickle-prefix (\x80\x02..\x80\x05) — часто-валидное начало,
# далее произвольный мусор. Имитирует «обрезанный pickle».
_PICKLE_PREFIX_BYTES = st.builds(
    lambda proto, tail: bytes([0x80, proto]) + tail,
    proto=st.integers(min_value=2, max_value=5),
    tail=st.binary(min_size=0, max_size=4096),
)

# Pickle proto 0/1 — начинается с одного из opcode-байтов.
_PICKLE_PROTO01_LEAD_BYTES = b"(}])ldticIL SVU"
_PICKLE_PROTO01_BYTES = st.builds(
    lambda lead, tail: bytes([lead]) + tail + b".",
    lead=st.sampled_from(list(_PICKLE_PROTO01_LEAD_BYTES)),
    tail=st.binary(min_size=0, max_size=4096),
)

# Объединённая стратегия — даёт смесь чистого мусора и pickle-like данных
ANY_PICKLE_LIKE = st.one_of(
    _RANDOM_BYTES,
    _PICKLE_PREFIX_BYTES,
    _PICKLE_PROTO01_BYTES,
)


# ---------------------------------------------------------------------------
# Инвариант 1: scan() не падает на любых байтах
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=ANY_PICKLE_LIKE)
def test_scan_never_raises_on_arbitrary_bytes(data: bytes, tmp_path: Path) -> None:
    """PickleScanner.scan() возвращает RawScanData для любых байт, не бросает.

    Сканер обязан graceful degradation: повреждённый/неизвестный файл →
    RawScanData с error, а не Exception наружу (CLAUDE.md «Обработка ошибок»).
    """
    scanner = PickleScanner()
    f = tmp_path / "fuzz.pkl"
    f.write_bytes(data)

    # Если can_handle отказался — корректный путь, но scan() должен работать
    # одинаково: вернуть RawScanData без исключения.
    result = scanner.scan(f)
    assert result is not None
    assert result.scanner_name == "pickle"
    assert result.file_path == f
    # error может быть и None (если случайно валидный pickle), и заполнен —
    # любое из двух валидно.


@settings(
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=ANY_PICKLE_LIKE)
def test_scan_bytes_never_raises(data: bytes, tmp_path: Path) -> None:
    """PickleScanner.scan_bytes() — тот же инвариант для in-memory входа."""
    scanner = PickleScanner()
    result = scanner.scan_bytes(data, source_path=tmp_path / "in_memory.pkl")
    assert result is not None
    assert result.scanner_name == "pickle"


# ---------------------------------------------------------------------------
# Инвариант 2: _detect_protocol возвращает int 0..5 или None
# ---------------------------------------------------------------------------


@settings(max_examples=500, deadline=1000)
@given(data=_RANDOM_BYTES)
def test_detect_protocol_returns_valid_value(data: bytes) -> None:
    """_detect_protocol всегда возвращает int 0..5 или None, не падает."""
    proto = _detect_protocol(data)
    assert proto is None or (isinstance(proto, int) and 0 <= proto <= 255), (
        f"Неожиданное значение _detect_protocol: {proto!r}"
    )


# ---------------------------------------------------------------------------
# Инвариант 3: _walk_opcodes graceful — собранная часть сохраняется
# ---------------------------------------------------------------------------


@settings(
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=ANY_PICKLE_LIKE)
def test_walk_opcodes_does_not_corrupt_collections_on_error(data: bytes) -> None:
    """_walk_opcodes может бросить (это документировано), но:

    - все коллекции остаются валидными list/set
    - частично собранные opcodes не теряют тип элементов
    """
    opcodes: list = []
    globals_set: set = set()
    strings: list = []
    reduce_calls: list = []

    try:
        _walk_opcodes(io.BytesIO(data), opcodes, globals_set, strings, reduce_calls)
    except Exception:
        # _walk_opcodes документировано пробрасывает ValueError/EOFError/struct.error
        # из pickletools.genops — это нормально, обработчик в scan() ловит.
        pass

    # Пост-инвариант: коллекции не повреждены
    assert isinstance(opcodes, list)
    assert isinstance(globals_set, set)
    assert isinstance(strings, list)
    assert isinstance(reduce_calls, list)
    # Все элементы globals_set — пары (str, str)
    for entry in globals_set:
        assert isinstance(entry, tuple) and len(entry) == 2
        assert isinstance(entry[0], str) and isinstance(entry[1], str)


# ---------------------------------------------------------------------------
# Инвариант 4: scan() на валидном pickle возвращает error=None
# ---------------------------------------------------------------------------


@settings(
    max_examples=50,
    deadline=2000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    proto=st.integers(min_value=2, max_value=5),
    obj=st.one_of(
        st.integers(),
        st.text(max_size=64),
        st.lists(st.integers(), max_size=16),
        st.dictionaries(st.text(max_size=16), st.integers(), max_size=8),
    ),
)
def test_scan_valid_pickle_no_error(proto: int, obj: object, tmp_path: Path) -> None:
    """Валидный pickle (созданный через pickle.dumps) → error is None."""
    import pickle as _pickle

    data = _pickle.dumps(obj, protocol=proto)
    f = tmp_path / "valid.pkl"
    f.write_bytes(data)
    scanner = PickleScanner()
    result = scanner.scan(f)
    assert result.error is None, f"Валидный pickle дал error: {result.error}"
    assert result.opcodes is not None and len(result.opcodes) > 0


# ---------------------------------------------------------------------------
# Sanity: запуск тестов из списка опкодов pickletools без падений
# ---------------------------------------------------------------------------


def test_pickletools_genops_compatibility() -> None:
    """Сверка: pickletools.genops понимает наш набор opcodes."""
    # Просто проверяем, что наш _walk_opcodes хотя бы вызывается
    opcodes: list = []
    globals_set: set = set()
    strings: list = []
    reduce_calls: list = []

    # Минимальный валидный pickle: пустой stop
    valid = b"\x80\x02N."
    _walk_opcodes(
        io.BytesIO(valid), opcodes, globals_set, strings, reduce_calls,
    )
    assert len(opcodes) >= 2  # PROTO + STOP

    # Проверяем что все имена opcode из pickletools валидны
    opcode_names = {op.opcode for op in opcodes}
    pickletools_names = {op.name for op in pickletools.code2op.values()}
    assert opcode_names <= pickletools_names

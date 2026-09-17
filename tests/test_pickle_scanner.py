"""Тесты для PickleScanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.scanners.pickle_scanner import PickleScanner, _detect_protocol

FIXTURES = Path(__file__).parent / "fixtures"
SAFE_DIR = FIXTURES / "safe"
MALICIOUS_DIR = FIXTURES / "malicious"

DANGEROUS_MODULES = {"os", "subprocess", "builtins", "posix"}
DANGEROUS_NAMES = {"system", "popen", "eval", "exec", "Popen"}

# Пары (module, name), хотя бы одна из которых должна быть в globals вредоносного payload
_DANGEROUS_PAIRS = frozenset(
    {
        ("os", "system"),
        ("subprocess", "Popen"),
        ("builtins", "eval"),
        ("builtins", "exec"),
    }
)

_PAYLOAD_FILES = [
    MALICIOUS_DIR / "payload_01_os_system.pkl",
    MALICIOUS_DIR / "payload_02_subprocess_popen.pkl",
    MALICIOUS_DIR / "payload_03_builtins_eval.pkl",
    MALICIOUS_DIR / "payload_04_builtins_exec.pkl",
    MALICIOUS_DIR / "payload_05_torch.pt",
    MALICIOUS_DIR / "payload_06_numpy.npz",
]


@pytest.fixture(scope="module")
def scanner() -> PickleScanner:
    """Экземпляр PickleScanner для всех тестов."""
    return PickleScanner()


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


def test_can_handle_simple_list(scanner: PickleScanner) -> None:
    """Файл с правильным расширением и magic bytes → True."""
    path = SAFE_DIR / "simple_list.pkl"
    assert scanner.can_handle(path) is True


def test_can_handle_simple_dict(scanner: PickleScanner) -> None:
    """Файл с правильным расширением и magic bytes → True."""
    path = SAFE_DIR / "simple_dict.pkl"
    assert scanner.can_handle(path) is True


def test_can_handle_malicious(scanner: PickleScanner) -> None:
    """Вредоносный pkl-файл распознаётся как обрабатываемый."""
    path = MALICIOUS_DIR / "os_system.pkl"
    assert scanner.can_handle(path) is True


def test_can_handle_wrong_extension(scanner: PickleScanner, tmp_path: Path) -> None:
    """Файл с чужим расширением → False, даже если содержимое pickle."""
    fake = tmp_path / "model.safetensors"
    fake.write_bytes(b"\x80\x02N.")
    assert scanner.can_handle(fake) is False


def test_can_handle_wrong_magic(scanner: PickleScanner, tmp_path: Path) -> None:
    """Файл .pkl с неверными magic bytes → False."""
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"\xff\xfeNot a pickle")
    assert scanner.can_handle(bad) is False


# ---------------------------------------------------------------------------
# safe файлы — globals не должны содержать опасные модули
# ---------------------------------------------------------------------------


def test_safe_list_no_dangerous_globals(scanner: PickleScanner) -> None:
    """simple_list.pkl не содержит глобалов из опасных модулей."""
    result = scanner.scan(SAFE_DIR / "simple_list.pkl")
    assert result.error is None
    if result.globals:
        for module, _name in result.globals:
            assert module not in DANGEROUS_MODULES, (
                f"Неожиданный опасный модуль в безопасном файле: {module}"
            )


def test_safe_dict_no_dangerous_globals(scanner: PickleScanner) -> None:
    """simple_dict.pkl не содержит глобалов из опасных модулей."""
    result = scanner.scan(SAFE_DIR / "simple_dict.pkl")
    assert result.error is None
    if result.globals:
        for module, _name in result.globals:
            assert module not in DANGEROUS_MODULES


def test_safe_list_has_opcodes(scanner: PickleScanner) -> None:
    """Безопасный файл содержит список opcodes."""
    result = scanner.scan(SAFE_DIR / "simple_list.pkl")
    assert result.opcodes is not None
    assert len(result.opcodes) > 0


def test_safe_list_protocol_detected(scanner: PickleScanner) -> None:
    """Версия протокола определяется для безопасного файла."""
    result = scanner.scan(SAFE_DIR / "simple_list.pkl")
    assert result.metadata is not None
    assert "protocol" in result.metadata


# ---------------------------------------------------------------------------
# malicious/os_system.pkl — должен содержать ("os", "system")
# ---------------------------------------------------------------------------


def test_malicious_globals_contains_os_system(scanner: PickleScanner) -> None:
    """Вредоносный файл содержит глобал ("os", "system") в RawScanData.globals."""
    result = scanner.scan(MALICIOUS_DIR / "os_system.pkl")
    assert result.globals is not None, "globals не должны быть None для вредоносного файла"
    assert ("os", "system") in result.globals


def test_malicious_reduce_calls_not_empty(scanner: PickleScanner) -> None:
    """Вредоносный файл содержит хотя бы один ReduceCall."""
    result = scanner.scan(MALICIOUS_DIR / "os_system.pkl")
    assert result.reduce_calls is not None
    assert len(result.reduce_calls) > 0


def test_malicious_reduce_call_module(scanner: PickleScanner) -> None:
    """ReduceCall для os.system корректно определяет module='os', name='system'."""
    result = scanner.scan(MALICIOUS_DIR / "os_system.pkl")
    assert result.reduce_calls is not None
    modules = {(rc.module, rc.name) for rc in result.reduce_calls}
    assert ("os", "system") in modules


def test_malicious_has_opcodes(scanner: PickleScanner) -> None:
    """Вредоносный файл содержит GLOBAL и REDUCE в списке opcodes."""
    result = scanner.scan(MALICIOUS_DIR / "os_system.pkl")
    assert result.opcodes is not None
    opcode_names = {op.opcode for op in result.opcodes}
    assert "GLOBAL" in opcode_names
    assert "REDUCE" in opcode_names


def test_malicious_strings_contain_command(scanner: PickleScanner) -> None:
    """Строки из вредоносного файла содержат команду 'echo pwned'."""
    result = scanner.scan(MALICIOUS_DIR / "os_system.pkl")
    assert result.strings is not None
    values = {s.value for s in result.strings}
    assert "echo pwned" in values


# ---------------------------------------------------------------------------
# scan_bytes — вредоносный payload через байты
# ---------------------------------------------------------------------------


def test_scan_bytes_malicious(scanner: PickleScanner, tmp_path: Path) -> None:
    """scan_bytes правильно парсит вредоносный payload."""
    data = (MALICIOUS_DIR / "os_system.pkl").read_bytes()
    result = scanner.scan_bytes(data, source_path=tmp_path / "test.pkl")
    assert result.globals is not None
    assert ("os", "system") in result.globals
    assert result.reduce_calls is not None


# ---------------------------------------------------------------------------
# повреждённые / случайные байты — сканер не падает
# ---------------------------------------------------------------------------


def test_corrupted_bytes_no_exception(scanner: PickleScanner, tmp_path: Path) -> None:
    """Случайные байты → RawScanData без необработанного исключения."""
    corrupted = tmp_path / "corrupted.pkl"
    corrupted.write_bytes(b"\x80\x04\xff\xee\xdd\xcc\xbb\xaa" * 32)
    result = scanner.scan(corrupted)
    assert result is not None


def test_corrupted_file_returns_raw_scan_data(
    scanner: PickleScanner, tmp_path: Path
) -> None:
    """Повреждённый файл возвращает RawScanData, а не бросает исключение."""
    corrupted = tmp_path / "garbage.pkl"
    corrupted.write_bytes(bytes(range(256)) * 4)
    result = scanner.scan(corrupted)
    # Поле file_path всегда заполнено
    assert result.file_path == corrupted
    assert result.scanner_name == "pickle"


def test_truncated_pickle_no_exception(scanner: PickleScanner, tmp_path: Path) -> None:
    """Обрезанный pickle (неполный) не роняет сканер."""
    truncated = tmp_path / "truncated.pkl"
    truncated.write_bytes(b"\x80\x04\x95\x10\x00")  # PROTO + неполный FRAME
    result = scanner.scan(truncated)
    assert result is not None


def test_empty_file_no_exception(scanner: PickleScanner, tmp_path: Path) -> None:
    """Пустой файл с расширением .pkl не роняет сканер."""
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(b"")
    result = scanner.scan(empty)
    assert result is not None
    assert result.scanner_name == "pickle"


# ---------------------------------------------------------------------------
# метаданные и базовые поля
# ---------------------------------------------------------------------------


def test_file_size_set(scanner: PickleScanner) -> None:
    """Поле file_size корректно заполняется."""
    path = SAFE_DIR / "simple_list.pkl"
    result = scanner.scan(path)
    assert result.file_size == path.stat().st_size


def test_file_hash_set(scanner: PickleScanner) -> None:
    """Поля хешей заполняются для существующего файла.

    Аудит #18: MD5 больше не считается по умолчанию (тройной IO).
    """
    result = scanner.scan(SAFE_DIR / "simple_dict.pkl")
    assert "sha256" in result.file_hash
    assert "sha512" in result.file_hash
    assert len(result.file_hash["sha256"]) == 64
    assert len(result.file_hash["sha512"]) == 128


def test_scanner_name(scanner: PickleScanner) -> None:
    """scanner_name всегда 'pickle'."""
    result = scanner.scan(SAFE_DIR / "simple_list.pkl")
    assert result.scanner_name == "pickle"


# ---------------------------------------------------------------------------
# Параметризованный тест: все 6 базовых payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload_path",
    _PAYLOAD_FILES,
    ids=[p.name for p in _PAYLOAD_FILES],
)
def test_all_payloads_detected(scanner: PickleScanner, payload_path: Path) -> None:
    """Каждый из 6 вредоносных payload детектируется PickleScanner.

    Проверяет: globals содержит хотя бы одну из опасных пар,
    reduce_calls не пуст.
    """
    result = scanner.scan(payload_path)

    assert result.globals is not None, (
        f"globals не должны быть None для {payload_path.name}"
    )
    assert result.reduce_calls is not None, (
        f"reduce_calls не должны быть None для {payload_path.name}"
    )
    assert len(result.reduce_calls) > 0, (
        f"reduce_calls пусты для {payload_path.name}"
    )
    detected = result.globals & _DANGEROUS_PAIRS
    assert detected, (
        f"Ни одна из {set(_DANGEROUS_PAIRS)} не найдена в globals={result.globals} "
        f"для {payload_path.name}"
    )


# ---------------------------------------------------------------------------
# Тест производительности: 10 МБ < 5 секунд
# ---------------------------------------------------------------------------


def test_performance_10mb_scan(scanner: PickleScanner, tmp_path: Path) -> None:
    """Сканирование pickle-файла ~10 МБ завершается менее чем за 5 секунд."""
    import pickle
    import time

    large_file = tmp_path / "large_10mb.pkl"
    # ~10 МБ: 100 000 строк по 100 символов (≈100K опкодов)
    large_file.write_bytes(pickle.dumps(["a" * 100] * 100_000, protocol=4))

    start = time.monotonic()
    result = scanner.scan(large_file)
    elapsed = time.monotonic() - start

    assert result is not None
    assert elapsed < 5.0, f"Сканирование заняло {elapsed:.2f}с (предел 5с)"


# ---------------------------------------------------------------------------
# Регрессия аудита #13: _detect_protocol работает для proto 0/1
# ---------------------------------------------------------------------------


def test_detect_protocol_proto2_returns_2() -> None:
    """\\x80\\x02 → protocol 2."""
    assert _detect_protocol(b"\x80\x02.") == 2


def test_detect_protocol_proto5_returns_5() -> None:
    """\\x80\\x05 → protocol 5."""
    assert _detect_protocol(b"\x80\x05.") == 5


def test_detect_protocol_proto0_returns_0() -> None:
    """Pickle proto 0 (без \\x80) → 0 по эвристике первого opcode + STOP."""
    # GLOBAL os.system → REDUCE → STOP, классический proto 0
    proto0 = b"cos\nsystem\n(tR."
    assert _detect_protocol(proto0) == 0


def test_detect_protocol_garbage_returns_none() -> None:
    """Произвольный мусор → None."""
    assert _detect_protocol(b"\x00\x00\x00\x00") is None
    assert _detect_protocol(b"") is None


def test_detect_protocol_pickle_lead_without_stop_returns_none() -> None:
    """Опкод-подобный лид без STOP-байта → не считаем pickle."""
    # 'c' — GLOBAL opcode, но без '.' это не валидный pickle
    assert _detect_protocol(b"cabc") is None

"""Генерация тестовых fixture-файлов для PickleScanner и JoblibScanner.

Запуск: python tests/fixtures/generate_fixtures.py
Безопасные файлы создаются через pickle.dumps / joblib.dump.
Вредоносные — вручную, через конструкцию opcode-потока,
НЕ через pickle.dumps злонамеренных объектов.

Для joblib-фикстур используется joblib для СОЗДАНИЯ файлов (это допустимо),
но внутренний pickle-payload конструируется вручную через opcode-конструкцию —
именно так поступает реальный атакующий: создаёт joblib-обёртку с вредоносным
pickle-потоком внутри.
"""

import bz2
import gzip
import io
import lzma
import pickle
import struct
import zipfile
import zlib
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
SAFE_DIR = FIXTURES_DIR / "safe"
MALICIOUS_DIR = FIXTURES_DIR / "malicious"


def _make_safe_fixtures() -> None:
    """Создаёт безопасные pickle-файлы через pickle.dumps стандартных объектов."""
    SAFE_DIR.mkdir(parents=True, exist_ok=True)

    (SAFE_DIR / "simple_list.pkl").write_bytes(
        pickle.dumps([1, 2, 3], protocol=2)
    )
    (SAFE_DIR / "simple_dict.pkl").write_bytes(
        pickle.dumps({"key": "value"}, protocol=2)
    )


def _make_safe_joblib_fixtures() -> None:
    """Создаёт безопасные joblib-файлы.

    Использует joblib.dump для СОЗДАНИЯ файлов — это допустимо
    (мы сохраняем только безопасные объекты: список и словарь).
    Joblib можно использовать для генерации тестовых данных, но НИКОГДА
    для загрузки пользовательских файлов при сканировании.
    """
    try:
        import joblib  # type: ignore[import-untyped]
    except ImportError:
        print("  [skip] joblib не установлен, пропускаем joblib-фикстуры")
        return

    SAFE_DIR.mkdir(parents=True, exist_ok=True)

    # Файл без сжатия (raw pickle)
    joblib.dump([1, 2, 3], str(SAFE_DIR / "simple_list.joblib"))

    # Файл с zlib-сжатием (compress=3 → zlib level 3)
    joblib.dump({"key": "value", "numbers": [1, 2, 3]},
                str(SAFE_DIR / "sklearn_model.joblib"),
                compress=3)

    # Файл с gzip-сжатием
    joblib.dump([1, 2, 3],
                str(SAFE_DIR / "simple_list_gzip.joblib"),
                compress=("gzip", 3))


# ---------------------------------------------------------------------------
# Вспомогательные opcode-константы
# ---------------------------------------------------------------------------

_PROTO2 = b"\x80\x02"   # PROTO 2
_EMPTY_TUPLE = b")"      # EMPTY_TUPLE (opcode 0x29)
_MARK = b"("             # MARK
_TUPLE = b"t"            # TUPLE (создаёт кортеж из элементов до MARK)
_REDUCE = b"R"           # REDUCE
_STOP = b"."             # STOP


# ---------------------------------------------------------------------------
# Payload 1 — os.system через GLOBAL + пустой MARK + REDUCE
# ---------------------------------------------------------------------------

def _build_os_system_pickle() -> bytes:
    """payload_1: os.system через GLOBAL + пустой MARK + REDUCE.

    Opcode-поток:
      PROTO 2
      GLOBAL "os system"
      MARK
      TUPLE   → пустой кортеж (аргументы)
      REDUCE  → вызов os.system()
      STOP
    """
    return (
        _PROTO2
        + b"cos\nsystem\n"  # GLOBAL os.system
        + _MARK
        + _TUPLE            # пустой кортеж (аргументы с MARK)
        + _REDUCE
        + _STOP
    )


# ---------------------------------------------------------------------------
# Payload 2 — subprocess.Popen через GLOBAL + REDUCE
# ---------------------------------------------------------------------------

def _build_subprocess_popen_pickle() -> bytes:
    """payload_2: subprocess.Popen через GLOBAL + EMPTY_TUPLE + REDUCE.

    Opcode-поток:
      PROTO 2
      GLOBAL "subprocess Popen"
      EMPTY_TUPLE
      REDUCE
      STOP
    """
    return (
        _PROTO2
        + b"csubprocess\nPopen\n"  # GLOBAL subprocess.Popen
        + _EMPTY_TUPLE
        + _REDUCE
        + _STOP
    )


# ---------------------------------------------------------------------------
# Payload 3 — builtins.eval через GLOBAL + REDUCE
# ---------------------------------------------------------------------------

def _build_builtins_eval_pickle() -> bytes:
    """payload_3: builtins.eval через GLOBAL + EMPTY_TUPLE + REDUCE.

    Opcode-поток:
      PROTO 2
      GLOBAL "builtins eval"
      EMPTY_TUPLE
      REDUCE
      STOP
    """
    return (
        _PROTO2
        + b"cbuiltins\neval\n"  # GLOBAL builtins.eval
        + _EMPTY_TUPLE
        + _REDUCE
        + _STOP
    )


# ---------------------------------------------------------------------------
# Payload 4 — builtins.exec через GLOBAL + REDUCE
# ---------------------------------------------------------------------------

def _build_builtins_exec_pickle() -> bytes:
    """payload_4: builtins.exec через GLOBAL + EMPTY_TUPLE + REDUCE.

    Opcode-поток:
      PROTO 2
      GLOBAL "builtins exec"
      EMPTY_TUPLE
      REDUCE
      STOP
    """
    return (
        _PROTO2
        + b"cbuiltins\nexec\n"  # GLOBAL builtins.exec
        + _EMPTY_TUPLE
        + _REDUCE
        + _STOP
    )


# ---------------------------------------------------------------------------
# Payload 5 — PyTorch .pt: ZIP с data.pkl внутри
# ---------------------------------------------------------------------------

def _build_pytorch_pt(pkl_bytes: bytes) -> bytes:
    """payload_5: упаковывает pickle-payload в PyTorch .pt (ZIP-архив с archive/data.pkl).

    Структура реального PyTorch .pt файла:
      archive.zip
        └── archive/data.pkl
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", pkl_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Payload 6 — NumPy .npz: ZIP с array_0.npy, содержащим pickle-payload
# ---------------------------------------------------------------------------

def _build_numpy_npz(pkl_bytes: bytes) -> bytes:
    """payload_6: вредоносный .npz (ZIP) с pickle-payload в array_0.npy.

    В атаке payload embedded в metadata записи array_0.npy:
    вместо легитимных numpy-байт entry содержит опасный pickle-поток.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("array_0.npy", pkl_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Joblib-обёртка для вредоносного pickle-payload
# ---------------------------------------------------------------------------

def make_malicious_joblib(payload_bytes: bytes, compress_method: str = "none") -> bytes:
    """Создаёт joblib-файл с произвольным pickle-payload внутри.

    Это реалистичный attack vector: атакующий создаёт joblib-обёртку,
    содержащую вредоносный pickle-поток, и распространяет файл как
    «обычную sklearn-модель».

    Args:
        payload_bytes: Произвольный pickle-поток (вредоносный).
        compress_method: Метод сжатия — 'none', 'zlib', 'gzip', 'bz2',
                         'lzma', 'xz'. По умолчанию без сжатия.

    Returns:
        bytes: Корректный joblib-файл с вредоносным payload внутри.
    """
    if compress_method == "none":
        # Joblib без сжатия = raw pickle
        return payload_bytes

    if compress_method == "zlib":
        return zlib.compress(payload_bytes, 3)

    if compress_method == "gzip":
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=3) as gf:
            gf.write(payload_bytes)
        return buf.getvalue()

    if compress_method == "bz2":
        return bz2.compress(payload_bytes, 3)

    if compress_method == "lzma":
        return lzma.compress(payload_bytes, format=lzma.FORMAT_ALONE)

    if compress_method == "xz":
        return lzma.compress(payload_bytes, format=lzma.FORMAT_XZ)

    raise ValueError(f"Неподдерживаемый метод: {compress_method!r}")


def _make_malicious_joblib_fixtures() -> None:
    """Создаёт вредоносные joblib-файлы с разными методами сжатия.

    Вредоносный pickle-payload конструируется вручную (opcode-конструкция),
    затем упаковывается в joblib-контейнер с разными методами сжатия.
    Это воспроизводит реальный attack vector.
    """
    MALICIOUS_DIR.mkdir(parents=True, exist_ok=True)

    os_pkl = _build_os_system_pickle()

    # Без сжатия (raw pickle)
    (MALICIOUS_DIR / "payload_joblib_none.joblib").write_bytes(
        make_malicious_joblib(os_pkl, "none")
    )

    # Zlib-сжатие
    (MALICIOUS_DIR / "payload_joblib_zlib.joblib").write_bytes(
        make_malicious_joblib(os_pkl, "zlib")
    )

    # Gzip-сжатие
    (MALICIOUS_DIR / "payload_joblib_gzip.joblib").write_bytes(
        make_malicious_joblib(os_pkl, "gzip")
    )

    # BZ2-сжатие
    (MALICIOUS_DIR / "payload_joblib_bz2.joblib").write_bytes(
        make_malicious_joblib(os_pkl, "bz2")
    )

    # LZMA-сжатие
    (MALICIOUS_DIR / "payload_joblib_lzma.joblib").write_bytes(
        make_malicious_joblib(os_pkl, "lzma")
    )

    # XZ-сжатие
    (MALICIOUS_DIR / "payload_joblib_xz.joblib").write_bytes(
        make_malicious_joblib(os_pkl, "xz")
    )


# ---------------------------------------------------------------------------
# Старый builder (сохранён для обратной совместимости с test_pickle_scanner.py)
# ---------------------------------------------------------------------------

def _build_os_system_pickle_with_arg() -> bytes:
    """Вредоносный pickle с os.system и аргументом-командой (для legacy-тестов)."""
    cmd = b"echo pwned"
    return (
        b"\x80\x02"
        b"cos\nsystem\n"
        b"("
        + b"\x8c" + bytes([len(cmd)]) + cmd
        + b"\x85"
        + b"R"
        + b"."
    )


# ---------------------------------------------------------------------------
# Генерация файлов
# ---------------------------------------------------------------------------

def _make_malicious_fixtures() -> None:
    """Создаёт вредоносные файлы через ручную opcode-конструкцию."""
    MALICIOUS_DIR.mkdir(parents=True, exist_ok=True)

    # Legacy os_system.pkl — сохранён для существующих тестов
    (MALICIOUS_DIR / "os_system.pkl").write_bytes(_build_os_system_pickle_with_arg())

    # 6 базовых payload
    os_pkl = _build_os_system_pickle()
    (MALICIOUS_DIR / "payload_01_os_system.pkl").write_bytes(os_pkl)
    (MALICIOUS_DIR / "payload_02_subprocess_popen.pkl").write_bytes(
        _build_subprocess_popen_pickle()
    )
    (MALICIOUS_DIR / "payload_03_builtins_eval.pkl").write_bytes(
        _build_builtins_eval_pickle()
    )
    (MALICIOUS_DIR / "payload_04_builtins_exec.pkl").write_bytes(
        _build_builtins_exec_pickle()
    )
    (MALICIOUS_DIR / "payload_05_torch.pt").write_bytes(_build_pytorch_pt(os_pkl))
    (MALICIOUS_DIR / "payload_06_numpy.npz").write_bytes(_build_numpy_npz(os_pkl))


# ---------------------------------------------------------------------------
# PyTorch-фикстуры: создаём .pt файл как ZIP без импорта torch
# ---------------------------------------------------------------------------

def make_pytorch_zip(pickle_bytes: bytes) -> bytes:
    """Создаёт .pt-файл как ZIP без импорта torch.

    Реализует минимальную структуру реального PyTorch ZIP-архива:
      archive/data.pkl  ← pickle-payload с весами / вредоносным кодом

    Используется для создания тестовых fixture без зависимости от torch.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", pickle_bytes)
    return buf.getvalue()


def _build_safe_pytorch_pickle() -> bytes:
    """Безопасный pickle имитирующий torch.nn.Linear (только torch.* globals).

    Opcode-поток:
      PROTO 2
      GLOBAL "torch.nn.modules.linear Linear"   ← безопасный PyTorch-импорт
      EMPTY_TUPLE
      REDUCE
      STOP
    """
    return (
        _PROTO2
        + b"ctorch.nn.modules.linear\nLinear\n"  # GLOBAL
        + _EMPTY_TUPLE
        + _REDUCE
        + _STOP
    )


def _make_safe_pytorch_fixtures() -> None:
    """Создаёт безопасные PyTorch .pt-фикстуры."""
    SAFE_DIR.mkdir(parents=True, exist_ok=True)

    # safe/simple_model.pt — чистый .pt с безопасным pickle внутри
    safe_pkl = _build_safe_pytorch_pickle()
    (SAFE_DIR / "simple_model.pt").write_bytes(make_pytorch_zip(safe_pkl))


def _make_malicious_pytorch_fixtures() -> None:
    """Создаёт вредоносные PyTorch-фикстуры (opcode-конструкция, без pickle.dumps).

    Оба файла (.pt и .bin) содержат os.system в archive/data.pkl.
    .bin — это HuggingFace-формат, использующий ту же ZIP-структуру.
    """
    MALICIOUS_DIR.mkdir(parents=True, exist_ok=True)

    os_pkl = _build_os_system_pickle()

    # malicious/pytorch_os_system.pt — вредоносный .pt
    (MALICIOUS_DIR / "pytorch_os_system.pt").write_bytes(make_pytorch_zip(os_pkl))

    # malicious/pytorch_os_system.bin — HuggingFace .bin, та же структура
    (MALICIOUS_DIR / "pytorch_os_system.bin").write_bytes(make_pytorch_zip(os_pkl))


if __name__ == "__main__":
    _make_safe_fixtures()
    _make_safe_joblib_fixtures()
    _make_safe_pytorch_fixtures()
    _make_malicious_fixtures()
    _make_malicious_joblib_fixtures()
    _make_malicious_pytorch_fixtures()
    print("Fixtures generated:")
    for f in sorted(FIXTURES_DIR.rglob("*")):
        if f.is_file() and f.suffix in {".pkl", ".pt", ".pth", ".bin", ".ckpt", ".npz", ".joblib"}:
            print(f"  {f.relative_to(FIXTURES_DIR)}")

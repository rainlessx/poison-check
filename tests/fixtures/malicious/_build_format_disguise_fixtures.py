"""Создание фикстур подмены формата (strict_format_detection, MLS-FMT-001).

Атака: файл называется так, будто он в безопасном формате, а внутри лежит
формат, исполняющий код при загрузке. Наивный потребитель смотрит на
расширение ``.safetensors``, выбирает «безопасный» путь загрузки — и получает
RCE, потому что внутри pickle.

Обе фикстуры строятся ВРУЧНУЮ побайтово; ``pickle.dumps`` на злонамеренном
объекте не используется, ``pickle.load`` не вызывается никогда.

Запуск: python tests/fixtures/malicious/_build_format_disguise_fixtures.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Опкоды pickle (см. pickletools)
# ---------------------------------------------------------------------------

PROTO = b"\x80"            # PROTO <версия>
GLOBAL = b"c"              # GLOBAL \n module \n name \n
SHORT_BINSTRING = b"U"     # SHORT_BINSTRING <len:1> <bytes>
TUPLE1 = b"\x85"           # TUPLE1
REDUCE = b"R"              # REDUCE
STOP = b"."                # STOP


def build_os_system_pickle(command: bytes = b"id") -> bytes:
    """Собирает pickle-поток ``os.system(command)`` из опкодов.

    :param command: Команда, передаваемая os.system (в фикстуре безобидная).
    :return: Байты pickle-потока протокола 2.
    """
    return (
        PROTO
        + b"\x02"
        + GLOBAL
        + b"os\nsystem\n"
        + SHORT_BINSTRING
        + bytes([len(command)])
        + command
        + TUPLE1
        + REDUCE
        + STOP
    )


def build_minimal_safetensors() -> bytes:
    """Собирает минимальный валидный safetensors-файл.

    Структура: uint64 LE длина JSON-заголовка, JSON-заголовок, данные тензора.

    :return: Байты safetensors-файла с одним тензором F32 из одного элемента.
    """
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x80\x3f"


def main() -> None:
    """Записывает обе фикстуры подмены формата в каталог malicious/."""
    disguised = OUT_DIR / "disguise_pickle_as_safetensors.safetensors"
    disguised.write_bytes(build_os_system_pickle())
    print(f"{disguised.name}: {disguised.stat().st_size} байт")

    reverse = OUT_DIR / "disguise_safetensors_as_pickle.pkl"
    reverse.write_bytes(build_minimal_safetensors())
    print(f"{reverse.name}: {reverse.stat().st_size} байт")


if __name__ == "__main__":
    main()

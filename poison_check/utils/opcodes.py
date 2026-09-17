"""Вспомогательные константы и функции для работы с pickle-опкодами."""

import pickletools

DANGEROUS_REDUCE_OPCODES: set[str] = {
    "REDUCE",
    "BUILD",
    "INST",
    "OBJ",
    "NEWOBJ",
    "NEWOBJ_EX",
    "STACK_GLOBAL",
}

ALL_OPCODE_NAMES: dict[int, str] = {
    ord(code.code): code.name for code in pickletools.opcodes
}


def opcode_name(byte: int) -> str:
    """Возвращает имя опкода по байтовому значению или UNKNOWN_0x{byte:02x}."""
    return ALL_OPCODE_NAMES.get(byte, f"UNKNOWN_0x{byte:02x}")


def is_reduce_opcode(opcode_name_str: str) -> bool:
    """Возвращает True если опкод относится к вызывающим (REDUCE, NEWOBJ и др.)."""
    return opcode_name_str in DANGEROUS_REDUCE_OPCODES

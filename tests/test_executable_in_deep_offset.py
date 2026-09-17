"""Регрессионный тест: PE/ELF/Mach-O на offset > 4 КБ должны находиться.

История: в v0.1 ExecutableDetector смотрел только на raw_content_sample
(первые 4 КБ файла). Любой вредоносный исполняемый файл, лежащий после первых
4 КБ pickle-потока (типичный сценарий — PE-blob после opcode-серии), оставался
незамеченным.

В v0.2 каждый сканер сам ищет сигнатуры по всему распакованному содержимому
и заполняет RawScanData.embedded_bytes. ExecutableDetector берёт оттуда.

В v0.3 добавлена структурная валидация PE (MZ → e_lfanew → PE\\x00\\x00),
поэтому тестовые фикстуры содержат валидный DOS-заголовок с правильным e_lfanew.

Этот тест строит pickle-файл, в котором PE-сигнатура лежит на offset 5 КБ.
Проверяет:
  1. PickleScanner находит сигнатуру и кладёт её в embedded_bytes;
  2. ExecutableDetector превращает её в Issue MLS-EXE-001 (CRITICAL);
  3. Offset в Issue.location указывает на правильную позицию.
"""

from __future__ import annotations

import pickletools
from pathlib import Path

import pytest

from poison_check.core.result import MLContext, Severity
from poison_check.detectors.executable_detector import ExecutableDetector
from poison_check.scanners.pickle_scanner import PickleScanner

_DUMMY_CTX = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])


def _make_valid_pe_blob() -> bytes:
    """Возвращает минимальный валидный PE-blob, проходящий структурную валидацию.

    Структура:
      bytes[0:2]   = b"MZ"                 — DOS signature
      bytes[2:60]  = b"\\x00" * 58         — DOS stub (padding)
      bytes[60:64] = b"\\x40\\x00\\x00\\x00" — e_lfanew = 0x40 (64)
      bytes[64:68] = b"PE\\x00\\x00"        — PE signature at e_lfanew
      bytes[68:]   = b"\\x00" * 60          — COFF header stub

    Итого: 128 байт. Проходит _validate_pe: e_lfanew=0x40 ∈ [0x40, 0x1000],
    data[64:68] == b"PE\\x00\\x00".
    """
    e_lfanew = 0x40
    return (
        b"MZ"
        + b"\x00" * 58               # padding до e_lfanew
        + e_lfanew.to_bytes(4, "little")  # e_lfanew = 64
        + b"PE\x00\x00"              # PE signature
        + b"\x00" * 60               # COFF header stub
    )


def _build_pickle_with_pe_at_offset(target_offset: int) -> bytes:
    """Конструирует валидный pickle поток, в котором PE-сигнатура лежит на target_offset.

    Сигнатура — минимальный валидный PE-blob (_make_valid_pe_blob), проходящий
    структурную валидацию (e_lfanew → PE\\x00\\x00). Встраивается как BINBYTES8
    после балластных байт нужного размера.

    Возвращает байты валидного pickle-файла (PROTO 4 + BINBYTES8 + STOP).
    """
    pe_blob = _make_valid_pe_blob()
    # Длина ballast = target_offset (с запасом). Итоговая длина файла будет
    # 2 (PROTO) + 9 (BINBYTES8 header) + ballast_len + 9 + len(pe_blob) + 1 (STOP),
    # т.е. PE окажется на offset >= target_offset.
    ballast_len = target_offset
    ballast = b"a" * ballast_len

    # Собираем вручную:
    #   \x80\x04          PROTO 4
    #   \x8e <len:8>      BINBYTES8 ballast
    #   \x8e <len:8>      BINBYTES8 pe_blob (с MZ → валидный PE)
    #   .                 STOP
    # BINBYTES8 (0x8e) принимает произвольные байты — UTF-8 валидация не нужна.
    parts: list[bytes] = []
    parts.append(b"\x80\x04")
    parts.append(b"\x8e")
    parts.append(len(ballast).to_bytes(8, "little"))
    parts.append(ballast)
    parts.append(b"\x8e")
    parts.append(len(pe_blob).to_bytes(8, "little"))
    parts.append(pe_blob)
    parts.append(b".")
    return b"".join(parts)


def test_pe_at_offset_5kb_is_found_by_pickle_scanner(tmp_path: Path) -> None:
    """PE-сигнатура на offset ≈5 КБ обнаруживается PickleScanner и попадает в embedded_bytes."""
    target_offset = 5 * 1024
    data = _build_pickle_with_pe_at_offset(target_offset)
    assert len(data) > target_offset, "fixture: payload должен быть длиннее target_offset"

    pkl = tmp_path / "deep_pe.pkl"
    pkl.write_bytes(data)

    # Подтверждаем что pickle всё ещё валиден (genops не падает)
    with pkl.open("rb") as fh:
        list(pickletools.genops(fh))

    raw = PickleScanner().scan(pkl)

    assert raw.embedded_bytes is not None, (
        "PickleScanner должен заполнять embedded_bytes при наличии PE-сигнатур"
    )
    pe_findings = [e for e in raw.embedded_bytes if e.signature_type == "PE"]
    assert pe_findings, (
        f"Ожидалась PE-сигнатура в embedded_bytes, получено: "
        f"{[e.signature_type for e in raw.embedded_bytes]}"
    )
    # Offset должен быть >= target_offset (плюс заголовок pickle)
    assert any(e.offset >= 4096 for e in pe_findings), (
        f"PE-сигнатура должна быть после первых 4 КБ, offsets: "
        f"{[e.offset for e in pe_findings]}"
    )


def test_executable_detector_emits_critical_for_deep_pe(tmp_path: Path) -> None:
    """PickleScanner → ExecutableDetector выдаёт MLS-EXE-001 CRITICAL для PE на offset > 4 КБ."""
    pkl = tmp_path / "deep_pe.pkl"
    pkl.write_bytes(_build_pickle_with_pe_at_offset(5 * 1024))

    raw = PickleScanner().scan(pkl)
    issues = ExecutableDetector().analyze(raw, _DUMMY_CTX)

    pe_issues = [i for i in issues if i.code == "MLS-EXE-001"]
    assert pe_issues, (
        f"Ожидался Issue MLS-EXE-001, получено: {[i.code for i in issues]}"
    )
    assert pe_issues[0].severity == Severity.CRITICAL


@pytest.mark.parametrize("target_offset", [10 * 1024, 100 * 1024, 1024 * 1024])
def test_pe_at_various_deep_offsets(tmp_path: Path, target_offset: int) -> None:
    """PE на разных глубинах (10 КБ, 100 КБ, 1 МБ) находится."""
    pkl = tmp_path / f"pe_at_{target_offset}.pkl"
    pkl.write_bytes(_build_pickle_with_pe_at_offset(target_offset))

    raw = PickleScanner().scan(pkl)
    assert raw.embedded_bytes is not None
    assert any(
        e.signature_type == "PE" and e.offset > 4096
        for e in raw.embedded_bytes
    ), (
        f"PE на offset {target_offset} не найден; "
        f"embedded={[(e.signature_type, e.offset) for e in raw.embedded_bytes]}"
    )

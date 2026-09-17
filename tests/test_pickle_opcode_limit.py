"""Регрессионные тесты: кап на накопление опкодов/строк (защита от OOM).

Pickle из миллиардов однобайтовых опкодов породил бы миллиарды OpcodeInfo и
исчерпал бы память (DoS — как на злонамеренном, так и на случайно-повреждённом
большом файле). Сканер ограничивает накопление (MAX_OPCODES/MAX_STRINGS/
MAX_REDUCE_CALLS), аккуратно прерывает разбор и выставляет metadata-флаг
``opcode_limit_exceeded``. Предупреждение MLS-PKL-006 (MEDIUM) эмитирует
BlocklistDetector по этому флагу — не сам сканер.

Гарантия: усечение opcodes не прячет опасные глобалы, собранные ДО лимита —
их по-прежнему видит globals_set, и CRITICAL-issue продолжает эмитироваться.

Payload-ы строятся вручную из байт (opcode-конструирование), без pickle.dumps.
Лимиты в большинстве тестов понижены через monkeypatch — иначе пришлось бы
аллоцировать 2 млн объектов ради проверки механизма.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import poison_check.scanners.pickle_scanner as pickle_scanner
from poison_check.core.result import MLContext, Severity
from poison_check.detectors.blocklist_detector import BlocklistDetector
from poison_check.scanners.pickle_scanner import PickleScanner


@pytest.fixture()
def scanner() -> PickleScanner:
    """Экземпляр PickleScanner для тестов."""
    return PickleScanner()


@pytest.fixture()
def detector() -> BlocklistDetector:
    """Экземпляр BlocklistDetector для тестов."""
    return BlocklistDetector()


@pytest.fixture()
def context() -> MLContext:
    """Нейтральный MLContext (blocklist от фреймворка не зависит)."""
    return MLContext(framework="unknown", confidence=1.0)


# ---------------------------------------------------------------------------
# Построители payload-ов (ручная сборка байт, без pickle.dumps)
# ---------------------------------------------------------------------------

# NONE = 'N' (0x4e) — безобидный однобайтовый опкод, кладёт None на стек.
_NONE = b"N"


def _make_long_benign_stream(n_none: int) -> bytes:
    """PROTO 4 + n однобайтовых NONE-опкодов + STOP. Полностью безобиден."""
    return b"\x80\x04" + _NONE * n_none + b"."


def _make_os_system_then_long_tail(n_none: int) -> bytes:
    """GLOBAL os.system в первых опкодах + длинный безобидный хвост NONE."""
    return (
        b"\x80\x02"          # PROTO 2
        b"cos\nsystem\n"     # GLOBAL os system  (собирается ДО любого лимита)
        b")"                 # EMPTY_TUPLE
        b"R"                 # REDUCE
        + _NONE * n_none     # длинный хвост, переполняющий лимит
        + b"."               # STOP
    )


def _make_small_clean_pickle() -> bytes:
    """Маленький валидный pickle: пустой словарь. Ничего подозрительного."""
    return b"\x80\x04}\x94."  # PROTO 4, EMPTY_DICT, MEMOIZE, STOP


# ---------------------------------------------------------------------------
# 1. Поток > лимита безобидных опкодов → metadata-флаг + MLS-PKL-006, без падения
# ---------------------------------------------------------------------------


class TestOpcodeLimitMetadata:
    """Усечение длинного потока отражается в metadata сканера."""

    def test_long_stream_sets_metadata_flag(
        self,
        scanner: PickleScanner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Поток длиннее MAX_OPCODES → metadata["opcode_limit_exceeded"]="true"."""
        monkeypatch.setattr(pickle_scanner, "MAX_OPCODES", 10)
        path = tmp_path / "long_benign.pkl"
        path.write_bytes(_make_long_benign_stream(100))

        result = scanner.scan(path)

        assert result.error is None  # аккуратное прерывание, не исключение
        assert result.metadata is not None
        assert result.metadata.get("opcode_limit_exceeded") == "true"
        assert result.metadata.get("opcode_limit") == "10"

    def test_opcodes_capped_at_limit(
        self,
        scanner: PickleScanner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Список opcodes не растёт сверх лимита — защита от OOM реально работает."""
        monkeypatch.setattr(pickle_scanner, "MAX_OPCODES", 10)
        path = tmp_path / "capped.pkl"
        path.write_bytes(_make_long_benign_stream(1000))

        result = scanner.scan(path)

        assert result.opcodes is not None
        assert len(result.opcodes) == 10

    def test_scan_bytes_path_also_capped(
        self,
        scanner: PickleScanner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Путь scan_bytes (joblib/pytorch) тоже выставляет флаг усечения."""
        monkeypatch.setattr(pickle_scanner, "MAX_OPCODES", 10)
        data = _make_long_benign_stream(100)

        result = scanner.scan_bytes(data, tmp_path / "inner.pkl")

        assert result.metadata is not None
        assert result.metadata.get("opcode_limit_exceeded") == "true"

    def test_detector_emits_mls_pkl_006_medium(
        self,
        scanner: PickleScanner,
        detector: BlocklistDetector,
        context: MLContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """metadata-флаг → BlocklistDetector эмитит MLS-PKL-006 MEDIUM."""
        monkeypatch.setattr(pickle_scanner, "MAX_OPCODES", 10)
        path = tmp_path / "long_benign.pkl"
        path.write_bytes(_make_long_benign_stream(100))

        result = scanner.scan(path)
        issues = detector.analyze(result, context)

        limit_issues = [i for i in issues if i.code == "MLS-PKL-006"]
        assert len(limit_issues) == 1
        assert limit_issues[0].severity == Severity.MEDIUM


# ---------------------------------------------------------------------------
# 2. Опасный глобал ДО лимита → CRITICAL сохраняется несмотря на усечение
# ---------------------------------------------------------------------------


class TestDangerousGlobalBeforeLimitSurvives:
    """Усечение не должно прятать глобалы, собранные до лимита."""

    def test_os_system_still_in_globals(
        self,
        scanner: PickleScanner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """os.system в первых опкодах остаётся в globals_set после усечения."""
        monkeypatch.setattr(pickle_scanner, "MAX_OPCODES", 10)
        path = tmp_path / "evil_then_tail.pkl"
        path.write_bytes(_make_os_system_then_long_tail(500))

        result = scanner.scan(path)

        assert result.globals is not None
        assert ("os", "system") in result.globals
        assert result.metadata is not None
        assert result.metadata.get("opcode_limit_exceeded") == "true"

    def test_critical_still_emitted_with_limit_warning(
        self,
        scanner: PickleScanner,
        detector: BlocklistDetector,
        context: MLContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CRITICAL по os.system эмитится, MLS-PKL-006 идёт дополнительно."""
        monkeypatch.setattr(pickle_scanner, "MAX_OPCODES", 10)
        path = tmp_path / "evil_then_tail.pkl"
        path.write_bytes(_make_os_system_then_long_tail(500))

        result = scanner.scan(path)
        issues = detector.analyze(result, context)

        codes = {i.code for i in issues}
        assert "MLS-PKL-006" in codes  # предупреждение об усечении
        criticals = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any(
            i.details.get("module") == "os" and i.details.get("name") == "system"
            for i in criticals
        )


# ---------------------------------------------------------------------------
# 3. Нормальный маленький pickle → нет флага, нет MLS-PKL-006
# ---------------------------------------------------------------------------


class TestNoRegressionOnSmallPickle:
    """Обычные файлы под лимитом ведут себя как раньше."""

    def test_small_pickle_no_flag(
        self, scanner: PickleScanner, tmp_path: Path
    ) -> None:
        """Маленький валидный pickle не выставляет opcode_limit_exceeded."""
        path = tmp_path / "clean.pkl"
        path.write_bytes(_make_small_clean_pickle())

        result = scanner.scan(path)

        assert result.error is None
        if result.metadata is not None:
            assert "opcode_limit_exceeded" not in result.metadata

    def test_small_pickle_no_mls_pkl_006(
        self,
        scanner: PickleScanner,
        detector: BlocklistDetector,
        context: MLContext,
        tmp_path: Path,
    ) -> None:
        """Маленький pickle → детектор не эмитит MLS-PKL-006."""
        path = tmp_path / "clean.pkl"
        path.write_bytes(_make_small_clean_pickle())

        result = scanner.scan(path)
        issues = detector.analyze(result, context)

        assert all(i.code != "MLS-PKL-006" for i in issues)

    def test_default_limit_not_triggered_by_modest_stream(
        self, scanner: PickleScanner, tmp_path: Path
    ) -> None:
        """Поток из 1000 опкодов при дефолтном лимите (2 млн) не усекается."""
        path = tmp_path / "modest.pkl"
        path.write_bytes(_make_long_benign_stream(1000))

        result = scanner.scan(path)

        if result.metadata is not None:
            assert "opcode_limit_exceeded" not in result.metadata

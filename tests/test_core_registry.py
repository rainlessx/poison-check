"""Тесты для ScannerRegistry и DetectorRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry, ScannerRegistry
from poison_check.core.result import Issue, MLContext, Severity
from poison_check.core.scanner_base import BaseScanner, RawScanData


@pytest.fixture(autouse=True)
def _isolate_registries() -> None:
    """Изолирует каждый тест — снимает snapshot реестров до и восстанавливает после.

    Аудит #12: раньше использовался ``ScannerRegistry._reset()``, что разрушало
    глобальное состояние и требовало ``_ensure_scanners_registered`` в Scanner.
    Snapshot/restore через копирование dict решает обе проблемы:
    другие тесты, выполняющиеся после этого модуля, видят корректно
    заполненный реестр без вызова специальных хаков.
    """
    saved_scanners = dict(ScannerRegistry._scanners)
    saved_detectors = dict(DetectorRegistry._detectors)
    ScannerRegistry._scanners.clear()
    DetectorRegistry._detectors.clear()
    yield  # type: ignore[misc]
    ScannerRegistry._scanners.clear()
    ScannerRegistry._scanners.update(saved_scanners)
    DetectorRegistry._detectors.clear()
    DetectorRegistry._detectors.update(saved_detectors)


# ---------------------------------------------------------------------------
# Вспомогательные конкретные классы
# ---------------------------------------------------------------------------


def _make_scanner(ext: str, scanner_name: str) -> type[BaseScanner]:
    """Фабрика: создаёт сканер-заглушку для указанного расширения."""

    class _S(BaseScanner):
        name = scanner_name
        description = f"Тестовый сканер для {ext}"
        supported_extensions = [ext]
        magic_bytes: list[bytes] = []

        @classmethod
        def can_handle(cls, path: Path) -> bool:
            return path.suffix == ext

        def scan(self, path: Path) -> RawScanData:
            return RawScanData(
                file_path=path,
                file_hash={},
                file_size=0,
                scanner_name=self.name,
            )

    _S.__name__ = scanner_name
    _S.__qualname__ = scanner_name
    return _S


def _make_detector(detector_name: str) -> type[BaseDetector]:
    """Фабрика: создаёт детектор-заглушку."""

    class _D(BaseDetector):
        name = detector_name
        description = f"Тестовый детектор {detector_name}"
        severity_range = (Severity.LOW, Severity.HIGH)

        def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
            return []

    _D.__name__ = detector_name
    _D.__qualname__ = detector_name
    return _D


# ---------------------------------------------------------------------------
# Тесты ScannerRegistry
# ---------------------------------------------------------------------------


def test_scanner_register_via_decorator() -> None:
    """Декоратор @ScannerRegistry.register добавляет сканер в реестр."""
    PickleScanner = _make_scanner(".pkl", "pickle")

    ScannerRegistry.register(PickleScanner)

    assert ScannerRegistry.get("pickle") is PickleScanner


def test_scanner_register_as_decorator_syntax() -> None:
    """Синтаксис @ScannerRegistry.register возвращает тот же класс."""
    PickleScanner = _make_scanner(".pkl", "pickle")
    result = ScannerRegistry.register(PickleScanner)
    assert result is PickleScanner


def test_find_scanner_returns_correct_for_known_extension(tmp_path: Path) -> None:
    """find_scanner возвращает правильный сканер для файла с известным расширением."""
    PickleScanner = _make_scanner(".pkl", "pickle")
    NumpyScanner = _make_scanner(".npy", "numpy")
    ScannerRegistry.register(PickleScanner)
    ScannerRegistry.register(NumpyScanner)

    pkl_file = tmp_path / "model.pkl"
    pkl_file.touch()

    found = ScannerRegistry.find_scanner(pkl_file)
    assert found is PickleScanner


def test_find_scanner_returns_none_for_unknown_format(tmp_path: Path) -> None:
    """find_scanner возвращает None если ни один сканер не подходит."""
    PickleScanner = _make_scanner(".pkl", "pickle")
    ScannerRegistry.register(PickleScanner)

    unknown = tmp_path / "model.xyz"
    unknown.touch()

    assert ScannerRegistry.find_scanner(unknown) is None


def test_scanner_duplicate_registration_raises_value_error() -> None:
    """Повторная регистрация сканера с тем же именем вызывает ValueError."""
    PickleScanner1 = _make_scanner(".pkl", "pickle")
    PickleScanner2 = _make_scanner(".pkl", "pickle")
    ScannerRegistry.register(PickleScanner1)

    with pytest.raises(ValueError, match="уже зарегистрирован"):
        ScannerRegistry.register(PickleScanner2)


def test_all_scanners_returns_all_registered() -> None:
    """all_scanners() возвращает все зарегистрированные классы."""
    PickleScanner = _make_scanner(".pkl", "pickle")
    NumpyScanner = _make_scanner(".npy", "numpy")
    SafetensorsScanner = _make_scanner(".safetensors", "safetensors")

    ScannerRegistry.register(PickleScanner)
    ScannerRegistry.register(NumpyScanner)
    ScannerRegistry.register(SafetensorsScanner)

    all_s = ScannerRegistry.all_scanners()
    assert len(all_s) == 3
    assert set(all_s) == {PickleScanner, NumpyScanner, SafetensorsScanner}


def test_scanner_get_unknown_name_returns_none() -> None:
    """get() возвращает None для незарегистрированного имени."""
    assert ScannerRegistry.get("nonexistent") is None


def test_all_scanners_empty_when_no_registrations() -> None:
    """all_scanners() возвращает пустой список если реестр пуст."""
    assert ScannerRegistry.all_scanners() == []


# ---------------------------------------------------------------------------
# Тесты DetectorRegistry
# ---------------------------------------------------------------------------


def test_detector_register_via_decorator() -> None:
    """Декоратор @DetectorRegistry.register добавляет детектор в реестр."""
    AllowlistDetector = _make_detector("allowlist")
    DetectorRegistry.register(AllowlistDetector)
    assert DetectorRegistry.get("allowlist") is AllowlistDetector


def test_detector_register_returns_same_class() -> None:
    """register() возвращает тот же класс (для синтаксиса декоратора)."""
    AllowlistDetector = _make_detector("allowlist")
    result = DetectorRegistry.register(AllowlistDetector)
    assert result is AllowlistDetector


def test_detector_duplicate_registration_raises_value_error() -> None:
    """Повторная регистрация детектора с тем же именем вызывает ValueError."""
    D1 = _make_detector("allowlist")
    D2 = _make_detector("allowlist")
    DetectorRegistry.register(D1)

    with pytest.raises(ValueError, match="уже зарегистрирован"):
        DetectorRegistry.register(D2)


def test_all_detectors_returns_all_registered() -> None:
    """all_detectors() возвращает все зарегистрированные классы."""
    D1 = _make_detector("allowlist")
    D2 = _make_detector("blocklist")
    DetectorRegistry.register(D1)
    DetectorRegistry.register(D2)

    all_d = DetectorRegistry.all_detectors()
    assert len(all_d) == 2
    assert set(all_d) == {D1, D2}


def test_enabled_for_policy_excludes_disabled() -> None:
    """enabled_for_policy фильтрует детекторы из disabled_detectors."""
    D_allow = _make_detector("allowlist")
    D_block = _make_detector("blocklist")
    D_cve = _make_detector("cve")
    DetectorRegistry.register(D_allow)
    DetectorRegistry.register(D_block)
    DetectorRegistry.register(D_cve)

    policy = {"disabled_detectors": ["blocklist"]}
    enabled = DetectorRegistry.enabled_for_policy(policy)

    assert D_block not in enabled
    assert D_allow in enabled
    assert D_cve in enabled


def test_enabled_for_policy_empty_disabled_returns_all() -> None:
    """enabled_for_policy без disabled_detectors возвращает все детекторы."""
    D1 = _make_detector("allowlist")
    D2 = _make_detector("cve")
    DetectorRegistry.register(D1)
    DetectorRegistry.register(D2)

    enabled = DetectorRegistry.enabled_for_policy({})
    assert set(enabled) == {D1, D2}


def test_enabled_for_policy_all_disabled_returns_empty() -> None:
    """enabled_for_policy может вернуть пустой список если всё отключено."""
    D1 = _make_detector("allowlist")
    DetectorRegistry.register(D1)

    enabled = DetectorRegistry.enabled_for_policy({"disabled_detectors": ["allowlist"]})
    assert enabled == []


def test_detector_get_unknown_returns_none() -> None:
    """get() возвращает None для незарегистрированного имени."""
    assert DetectorRegistry.get("nonexistent") is None

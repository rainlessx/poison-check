"""Тесты для KerasScanner (.keras / .h5).

Сканер извлекает ФАКТЫ (Lambda-слои, custom-объекты, тела функций) в
metadata/strings; УГРОЗУ эмитит KerasThreatDetector (см. test_keras_detector.py).

Все .keras-фикстуры собираются ВРУЧНУЮ как ZIP с config.json — без вызова
keras.save / keras.models.load_model. HDF5-путь тестируется через h5py: он
переведён в обязательные зависимости, поэтому тест не пропускается. Поведение
при сломанной установке (h5py не импортируется) проверяется отдельно через
подмену sys.modules — см. также tests/test_h5py_required.py.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from poison_check.core.registry import ScannerRegistry
from poison_check.scanners.keras_scanner import (
    KerasScanner,
    _collect_keras_facts,
)

# ---------------------------------------------------------------------------
# Builder-функции (.keras собирается вручную как ZIP)
# ---------------------------------------------------------------------------


def _make_keras_zip(
    config: dict[str, object],
    metadata: dict[str, object] | None = None,
    *,
    weights: bytes = b"\x89HDF\r\n\x1a\nweights",
) -> bytes:
    """Строит валидный .keras (ZIP) из config-словаря."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("config.json", json.dumps(config))
        zf.writestr("metadata.json", json.dumps(metadata or {"keras_version": "3.5.0"}))
        zf.writestr("model.weights.h5", weights)
    return buf.getvalue()


def _clean_config() -> dict[str, object]:
    """Чистая Sequential-модель: только Dense, без Lambda и custom-объектов."""
    return {
        "module": "keras",
        "class_name": "Sequential",
        "registered_name": None,
        "config": {
            "name": "sequential",
            "layers": [
                {
                    "module": "keras.layers",
                    "class_name": "Dense",
                    "registered_name": None,
                    "config": {"name": "dense", "units": 16, "activation": "relu"},
                },
                {
                    "module": "keras.layers",
                    "class_name": "Dense",
                    "registered_name": None,
                    "config": {"name": "dense_1", "units": 1, "activation": "sigmoid"},
                },
            ],
        },
    }


def _lambda_config(name: str = "evil_lambda") -> dict[str, object]:
    """Модель с Lambda-слоем, несущим сериализованный код."""
    return {
        "module": "keras",
        "class_name": "Sequential",
        "registered_name": None,
        "config": {
            "name": "sequential",
            "layers": [
                {
                    "module": "keras.layers",
                    "class_name": "Lambda",
                    "registered_name": None,
                    "config": {
                        "name": name,
                        "function": {
                            "class_name": "__lambda__",
                            "config": {
                                "code": "4wEAAAAAAAAAAAAAAAEAAABTAAAA",  # похоже на base64-байткод
                                "defaults": None,
                                "closure": None,
                            },
                        },
                    },
                }
            ],
        },
    }


def _custom_config() -> dict[str, object]:
    """Модель с пользовательским custom-объектом (registered_name)."""
    return {
        "module": "keras",
        "class_name": "Sequential",
        "registered_name": None,
        "config": {
            "name": "sequential",
            "layers": [
                {
                    "module": "my_evil_package.layers",
                    "class_name": "MyLayer",
                    "registered_name": "my_evil_package>MyLayer",
                    "config": {"name": "my_layer"},
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# Контракт сканера и регистрация
# ---------------------------------------------------------------------------


class TestKerasScannerContract:
    """Атрибуты и регистрация сканера."""

    def test_name(self) -> None:
        assert KerasScanner.name == "keras"

    def test_extensions(self) -> None:
        assert set(KerasScanner.supported_extensions) == {".keras", ".h5", ".hdf5"}

    def test_registered(self) -> None:
        assert ScannerRegistry.get("keras") is KerasScanner

    def test_can_handle_keras_zip(self, tmp_path: Path) -> None:
        f = tmp_path / "model.keras"
        f.write_bytes(_make_keras_zip(_clean_config()))
        assert KerasScanner.can_handle(f) is True

    def test_can_handle_h5_magic(self, tmp_path: Path) -> None:
        f = tmp_path / "model.h5"
        f.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 32)
        assert KerasScanner.can_handle(f) is True

    def test_can_handle_rejects_wrong_magic(self, tmp_path: Path) -> None:
        """.keras без ZIP-magic не берётся (это не наш формат)."""
        f = tmp_path / "model.keras"
        f.write_bytes(b"not a zip at all")
        assert KerasScanner.can_handle(f) is False

    def test_can_handle_rejects_other_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"PK\x03\x04")
        assert KerasScanner.can_handle(f) is False


# ---------------------------------------------------------------------------
# Разбор .keras (ZIP)
# ---------------------------------------------------------------------------


class TestKerasZipScan:
    """Извлечение фактов из config.json .keras-архива."""

    def test_clean_model_no_facts(self, tmp_path: Path) -> None:
        """Чистая модель: нет Lambda, нет custom, error отсутствует."""
        f = tmp_path / "clean.keras"
        f.write_bytes(_make_keras_zip(_clean_config()))
        raw = KerasScanner().scan(f)
        assert raw.error is None
        assert raw.scanner_name == "keras"
        assert raw.metadata is not None
        assert raw.metadata["keras_format"] == "keras_zip"
        assert raw.metadata["lambda_layer_count"] == "0"
        assert raw.metadata["custom_object_count"] == "0"
        assert raw.metadata.get("keras_version") == "3.5.0"

    def test_lambda_recorded_in_metadata(self, tmp_path: Path) -> None:
        """Lambda-слой фиксируется в metadata (счётчик + имя)."""
        f = tmp_path / "lam.keras"
        f.write_bytes(_make_keras_zip(_lambda_config("evil_lambda")))
        raw = KerasScanner().scan(f)
        assert raw.error is None
        assert raw.metadata is not None
        assert raw.metadata["lambda_layer_count"] == "1"
        assert "evil_lambda" in raw.metadata["lambda_layers"]

    def test_lambda_function_code_in_strings(self, tmp_path: Path) -> None:
        """Тело Lambda-функции попадает в strings (для SecretsDetector/forensics)."""
        f = tmp_path / "lam.keras"
        f.write_bytes(_make_keras_zip(_lambda_config()))
        raw = KerasScanner().scan(f)
        assert raw.strings is not None
        joined = " ".join(s.value for s in raw.strings)
        assert "4wEAAAAAAAAAAAAAAAEAAABTAAAA" in joined

    def test_custom_object_recorded(self, tmp_path: Path) -> None:
        """registered_name пользовательского объекта фиксируется в metadata."""
        f = tmp_path / "custom.keras"
        f.write_bytes(_make_keras_zip(_custom_config()))
        raw = KerasScanner().scan(f)
        assert raw.metadata is not None
        assert raw.metadata["custom_object_count"] == "1"
        assert "my_evil_package>MyLayer" in raw.metadata["custom_objects"]

    def test_functional_nested_lambda_found(self, tmp_path: Path) -> None:
        """Lambda в глубоко вложенной Functional-структуре тоже находится."""
        config = {
            "module": "keras",
            "class_name": "Functional",
            "registered_name": None,
            "config": {
                "name": "model",
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "registered_name": None,
                        "config": {"name": "input"},
                    },
                    {
                        "class_name": "Lambda",
                        "registered_name": None,
                        "config": {"name": "deep_lambda", "function": ["ADD", None, None]},
                        "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__"}]}],
                    },
                ],
            },
        }
        f = tmp_path / "func.keras"
        f.write_bytes(_make_keras_zip(config))
        raw = KerasScanner().scan(f)
        assert raw.metadata is not None
        assert raw.metadata["lambda_layer_count"] == "1"
        assert "deep_lambda" in raw.metadata["lambda_layers"]

    def test_missing_config_json_is_error(self, tmp_path: Path) -> None:
        """.keras без config.json → error, не исключение."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", json.dumps({"keras_version": "3.0"}))
            zf.writestr("model.weights.h5", b"\x89HDF\r\n\x1a\n")
        f = tmp_path / "nocfg.keras"
        f.write_bytes(buf.getvalue())
        raw = KerasScanner().scan(f)
        assert raw.error is not None
        assert "config.json" in raw.error

    def test_invalid_json_is_error(self, tmp_path: Path) -> None:
        """Битый config.json → error, не исключение."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("config.json", b"{ this is not json ")
        f = tmp_path / "badjson.keras"
        f.write_bytes(buf.getvalue())
        raw = KerasScanner().scan(f)
        assert raw.error is not None
        assert "JSON" in raw.error

    def test_corrupted_zip_is_error_not_exception(self, tmp_path: Path) -> None:
        """ZIP-magic + мусор → error, сканер не бросает исключение."""
        f = tmp_path / "corrupt.keras"
        f.write_bytes(b"PK\x03\x04" + b"\xff" * 64)
        raw = KerasScanner().scan(f)
        assert raw.error is not None
        assert raw.scanner_name == "keras"


# ---------------------------------------------------------------------------
# Разбор .h5 (HDF5)
# ---------------------------------------------------------------------------


class TestKerasH5Scan:
    """HDF5-путь: штатно — полный разбор, при сломанной установке — degradation."""

    def test_h5_without_h5py_sets_fact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Неимпортируемый h5py → факт h5py_available=false, error отсутствует.

        h5py обязателен, поэтому «отсутствие» моделируется подменой sys.modules
        (запись None заставляет import выбросить ImportError), а не skipif.
        """
        monkeypatch.setitem(sys.modules, "h5py", None)
        f = tmp_path / "model.h5"
        f.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
        raw = KerasScanner().scan(f)
        assert raw.error is None
        assert raw.metadata is not None
        assert raw.metadata["keras_format"] == "h5"
        assert raw.metadata["h5py_available"] == "false"

    def test_h5_with_h5py_finds_lambda(self, tmp_path: Path) -> None:
        """С h5py Lambda-слой из model_config тоже находится."""
        import h5py  # noqa: PLC0415

        f = tmp_path / "model.h5"
        with h5py.File(f, "w") as hf:
            hf.attrs["keras_version"] = "2.15.0"
            hf.attrs["backend"] = "tensorflow"
            hf.attrs["model_config"] = json.dumps(_lambda_config("h5_lambda"))
        raw = KerasScanner().scan(f)
        assert raw.error is None
        assert raw.metadata is not None
        assert raw.metadata["h5py_available"] == "true"
        assert raw.metadata["lambda_layer_count"] == "1"
        assert "h5_lambda" in raw.metadata["lambda_layers"]


# ---------------------------------------------------------------------------
# Юнит-тесты обхода config-дерева
# ---------------------------------------------------------------------------


class TestCollectKerasFacts:
    """_collect_keras_facts извлекает факты без интерпретации угроз."""

    def test_clean_config_empty_facts(self) -> None:
        facts = _collect_keras_facts(_clean_config())
        assert facts.lambda_layers == []
        assert facts.custom_objects == []

    def test_lambda_collected(self) -> None:
        facts = _collect_keras_facts(_lambda_config("x"))
        assert facts.lambda_layers == ["x"]
        assert facts.function_payloads  # тело функции сохранено

    def test_custom_collected(self) -> None:
        facts = _collect_keras_facts(_custom_config())
        assert facts.custom_objects == ["my_evil_package>MyLayer"]

    def test_function_class_name_not_custom(self) -> None:
        """registered_name='function' (штатная сериализация) не считается custom."""
        node = {"class_name": "function", "registered_name": "function", "config": "relu"}
        facts = _collect_keras_facts(node)
        assert facts.custom_objects == []
        # но тело функции всё равно зафиксировано как payload
        assert facts.function_payloads

    def test_depth_limit_no_recursion_error(self) -> None:
        """Аномально глубокий config не роняет обход (RecursionError)."""
        node: dict[str, object] = {"config": {}}
        cur = node
        for _ in range(1000):
            child: dict[str, object] = {"config": {}}
            cur["child"] = child
            cur = child
        # Не должно бросить исключение
        facts = _collect_keras_facts(node)
        assert facts.lambda_layers == []

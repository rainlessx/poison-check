"""Тесты слоя фактов о формате (``poison_check.scanners.format_facts``).

Модуль отвечает на два вопроса о файле — «совпадает ли содержимое с обещанием
расширения» и «исполняет ли формат код при загрузке» — и не эмитит ни одной
Issue. Здесь проверяется сам расчёт фактов и полнота двух реестров:
:data:`FORMAT_SAFETY` (классификация форматов) и :data:`SCANNER_FORMATS`
(соответствие сканер → формат). Мета-тесты полноты нужны, чтобы завтрашний
формат или сканер нельзя было добавить, забыв его классифицировать.

Поведение политик поверх этих фактов — в ``tests/test_format_policy.py``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

import poison_check.scanners  # noqa: F401 — регистрация сканеров в реестре
from poison_check.core.format_detector import FileFormat, FormatDetector
from poison_check.core.registry import ScannerRegistry
from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.format_facts import (
    EXPECTED_CONTENT_FORMATS,
    FORMAT_SAFETY,
    META_DETECTED_FORMAT,
    META_FORMAT_MISMATCH,
    META_FORMAT_UNSUPPORTED,
    SCANNER_FORMATS,
    FormatSafety,
    attach_format_facts,
    build_format_facts,
    format_facts_of,
)
from poison_check.scanners.numpy_scanner import META_OBJECT_DTYPE

_PICKLE_BYTES = b"\x80\x02]q\x00(K\x01K\x02e."
_ZIP_BYTES = b"PK\x03\x04" + b"\x00" * 26
_NPY_BYTES = b"\x93NUMPY\x01\x00" + b"\x00" * 8
_GGUF_BYTES = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) * 2
_HDF5_BYTES = b"\x89HDF\r\n\x1a\n" + b"\x00" * 16


def _safetensors_bytes() -> bytes:
    """Минимальный валидный safetensors."""
    header = json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    raw = header.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + b"\x00\x00\x80\x3f"


def _raw(scanner_name: str = "pickle", metadata: dict[str, object] | None = None) -> RawScanData:
    """Собирает минимальный RawScanData нужного сканера."""
    return RawScanData(
        file_path=Path("model.bin"),
        file_hash={},
        file_size=0,
        scanner_name=scanner_name,
        metadata=metadata,  # type: ignore[arg-type]
    )


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    """Пишет файл во временный каталог теста."""
    target = tmp_path / name
    target.write_bytes(data)
    return target


# ---------------------------------------------------------------------------
# 1. Определение формата по содержимому игнорирует расширение
# ---------------------------------------------------------------------------


class TestContentDetectionIgnoresExtension:
    """``detect_content`` смотрит только в байты."""

    @pytest.mark.parametrize(
        ("name", "data", "expected"),
        [
            ("model.safetensors", _PICKLE_BYTES, FileFormat.PICKLE),
            ("model.pkl", _safetensors_bytes(), FileFormat.SAFETENSORS),
            ("model.pkl", _GGUF_BYTES, FileFormat.GGUF),
            ("model.npy", _ZIP_BYTES, FileFormat.ZIP_ARCHIVE),
            ("model.h5", _HDF5_BYTES, FileFormat.KERAS_H5),
        ],
    )
    def test_format_comes_from_bytes(
        self, tmp_path: Path, name: str, data: bytes, expected: FileFormat
    ) -> None:
        """Формат определяется по байтам, каким бы ни было имя файла."""
        evidence = FormatDetector.detect_content(_write(tmp_path, name, data))

        assert evidence.format is expected

    def test_evidence_carries_raw_basis(self, tmp_path: Path) -> None:
        """Признак определения формата — сырой и непустой."""
        evidence = FormatDetector.detect_content(
            _write(tmp_path, "model.safetensors", _PICKLE_BYTES)
        )

        assert "PROTO" in evidence.basis, evidence.basis

    def test_container_stays_generic_without_extension(self, tmp_path: Path) -> None:
        """ZIP остаётся ZIP: разрешение .pt/.npz — задача detect(), не detect_content()."""
        target = _write(tmp_path, "model.pt", _ZIP_BYTES)

        assert FormatDetector.detect_content(target).format is FileFormat.ZIP_ARCHIVE
        assert FormatDetector.detect(target) is FileFormat.PYTORCH

    def test_unreadable_file_is_unknown(self, tmp_path: Path) -> None:
        """Недоступный файл → UNKNOWN без исключения наружу."""
        evidence = FormatDetector.detect_content(tmp_path / "no_such_file.pkl")

        assert evidence.format is FileFormat.UNKNOWN
        assert evidence.basis == ""

    def test_detect_content_bytes_matches_detect_content(self, tmp_path: Path) -> None:
        """Путь и байты дают одинаковый результат."""
        target = _write(tmp_path, "model.gguf", _GGUF_BYTES)

        assert (
            FormatDetector.detect_content_bytes(_GGUF_BYTES).format
            == FormatDetector.detect_content(target).format
        )


# ---------------------------------------------------------------------------
# 2. Расхождение расширения и содержимого
# ---------------------------------------------------------------------------


class TestMismatchComputation:
    """Расхождение считается только когда формат определён и не совпал."""

    def test_pickle_in_safetensors_is_mismatch(self, tmp_path: Path) -> None:
        """Классический вектор: pickle под именем .safetensors."""
        facts = build_format_facts(
            _write(tmp_path, "model.safetensors", _PICKLE_BYTES), _raw("safetensors")
        )

        assert facts.mismatch is True
        assert facts.detected_format is FileFormat.PICKLE

    def test_matching_format_is_not_mismatch(self, tmp_path: Path) -> None:
        """Валидный safetensors под своим именем — расхождения нет."""
        facts = build_format_facts(
            _write(tmp_path, "model.safetensors", _safetensors_bytes()),
            _raw("safetensors"),
        )

        assert facts.mismatch is False

    def test_unknown_content_is_not_mismatch(self, tmp_path: Path) -> None:
        """Неопознанное содержимое — зона MLS-PARSE-001, а не подмены формата."""
        facts = build_format_facts(
            _write(tmp_path, "model.safetensors", b"\x00\x01\x02"), _raw("safetensors")
        )

        assert facts.detected_format is FileFormat.UNKNOWN
        assert facts.mismatch is False

    def test_extension_without_promise_is_not_mismatch(self, tmp_path: Path) -> None:
        """``.bin`` ничего не обещает — нарушать нечего."""
        facts = build_format_facts(
            _write(tmp_path, "pytorch_model.bin", _PICKLE_BYTES), _raw("pytorch")
        )

        assert facts.expected_formats == frozenset()
        assert facts.mismatch is False

    @pytest.mark.parametrize("data", [_ZIP_BYTES, _PICKLE_BYTES])
    def test_pytorch_accepts_both_historical_layouts(
        self, tmp_path: Path, data: bytes
    ) -> None:
        """У .pt два легитимных представления: ZIP-контейнер и legacy pickle."""
        facts = build_format_facts(_write(tmp_path, "model.pt", data), _raw("pytorch"))

        assert facts.mismatch is False

    def test_compressed_joblib_is_not_mismatch(self, tmp_path: Path) -> None:
        """Сжатый joblib начинается с заголовка кодека — формат не опознан, не подмена."""
        facts = build_format_facts(
            _write(tmp_path, "model.joblib", b"\x78\x9c" + b"\x00" * 16), _raw("joblib")
        )

        assert facts.detected_format is FileFormat.UNKNOWN
        assert facts.mismatch is False


# ---------------------------------------------------------------------------
# 3. Классификация безопасности формата
# ---------------------------------------------------------------------------


class TestSafetyClassification:
    """Класс берётся строгий из двух источников: содержимое и сканер."""

    def test_safetensors_is_safe(self, tmp_path: Path) -> None:
        """Формат без исполнения кода."""
        facts = build_format_facts(
            _write(tmp_path, "m.safetensors", _safetensors_bytes()), _raw("safetensors")
        )

        assert facts.safety is FormatSafety.SAFE

    def test_pickle_is_code_bearing(self, tmp_path: Path) -> None:
        """Pickle исполняет код по построению формата."""
        facts = build_format_facts(_write(tmp_path, "m.pkl", _PICKLE_BYTES), _raw("pickle"))

        assert facts.safety is FormatSafety.CODE_BEARING
        assert facts.safety_rationale

    def test_content_overrides_safe_scanner(self, tmp_path: Path) -> None:
        """Pickle внутри .safetensors: сканер безопасный, содержимое — нет."""
        facts = build_format_facts(
            _write(tmp_path, "m.safetensors", _PICKLE_BYTES), _raw("safetensors")
        )

        assert facts.safety is FormatSafety.CODE_BEARING

    def test_scanner_overrides_unknown_content(self, tmp_path: Path) -> None:
        """Сжатый joblib: содержимое не опознано, но сканер знает про pickle внутри."""
        facts = build_format_facts(
            _write(tmp_path, "m.joblib", b"\x78\x9c" + b"\x00" * 16), _raw("joblib")
        )

        assert facts.safety is FormatSafety.CODE_BEARING

    def test_npz_container_resolved_by_scanner(self, tmp_path: Path) -> None:
        """ZIP сам по себе не классифицируется; .npz решает сканер numpy."""
        facts = build_format_facts(_write(tmp_path, "arr.npz", _ZIP_BYTES), _raw("numpy"))

        assert facts.safety is FormatSafety.SAFE

    def test_object_dtype_makes_numpy_code_bearing(self, tmp_path: Path) -> None:
        """dtype=object превращает безопасный .npy в code-bearing."""
        facts = build_format_facts(
            _write(tmp_path, "arr.npy", _NPY_BYTES),
            _raw("numpy", {META_OBJECT_DTYPE: "true"}),
        )

        assert facts.safety is FormatSafety.CODE_BEARING
        assert "object" in facts.safety_rationale

    def test_unknown_format_is_undetermined(self, tmp_path: Path) -> None:
        """Неопознанный формат не объявляется ни безопасным, ни опасным."""
        facts = build_format_facts(
            _write(tmp_path, "x.xyz", b"\x00\x01\x02"), _raw("unknown")
        )

        assert facts.safety is FormatSafety.UNDETERMINED


# ---------------------------------------------------------------------------
# 4. Полнота реестров
# ---------------------------------------------------------------------------


class TestSafetyRegistryIsComplete:
    """Ни один формат и ни один сканер не может остаться неклассифицированным."""

    def test_every_file_format_is_classified(self) -> None:
        """Каждое значение FileFormat имеет запись в FORMAT_SAFETY."""
        missing = sorted(fmt.value for fmt in FileFormat if fmt not in FORMAT_SAFETY)

        assert not missing, (
            f"Форматы без классификации безопасности: {missing}. Добавьте запись "
            f"в FORMAT_SAFETY с обоснованием — иначе новый формат молча выпадет "
            f"из правила require_safetensors."
        )

    def test_every_spec_has_title_and_rationale(self) -> None:
        """У каждой записи есть имя формата и обоснование класса."""
        for fmt, spec in FORMAT_SAFETY.items():
            assert spec.title.strip(), f"{fmt}: пустой title"
            assert spec.rationale.strip(), f"{fmt}: пустое обоснование"

    def test_every_registered_scanner_has_format(self) -> None:
        """Каждый зарегистрированный сканер сопоставлен формату."""
        registered = {scanner.name for scanner in ScannerRegistry.all_scanners()}
        if not registered:
            pytest.skip("Реестр сканеров пуст")

        missing = sorted(registered - set(SCANNER_FORMATS))

        assert not missing, (
            f"Сканеры без записи в SCANNER_FORMATS: {missing}. Без неё формат "
            f"файла не будет классифицирован по require_safetensors."
        )

    def test_scanner_formats_reference_known_formats(self) -> None:
        """Реестр сканеров ссылается только на классифицированные форматы."""
        for scanner_name, fmt in SCANNER_FORMATS.items():
            assert fmt in FORMAT_SAFETY, f"{scanner_name} → {fmt} вне FORMAT_SAFETY"

    def test_expected_formats_reference_known_formats(self) -> None:
        """Таблица ожиданий ссылается только на существующие форматы."""
        for ext, formats in EXPECTED_CONTENT_FORMATS.items():
            assert formats, f"{ext}: пустое множество ожидаемых форматов"
            for fmt in formats:
                assert fmt in FORMAT_SAFETY, f"{ext} → {fmt} вне FORMAT_SAFETY"

    def test_safetensors_is_the_reference_safe_format(self) -> None:
        """Формат, ради которого существует правило, обязан быть безопасным."""
        assert FORMAT_SAFETY[FileFormat.SAFETENSORS].safety is FormatSafety.SAFE


# ---------------------------------------------------------------------------
# 5. Запись и чтение фактов
# ---------------------------------------------------------------------------


class TestFactsRoundTrip:
    """Факты кладутся в metadata и читаются обратно без потерь."""

    def test_attach_then_read_back(self, tmp_path: Path) -> None:
        """``format_facts_of`` восстанавливает то, что положил ``attach``."""
        target = _write(tmp_path, "model.safetensors", _PICKLE_BYTES)
        raw = _raw("safetensors")

        attach_format_facts(raw, target)
        facts = format_facts_of(raw)

        assert facts is not None
        assert facts.declared_ext == ".safetensors"
        assert facts.detected_format is FileFormat.PICKLE
        assert facts.expected_formats == frozenset({FileFormat.SAFETENSORS})
        assert facts.mismatch is True
        assert facts.safety is FormatSafety.CODE_BEARING

    def test_attach_preserves_existing_metadata(self, tmp_path: Path) -> None:
        """Существующие ключи сканера не затираются."""
        target = _write(tmp_path, "arr.npy", _NPY_BYTES)
        raw = _raw("numpy", {META_OBJECT_DTYPE: "true"})

        attach_format_facts(raw, target)

        assert raw.metadata is not None
        assert raw.metadata[META_OBJECT_DTYPE] == "true"
        assert raw.metadata[META_DETECTED_FORMAT] == FileFormat.NUMPY_NPY.value

    def test_facts_keys_are_system_prefixed(self, tmp_path: Path) -> None:
        """Ключи фактов помечены «_» — их не разбирают metadata-детекторы.

        Без префикса GGUFMetadataDetector попытался бы искать URL в служебных
        значениях и выдал бы шум на каждом GGUF-файле.
        """
        target = _write(tmp_path, "model.gguf", _GGUF_BYTES)
        raw = _raw("gguf")

        attach_format_facts(raw, target)

        assert raw.metadata is not None
        added = [key for key in raw.metadata if key.startswith("_format")]
        assert added, "Факты формата не записаны"
        assert all(key.startswith("_") for key in added)

    def test_unsupported_flag_set_only_on_request(self, tmp_path: Path) -> None:
        """Флаг «формат не поддержан» ставится явно, а не угадывается."""
        target = _write(tmp_path, "x.xyz", b"\x00\x01")
        raw_supported = _raw("pickle")
        raw_unsupported = _raw("unknown")

        attach_format_facts(raw_supported, target)
        attach_format_facts(raw_unsupported, target, unsupported=True)

        assert raw_supported.metadata is not None
        assert raw_unsupported.metadata is not None
        assert META_FORMAT_UNSUPPORTED not in raw_supported.metadata
        assert raw_unsupported.metadata[META_FORMAT_UNSUPPORTED] is True

    def test_missing_facts_read_as_none(self) -> None:
        """RawScanData без фактов (сканер вызван напрямую) → None, а не догадки."""
        assert format_facts_of(_raw("pickle")) is None
        assert format_facts_of(_raw("pickle", {"other": "value"})) is None

    def test_mismatch_stored_as_bool(self, tmp_path: Path) -> None:
        """Флаг расхождения — булев, чтобы SecretsDetector его пропускал."""
        target = _write(tmp_path, "model.safetensors", _PICKLE_BYTES)
        raw = _raw("safetensors")

        attach_format_facts(raw, target)

        assert raw.metadata is not None
        assert raw.metadata[META_FORMAT_MISMATCH] is True

    def test_attach_survives_missing_file(self, tmp_path: Path) -> None:
        """Исчезнувший файл не роняет пайплайн — формат просто UNKNOWN."""
        raw = _raw("pickle")

        attach_format_facts(raw, tmp_path / "gone.pkl")

        facts = format_facts_of(raw)
        assert facts is not None
        assert facts.detected_format is FileFormat.UNKNOWN

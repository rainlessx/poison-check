"""Тесты для NumpyScanner.

Все fixture строятся вручную по бинарной спецификации .npy:
  [6 байт magic: b'\\x93NUMPY']
  [1 байт major] [1 байт minor]
  [2 байта header_len uint16 LE для v1, 4 байта uint32 LE для v2]
  [header_len байт ASCII Python-dict]
  [бинарные данные массива]

Для .npz: стандартный ZIP, содержащий .npy записи.
Вредоносные pickle-payload строятся через opcode-конструкцию (не pickle.dumps).
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from poison_check.scanners.numpy_scanner import (
    NumpyScanner,
    _is_object_dtype,
    _parse_npy_header,
    _find_pickle_magic,
)


# ---------------------------------------------------------------------------
# Вспомогательные builder-функции
# ---------------------------------------------------------------------------


def _make_npy_v1(dtype: str, shape: tuple[int, ...], data: bytes) -> bytes:
    """Строит .npy v1.0 файл из dtype, shape и бинарных данных.

    Заголовок выравнивается пробелами до кратности 64 байтам, согласно
    спецификации NumPy.
    """
    fortran_order = False
    # Добавляем byte-order prefix если нет (npy требует '<'/'>' для числовых dtype)
    if dtype in ("O", "object"):
        descr = "|O"
    elif dtype.startswith(("f", "i", "u")):
        descr = f"<{dtype}"
    else:
        descr = dtype

    header_dict = (
        f"{{'descr': '{descr}', 'fortran_order': {fortran_order}, "
        f"'shape': {shape}, }}"
    )
    # Выравниваем: 10 байт preamble + header + 1 байт '\n' кратно 64
    total_no_data = 10  # 6 magic + 1 major + 1 minor + 2 header_len
    header_no_newline = header_dict.encode("latin-1")
    pad_needed = 64 - ((total_no_data + len(header_no_newline) + 1) % 64)
    if pad_needed == 64:
        pad_needed = 0
    header_bytes = header_no_newline + b" " * pad_needed + b"\n"
    header_len = len(header_bytes)

    return (
        b"\x93NUMPY"
        + bytes([1, 0])                          # major=1, minor=0
        + struct.pack("<H", header_len)           # 2-байтовый uint16 LE
        + header_bytes
        + data
    )


def _make_npy_v2(dtype: str, shape: tuple[int, ...], data: bytes) -> bytes:
    """Строит .npy v2.0 файл (4-байтовый header_len)."""
    if dtype in ("O", "object"):
        descr = "|O"
    elif dtype.startswith(("f", "i", "u")):
        descr = f"<{dtype}"
    else:
        descr = dtype

    header_dict = (
        f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"
    )
    # Для v2: preamble = 12 байт
    total_no_data = 12
    header_no_newline = header_dict.encode("latin-1")
    pad_needed = 64 - ((total_no_data + len(header_no_newline) + 1) % 64)
    if pad_needed == 64:
        pad_needed = 0
    header_bytes = header_no_newline + b" " * pad_needed + b"\n"
    header_len = len(header_bytes)

    return (
        b"\x93NUMPY"
        + bytes([2, 0])
        + struct.pack("<I", header_len)           # 4-байтовый uint32 LE
        + header_bytes
        + data
    )


def _make_float32_npy(shape: tuple[int, ...] = (4,)) -> bytes:
    """Валидный .npy с float32 массивом (нулевые данные)."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    data = b"\x00" * (n_elements * 4)  # float32 = 4 байта
    return _make_npy_v1("f4", shape, data)


def _make_object_npy_no_pickle() -> bytes:
    """Object-dtype массив БЕЗ pickle — данные не начинаются с pickle magic."""
    # Реально NumPy сериализовал бы через pickle, но здесь
    # имитируем данные без pickle magic bytes (просто текст)
    data = b"some arbitrary bytes without pickle signature"
    return _make_npy_v1("O", (3,), data)


def _make_os_system_pickle() -> bytes:
    """Вредоносный pickle-payload: os.system через opcode-конструкцию."""
    return (
        b"\x80\x02"               # PROTO 2
        + b"cos\nsystem\n"        # GLOBAL os.system
        + b"("                    # MARK
        + b"t"                    # TUPLE → ()
        + b"R"                    # REDUCE
        + b"."                    # STOP
    )


def _make_os_system_pickle_proto0() -> bytes:
    """Вредоносный pickle proto 0 — БЕЗ \\x80 magic (regression test #8).

    Атакующий может явно указать ``protocol=0`` в pickle.dumps, чтобы обойти
    детекцию magic bytes ``\\x80\\xNN``. Этот payload начинается прямо с
    GLOBAL-opcode 'c'.
    """
    return (
        b"cos\nsystem\n"          # GLOBAL os.system
        + b"("                    # MARK
        + b"t"                    # TUPLE → ()
        + b"R"                    # REDUCE
        + b"."                    # STOP
    )


def _make_object_npy_with_pickle_proto0() -> bytes:
    """Object-dtype массив с proto-0 pickle payload."""
    return _make_npy_v1("O", (1,), _make_os_system_pickle_proto0())


def _make_object_npy_with_pickle() -> bytes:
    """Object-dtype массив С вредоносным pickle-payload в данных."""
    # В реальном атаке payload вложен внутри данных массива.
    # Здесь: данные = junk prefix + pickle magic + payload
    junk = b"\x00" * 8
    pickle_payload = _make_os_system_pickle()
    data = junk + pickle_payload
    return _make_npy_v1("O", (1,), data)


def _make_npz(*npy_contents: tuple[str, bytes]) -> bytes:
    """Строит .npz (ZIP) из пар (name, npy_bytes).

    Args:
        npy_contents: Пары (имя_без_расширения, npy_байты).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, content in npy_contents:
            zf.writestr(f"{name}.npy", content)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Тест 1: простой float32 array → нет issues, нет error
# ---------------------------------------------------------------------------


class TestNumpyScannerFloat32:
    """Простой числовой массив должен сканироваться без проблем."""

    def test_float32_no_error(self, tmp_path: Path) -> None:
        """Float32 .npy возвращает error=None."""
        f = tmp_path / "weights.npy"
        f.write_bytes(_make_float32_npy((3, 4)))

        result = NumpyScanner().scan(f)
        assert result.error is None

    def test_float32_tensor_info(self, tmp_path: Path) -> None:
        """Float32 .npy заполняет tensor_info с правильным dtype."""
        f = tmp_path / "weights.npy"
        f.write_bytes(_make_float32_npy((3, 4)))

        result = NumpyScanner().scan(f)
        assert result.tensor_info is not None
        assert len(result.tensor_info) == 1
        ti = result.tensor_info[0]
        assert ti.dtype == "f4"
        assert ti.shape == [3, 4]

    def test_float32_no_globals(self, tmp_path: Path) -> None:
        """Float32 .npy не содержит pickle globals."""
        f = tmp_path / "weights.npy"
        f.write_bytes(_make_float32_npy())

        result = NumpyScanner().scan(f)
        assert result.globals is None

    def test_float32_scanner_name(self, tmp_path: Path) -> None:
        """scanner_name == 'numpy'."""
        f = tmp_path / "arr.npy"
        f.write_bytes(_make_float32_npy())

        result = NumpyScanner().scan(f)
        assert result.scanner_name == "numpy"

    def test_float32_hashes(self, tmp_path: Path) -> None:
        """Хеши вычислены."""
        f = tmp_path / "arr.npy"
        f.write_bytes(_make_float32_npy())

        result = NumpyScanner().scan(f)
        assert "sha256" in result.file_hash
        assert len(result.file_hash["sha256"]) == 64

    def test_npy_v2_format(self, tmp_path: Path) -> None:
        """Формат .npy v2.0 (4-байтовый header_len) тоже парсится."""
        f = tmp_path / "v2.npy"
        f.write_bytes(_make_npy_v2("f4", (2, 2), b"\x00" * 16))

        result = NumpyScanner().scan(f)
        assert result.error is None
        assert result.tensor_info is not None
        ti = result.tensor_info[0]
        assert ti.dtype == "f4"
        assert ti.shape == [2, 2]


# ---------------------------------------------------------------------------
# Тест 2: object array без pickle → Issue MEDIUM
# ---------------------------------------------------------------------------


class TestNumpyScannerObjectDtype:
    """Object-dtype → факт object_dtype_detected в metadata.

    Сам Issue MLS-NPY-001 эмитит NumpyMetadataDetector (см.
    tests/test_numpy_metadata_detector.py) — сканер только фиксирует факты.
    """

    def test_object_dtype_error_or_metadata(self, tmp_path: Path) -> None:
        """Object-dtype вызывает предупреждение (object_dtype_detected в metadata)."""
        f = tmp_path / "obj.npy"
        f.write_bytes(_make_object_npy_no_pickle())

        result = NumpyScanner().scan(f)
        # Файл технически корректен — error не обязателен
        # Важно: object dtype отражён в metadata
        assert result.metadata is not None
        assert result.metadata.get("object_dtype_detected") == "true"

    def test_object_dtype_no_pickle_in_globals(self, tmp_path: Path) -> None:
        """Без pickle-payload глобалов os.system нет."""
        f = tmp_path / "obj.npy"
        f.write_bytes(_make_object_npy_no_pickle())

        result = NumpyScanner().scan(f)
        if result.globals:
            assert ("os", "system") not in result.globals

    def test_object_dtype_tensor_info(self, tmp_path: Path) -> None:
        """tensor_info содержит dtype=O для object массива."""
        f = tmp_path / "obj.npy"
        f.write_bytes(_make_object_npy_no_pickle())

        result = NumpyScanner().scan(f)
        assert result.tensor_info is not None
        assert any("O" in ti.dtype for ti in result.tensor_info)


# ---------------------------------------------------------------------------
# Тест 3: object array С pickle → globals содержат вредоносный вызов
# ---------------------------------------------------------------------------


class TestNumpyScannerObjectWithPickle:
    """Object-dtype + pickle-payload → рекурсивный анализ через PickleScanner."""

    def test_pickle_payload_globals_detected(self, tmp_path: Path) -> None:
        """Вредоносный os.system попадает в globals."""
        f = tmp_path / "malicious.npy"
        f.write_bytes(_make_object_npy_with_pickle())

        result = NumpyScanner().scan(f)

        assert result.globals is not None
        assert ("os", "system") in result.globals, (
            f"Ожидался ('os', 'system') в globals, получено: {result.globals!r}"
        )

    def test_pickle_payload_metadata_flag(self, tmp_path: Path) -> None:
        """pickle_payload_detected выставляется в metadata."""
        f = tmp_path / "malicious.npy"
        f.write_bytes(_make_object_npy_with_pickle())

        result = NumpyScanner().scan(f)
        assert result.metadata is not None
        assert result.metadata.get("pickle_payload_detected") == "true"

    def test_pickle_payload_object_dtype_also_flagged(self, tmp_path: Path) -> None:
        """object_dtype_detected тоже выставлен."""
        f = tmp_path / "malicious.npy"
        f.write_bytes(_make_object_npy_with_pickle())

        result = NumpyScanner().scan(f)
        assert result.metadata is not None
        assert result.metadata.get("object_dtype_detected") == "true"

    def test_pickle_payload_nested_files(self, tmp_path: Path) -> None:
        """Рекурсивный PickleScanner попадает в nested_files."""
        f = tmp_path / "malicious.npy"
        f.write_bytes(_make_object_npy_with_pickle())

        result = NumpyScanner().scan(f)
        assert result.nested_files is not None
        assert len(result.nested_files) >= 1

    def test_reduce_calls_detected(self, tmp_path: Path) -> None:
        """ReduceCall для os.system фиксируется."""
        f = tmp_path / "malicious.npy"
        f.write_bytes(_make_object_npy_with_pickle())

        result = NumpyScanner().scan(f)
        assert result.reduce_calls is not None
        found = any(
            rc.module == "os" and rc.name == "system"
            for rc in result.reduce_calls
        )
        assert found, f"Ожидался ReduceCall(os.system), получено: {result.reduce_calls!r}"

    def test_existing_fixture_npz(self) -> None:
        """Существующая вредоносная fixture payload_06_numpy.npz обнаруживается."""
        fixture = Path("tests/fixtures/malicious/payload_06_numpy.npz")
        if not fixture.exists():
            pytest.skip("Fixture не найдена")

        result = NumpyScanner().scan(fixture)
        # .npz с pickle-payload → не должен падать
        assert result is not None
        assert result.scanner_name == "numpy"

    def test_pickle_proto0_payload_detected(self, tmp_path: Path) -> None:
        """Регрессия аудита #8: pickle proto 0 (без \\x80 magic) детектируется.

        До фикса _find_pickle_magic искал только b'\\x80\\x02..\\x80\\x05', что
        пропускало object-массивы с явно указанным protocol=0/1.
        """
        f = tmp_path / "malicious_proto0.npy"
        f.write_bytes(_make_object_npy_with_pickle_proto0())

        result = NumpyScanner().scan(f)
        assert result.globals is not None, (
            "globals должны быть извлечены из pickle proto 0 payload"
        )
        assert ("os", "system") in result.globals, (
            f"Ожидался ('os', 'system') в globals proto 0, получено: {result.globals!r}"
        )
        assert result.metadata is not None
        assert result.metadata.get("pickle_payload_detected") == "true"


# ---------------------------------------------------------------------------
# Тест 4: .npz с несколькими массивами → все проверяются
# ---------------------------------------------------------------------------


class TestNumpyScannerNpz:
    """NPZ со несколькими массивами: все должны быть просканированы."""

    def test_npz_multiple_arrays_tensor_info(self, tmp_path: Path) -> None:
        """.npz с двумя float32 массивами → два TensorInfo."""
        data = _make_npz(
            ("layer_0", _make_float32_npy((8, 16))),
            ("layer_1", _make_float32_npy((16,))),
        )
        f = tmp_path / "model.npz"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result.tensor_info is not None
        assert len(result.tensor_info) == 2
        names = {ti.name for ti in result.tensor_info}
        assert names == {"layer_0", "layer_1"}

    def test_npz_no_error_for_clean(self, tmp_path: Path) -> None:
        """Чистый .npz без pickle → error=None."""
        data = _make_npz(
            ("a", _make_float32_npy((2, 2))),
            ("b", _make_float32_npy((4,))),
        )
        f = tmp_path / "clean.npz"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result.error is None

    def test_npz_malicious_array_detected(self, tmp_path: Path) -> None:
        """.npz с одним чистым и одним вредоносным массивом → os.system в globals."""
        data = _make_npz(
            ("weights", _make_float32_npy((4,))),
            ("labels", _make_object_npy_with_pickle()),
        )
        f = tmp_path / "mixed.npz"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result.globals is not None
        assert ("os", "system") in result.globals

    def test_npz_nested_files_count(self, tmp_path: Path) -> None:
        """nested_files содержит по одному RawScanData на каждый .npy."""
        data = _make_npz(
            ("a", _make_float32_npy()),
            ("b", _make_float32_npy()),
            ("c", _make_float32_npy()),
        )
        f = tmp_path / "three.npz"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result.nested_files is not None
        assert len(result.nested_files) == 3

    def test_npz_scanner_name(self, tmp_path: Path) -> None:
        """scanner_name == 'numpy' для .npz."""
        data = _make_npz(("arr", _make_float32_npy()))
        f = tmp_path / "single.npz"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result.scanner_name == "numpy"

    def test_npz_member_count_in_metadata(self, tmp_path: Path) -> None:
        """member_count присутствует в metadata."""
        data = _make_npz(
            ("x", _make_float32_npy()),
            ("y", _make_float32_npy()),
        )
        f = tmp_path / "two.npz"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result.metadata is not None
        assert result.metadata.get("member_count") == "2"


# ---------------------------------------------------------------------------
# Тест 4b: .npz распаковывается потоково через ContainerExtractor
# ---------------------------------------------------------------------------


class TestNumpyScannerNpzStreaming:
    """Регрессия: .npz грузился в RAM целиком через path.read_bytes().

    Старая реализация делала ``zipfile.ZipFile(io.BytesIO(path.read_bytes()))``
    и читала члены через ``zf.read()`` без лимитов — это DoS (decompression
    bomb: .npz — обычный ZIP). Теперь распаковка идёт через
    ContainerExtractor.extract_zip_members с MAX_MEMBER_SIZE / MAX_EXTRACT_SIZE
    и защитой от path-traversal.
    """

    def test_npz_uses_container_extractor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_scan_npz вызывает ContainerExtractor.extract_zip_members."""
        from poison_check.core.container import ContainerExtractor

        calls: list[Path] = []
        original = ContainerExtractor.extract_zip_members.__func__  # type: ignore[attr-defined]

        def _spy(cls: type[ContainerExtractor], path: Path):  # type: ignore[no-untyped-def]
            calls.append(path)
            return original(cls, path)

        monkeypatch.setattr(
            ContainerExtractor, "extract_zip_members", classmethod(_spy)
        )

        f = tmp_path / "model.npz"
        f.write_bytes(_make_npz(("a", _make_float32_npy())))

        result = NumpyScanner().scan(f)
        assert calls == [f], (
            f"Ожидался вызов extract_zip_members({f}), получено: {calls}"
        )
        assert result.tensor_info is not None

    def test_npz_never_reads_whole_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path.read_bytes не вызывается при сканировании .npz (потоковость)."""
        def _boom(self: Path) -> bytes:
            raise AssertionError(
                "Path.read_bytes() запрещён в _scan_npz — .npz читается потоково"
            )

        monkeypatch.setattr(Path, "read_bytes", _boom)

        f = tmp_path / "model.npz"
        f.write_bytes(
            _make_npz(("a", _make_float32_npy()), ("b", _make_float32_npy((4, 4))))
        )

        result = NumpyScanner().scan(f)
        assert result.error is None
        assert result.tensor_info is not None
        assert len(result.tensor_info) == 2

    def test_npz_member_over_limit_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Член архива больше MAX_MEMBER_SIZE → RawScanData с error, не Exception."""
        from poison_check.core.container import ContainerExtractor

        monkeypatch.setattr(ContainerExtractor, "MAX_MEMBER_SIZE", 64)

        f = tmp_path / "bomb.npz"
        f.write_bytes(_make_npz(("big", _make_float32_npy((1000,)))))

        result = NumpyScanner().scan(f)
        assert result.error is not None
        assert "распаковки" in result.error or "лимит" in result.error

    def test_npz_extract_limit_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Суммарное превышение MAX_EXTRACT_SIZE → error (защита от zip-bomb)."""
        from poison_check.core.container import ContainerExtractor

        monkeypatch.setattr(ContainerExtractor, "MAX_EXTRACT_SIZE", 128)

        f = tmp_path / "bomb2.npz"
        f.write_bytes(
            _make_npz(
                ("a", _make_float32_npy((100,))),
                ("b", _make_float32_npy((100,))),
            )
        )

        result = NumpyScanner().scan(f)
        assert result.error is not None

    def test_npz_traversal_member_skipped(self, tmp_path: Path) -> None:
        """Член с path-traversal именем пропускается ContainerExtractor."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("../../evil.npy", _make_object_npy_with_pickle())
            zf.writestr("clean.npy", _make_float32_npy())
        f = tmp_path / "traversal.npz"
        f.write_bytes(buf.getvalue())

        result = NumpyScanner().scan(f)
        assert result.metadata is not None
        assert result.metadata.get("member_count") == "1"
        assert result.tensor_info is not None
        assert [ti.name for ti in result.tensor_info] == ["clean"]


# ---------------------------------------------------------------------------
# Тест 5: повреждённые файлы → не крашится
# ---------------------------------------------------------------------------


class TestNumpyScannerMalformed:
    """Повреждённые/нестандартные .npy файлы не должны валить сканер.

    Устойчивость к известной проблеме разбора .npy, на которой парсеры
    могут падать: сканер никогда не бросает необработанных исключений.
    """

    def test_empty_file_no_crash(self, tmp_path: Path) -> None:
        """Пустой .npy не вызывает исключения."""
        f = tmp_path / "empty.npy"
        f.write_bytes(b"")

        result = NumpyScanner().scan(f)
        assert result is not None
        assert result.error is not None

    def test_only_magic_no_crash(self, tmp_path: Path) -> None:
        """Файл только с magic bytes (без заголовка) → error, не crash."""
        f = tmp_path / "magic_only.npy"
        f.write_bytes(b"\x93NUMPY")

        result = NumpyScanner().scan(f)
        assert result is not None
        assert result.error is not None

    def test_wrong_magic_no_crash(self, tmp_path: Path) -> None:
        """Файл с неверными magic bytes → error, не crash."""
        f = tmp_path / "bad_magic.npy"
        f.write_bytes(b"\xff\xfe\xfd\xfc\xfb\xfa" + b"\x01\x00" + b"\x10\x00" + b"{'x': 1}\n")

        result = NumpyScanner().scan(f)
        assert result is not None
        assert result.error is not None

    def test_truncated_header_no_crash(self, tmp_path: Path) -> None:
        """Заголовок обещает 1000 байт, файл заканчивается после 20 → error, не crash."""
        data = (
            b"\x93NUMPY"
            + bytes([1, 0])
            + struct.pack("<H", 1000)  # обещаем 1000 байт заголовка
            + b"{'descr': '<f4'"       # но пишем только 16
        )
        f = tmp_path / "truncated.npy"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result is not None
        assert result.error is not None

    def test_malformed_header_json_no_crash(self, tmp_path: Path) -> None:
        """Заголовок содержит не Python-dict → предупреждение, не crash."""
        bad_header = b"not a python dict!!!\n"
        # Добиваем до кратного 64
        pad = 64 - ((10 + len(bad_header)) % 64)
        if pad == 64:
            pad = 0
        header_bytes = bad_header + b" " * pad
        data = (
            b"\x93NUMPY"
            + bytes([1, 0])
            + struct.pack("<H", len(header_bytes))
            + header_bytes
            + b"\x00" * 16
        )
        f = tmp_path / "bad_header.npy"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result is not None
        # Не должен упасть; ошибка парсинга фиксируется в metadata или error
        # Важно: исключений нет

    def test_random_bytes_no_crash(self, tmp_path: Path) -> None:
        """Случайные байты с .npy расширением → error, не crash."""
        import os as _os
        f = tmp_path / "random.npy"
        # Начинаем с magic чтобы can_handle вернул True
        f.write_bytes(b"\x93NUMPY" + _os.urandom(200))

        result = NumpyScanner().scan(f)
        assert result is not None
        # Может быть error или нет, главное — не Exception

    def test_corrupted_npz_no_crash(self, tmp_path: Path) -> None:
        """Повреждённый ZIP (.npz) → error, не crash."""
        f = tmp_path / "corrupt.npz"
        f.write_bytes(b"PK\x03\x04" + b"\xff" * 100)  # ZIP header + мусор

        result = NumpyScanner().scan(f)
        assert result is not None
        assert result.error is not None

    def test_zero_header_len_no_crash(self, tmp_path: Path) -> None:
        """header_len=0 → пустой заголовок → dtype пустой, не crash."""
        data = (
            b"\x93NUMPY"
            + bytes([1, 0])
            + struct.pack("<H", 0)   # заголовок нулевой длины
            + b"\x00" * 16          # данные
        )
        f = tmp_path / "zero_header.npy"
        f.write_bytes(data)

        result = NumpyScanner().scan(f)
        assert result is not None
        # Возможна ошибка парсинга, но не Exception


# ---------------------------------------------------------------------------
# Тест 6: can_handle
# ---------------------------------------------------------------------------


class TestNumpyScannerCanHandle:
    """can_handle проверяет расширение и magic bytes."""

    def test_can_handle_npy_valid(self, tmp_path: Path) -> None:
        """Валидный .npy распознаётся."""
        f = tmp_path / "arr.npy"
        f.write_bytes(_make_float32_npy())
        assert NumpyScanner.can_handle(f) is True

    def test_can_handle_npz_valid(self, tmp_path: Path) -> None:
        """Валидный .npz распознаётся."""
        f = tmp_path / "archive.npz"
        f.write_bytes(_make_npz(("arr", _make_float32_npy())))
        assert NumpyScanner.can_handle(f) is True

    def test_can_handle_wrong_ext(self, tmp_path: Path) -> None:
        """Неверное расширение → False."""
        f = tmp_path / "model.pkl"
        f.write_bytes(_make_float32_npy())
        assert NumpyScanner.can_handle(f) is False

    def test_can_handle_npy_wrong_magic(self, tmp_path: Path) -> None:
        """.npy с неверными magic bytes → False."""
        f = tmp_path / "fake.npy"
        f.write_bytes(b"\x00\x00\x00\x00\x00\x00" + b"\x01\x00\x10\x00")
        assert NumpyScanner.can_handle(f) is False

    def test_can_handle_case_insensitive(self, tmp_path: Path) -> None:
        """Расширение регистронезависимо."""
        f = tmp_path / "ARR.NPY"
        f.write_bytes(_make_float32_npy())
        assert NumpyScanner.can_handle(f) is True


# ---------------------------------------------------------------------------
# Тест 7: вспомогательные функции (unit)
# ---------------------------------------------------------------------------


class TestNumpyScannerHelpers:
    """Unit-тесты вспомогательных функций модуля."""

    # --- _parse_npy_header ---

    def test_parse_header_float32(self) -> None:
        """Парсинг заголовка float32 массива."""
        h = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }"
        dtype, shape, err = _parse_npy_header(h)
        assert err is None
        assert dtype == "f4"
        assert shape == [3, 4]

    def test_parse_header_object(self) -> None:
        """Парсинг заголовка object массива."""
        h = "{'descr': '|O', 'fortran_order': False, 'shape': (5,), }"
        dtype, shape, err = _parse_npy_header(h)
        assert err is None
        assert dtype == "O"
        assert shape == [5]

    def test_parse_header_malformed(self) -> None:
        """Malformed заголовок → error строка, не исключение."""
        dtype, shape, err = _parse_npy_header("not a dict")
        assert err is not None

    def test_parse_header_empty(self) -> None:
        """Пустой заголовок → error, не исключение."""
        dtype, shape, err = _parse_npy_header("")
        assert err is not None

    # --- _is_object_dtype ---

    def test_is_object_dtype_O(self) -> None:
        assert _is_object_dtype("O") is True

    def test_is_object_dtype_pipe_O(self) -> None:
        """'|O' — стандартное NumPy обозначение object dtype, должен быть True."""
        assert _is_object_dtype("|O") is True  # функция стрипует '|' перед сравнением

    def test_is_object_dtype_object(self) -> None:
        assert _is_object_dtype("object") is True

    def test_is_object_dtype_f4(self) -> None:
        assert _is_object_dtype("f4") is False

    def test_is_object_dtype_i8(self) -> None:
        assert _is_object_dtype("i8") is False

    def test_is_object_dtype_O8(self) -> None:
        """O8 — object с явным размером."""
        assert _is_object_dtype("O8") is True

    # --- _find_pickle_magic ---

    def test_find_pickle_magic_proto2(self) -> None:
        """Pickle protocol 2 magic найден."""
        data = b"\x00" * 10 + b"\x80\x02" + b"\x00" * 5
        assert _find_pickle_magic(data) is True

    def test_find_pickle_magic_not_found(self) -> None:
        """Нет pickle magic → False."""
        data = b"\x00" * 50 + b"hello world"
        assert _find_pickle_magic(data) is False

    def test_find_pickle_magic_proto4(self) -> None:
        """Pickle protocol 4 magic найден."""
        data = b"\x80\x04rest of pickle"
        assert _find_pickle_magic(data) is True


# ---------------------------------------------------------------------------
# Регрессия аудита #2: NumpyScanner потоковый (не загружает .npy в RAM целиком)
# ---------------------------------------------------------------------------


class TestNumpyScannerStreaming:
    """Сканер не должен делать path.read_bytes() на больших non-object .npy."""

    def test_large_float_array_does_not_oom(self, tmp_path: Path) -> None:
        """Большой float-массив сканируется без чтения всех данных в RAM.

        Создаём .npy 32 МБ — мало для теста, но достаточно для проверки
        что _scan_npy_stream не делает full read. Ключевое: dtype != object,
        значит array_data вообще не должен загружаться.
        """
        # 32 МБ float32 = 8 миллионов элементов
        data = _make_npy_v1("f4", (8_000_000,), b"\x00" * (8_000_000 * 4))
        f = tmp_path / "big.npy"
        f.write_bytes(data)
        assert f.stat().st_size > 30 * 1024 * 1024

        result = NumpyScanner().scan(f)
        assert result.error is None
        assert result.tensor_info is not None
        assert result.tensor_info[0].shape == [8_000_000]
        # Не нужно никаких pickle-проверок для float-массива
        assert result.metadata is None or result.metadata.get("object_dtype_detected") != "true"

    def test_object_array_truncated_at_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Object-array больше лимита читается только до лимита (защита от OOM).

        После truncate в metadata должен быть флаг array_data_truncated=true.
        """
        from poison_check.scanners.numpy_scanner import NumpyScanner as _NS
        monkeypatch.setattr(_NS, "_MAX_OBJECT_ARRAY_BYTES", 64)

        # object-массив с pickle payload, но с длинным prefix мусора (>64 байт)
        long_data = b"\x00" * 200 + _make_os_system_pickle()
        npy = _make_npy_v1("O", (1,), long_data)
        f = tmp_path / "huge_obj.npy"
        f.write_bytes(npy)

        result = NumpyScanner().scan(f)
        assert result.metadata is not None
        assert result.metadata.get("array_data_truncated") == "true"


class TestSafetensorsFileSize:
    """Регрессия аудита #2: SafetensorsScanner вызывает _check_file_size."""

    def test_safetensors_oversize_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Файл больше max_file_size возвращает RawScanData с error, не падает."""
        from poison_check.scanners.safetensors_scanner import SafetensorsScanner

        scanner = SafetensorsScanner(max_file_size=64)  # 64 байта
        f = tmp_path / "big.safetensors"
        f.write_bytes(b"\x00" * 100)

        result = scanner.scan(f)
        assert result.error is not None
        assert "слишком большой" in result.error or "ГБ" in result.error

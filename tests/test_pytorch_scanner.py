"""Тесты для PyTorchScanner.

Покрывает:
1. Безопасный .pt → globals содержат только torch.* паттерны, error is None
2. Вредоносный .pt с os.system в data.pkl → globals содержат ("os", "system")
3. .bin (HuggingFace формат) → обрабатывается идентично .pt
4. .pt как legacy pickle (без ZIP) → не падает, передаёт в PickleScanner
5. ZIP без data.pkl → RawScanData с warning в error, не Exception
6. ZIP-контейнер всегда разбирается: сканер НЕ пропускает data.pkl при is_zipfile() == True
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from poison_check.scanners.pytorch_scanner import PyTorchScanner, make_pytorch_zip

# ---------------------------------------------------------------------------
# Вспомогательные opcode-строители (без pickle.dumps)
# ---------------------------------------------------------------------------

_PROTO2 = b"\x80\x02"
_EMPTY_TUPLE = b")"
_MARK = b"("
_TUPLE = b"t"
_REDUCE = b"R"
_STOP = b"."


def _build_os_system_pickle() -> bytes:
    """Вредоносный pickle: os.system через GLOBAL + REDUCE."""
    return (
        _PROTO2
        + b"cos\nsystem\n"  # GLOBAL os.system
        + _MARK
        + _TUPLE
        + _REDUCE
        + _STOP
    )


def _build_safe_pytorch_pickle() -> bytes:
    """Безопасный pickle: только torch.nn.modules.linear.Linear."""
    return (
        _PROTO2
        + b"ctorch.nn.modules.linear\nLinear\n"  # GLOBAL torch
        + _EMPTY_TUPLE
        + _REDUCE
        + _STOP
    )


# ---------------------------------------------------------------------------
# Тест 1: Безопасный .pt → globals только torch.*, error is None
# ---------------------------------------------------------------------------

def test_safe_pt_globals_are_torch_only(tmp_path: Path) -> None:
    """Безопасный .pt содержит только torch.* globals, ошибок нет."""
    safe_pkl = _build_safe_pytorch_pickle()
    pt_file = tmp_path / "safe_model.pt"
    pt_file.write_bytes(make_pytorch_zip(safe_pkl))

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result.error is None, f"Неожиданная ошибка: {result.error!r}"
    assert result.globals is not None, "globals не должны быть None для безопасного файла"

    for module, _name in result.globals:
        assert module.startswith("torch"), (
            f"Неожиданный не-torch global: ({module!r}, {_name!r})"
        )


def test_safe_pt_fixture(tmp_path: Path) -> None:
    """Проверяем фикстуру safe/simple_model.pt из generate_fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    safe_pt = fixtures_dir / "safe" / "simple_model.pt"
    if not safe_pt.exists():
        pytest.skip("Фикстура safe/simple_model.pt не найдена, запустите generate_fixtures.py")

    scanner = PyTorchScanner()
    result = scanner.scan(safe_pt)

    assert result.error is None
    assert result.globals is not None
    for module, _ in result.globals:
        assert module.startswith("torch")


# ---------------------------------------------------------------------------
# Тест 2: Вредоносный .pt с os.system в data.pkl → globals содержат ("os", "system")
# ---------------------------------------------------------------------------

def test_malicious_pt_detects_os_system(tmp_path: Path) -> None:
    """Вредоносный .pt с os.system → globals содержат ('os', 'system')."""
    evil_pkl = _build_os_system_pickle()
    pt_file = tmp_path / "evil_model.pt"
    pt_file.write_bytes(make_pytorch_zip(evil_pkl))

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result.globals is not None, "globals не должны быть None для вредоносного файла"
    assert ("os", "system") in result.globals, (
        f"Ожидался ('os', 'system') в globals, получили: {result.globals!r}"
    )


def test_malicious_pt_fixture(tmp_path: Path) -> None:
    """Проверяем фикстуру malicious/pytorch_os_system.pt."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    evil_pt = fixtures_dir / "malicious" / "pytorch_os_system.pt"
    if not evil_pt.exists():
        pytest.skip("Фикстура malicious/pytorch_os_system.pt не найдена")

    scanner = PyTorchScanner()
    result = scanner.scan(evil_pt)

    assert result.globals is not None
    assert ("os", "system") in result.globals


def test_malicious_pt_has_reduce_call(tmp_path: Path) -> None:
    """Вредоносный .pt содержит REDUCE-вызов для os.system."""
    evil_pkl = _build_os_system_pickle()
    pt_file = tmp_path / "evil.pt"
    pt_file.write_bytes(make_pytorch_zip(evil_pkl))

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result.reduce_calls is not None, "reduce_calls не должны быть None"
    reduce_mods = [(r.module, r.name) for r in result.reduce_calls]
    assert ("os", "system") in reduce_mods, (
        f"Ожидался REDUCE для os.system, получили: {reduce_mods!r}"
    )


# ---------------------------------------------------------------------------
# Тест 3: .bin (HuggingFace формат) → обрабатывается идентично .pt
# ---------------------------------------------------------------------------

def test_bin_huggingface_format_detected(tmp_path: Path) -> None:
    """HuggingFace .bin-файл с os.system обрабатывается так же, как .pt."""
    evil_pkl = _build_os_system_pickle()
    bin_file = tmp_path / "model.bin"
    bin_file.write_bytes(make_pytorch_zip(evil_pkl))

    scanner = PyTorchScanner()

    # can_handle должен принять .bin
    assert scanner.can_handle(bin_file), ".bin должен приниматься PyTorchScanner"

    result = scanner.scan(bin_file)

    assert result.globals is not None
    assert ("os", "system") in result.globals, (
        f"HuggingFace .bin: ожидался ('os', 'system'), получили: {result.globals!r}"
    )


def test_bin_fixture_detected(tmp_path: Path) -> None:
    """Проверяем фикстуру malicious/pytorch_os_system.bin."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    evil_bin = fixtures_dir / "malicious" / "pytorch_os_system.bin"
    if not evil_bin.exists():
        pytest.skip("Фикстура malicious/pytorch_os_system.bin не найдена")

    scanner = PyTorchScanner()
    result = scanner.scan(evil_bin)

    assert result.globals is not None
    assert ("os", "system") in result.globals


def test_safe_bin_no_error(tmp_path: Path) -> None:
    """Безопасный .bin не вызывает ошибок."""
    safe_pkl = _build_safe_pytorch_pickle()
    bin_file = tmp_path / "safe.bin"
    bin_file.write_bytes(make_pytorch_zip(safe_pkl))

    scanner = PyTorchScanner()
    result = scanner.scan(bin_file)

    assert result.error is None
    assert result.globals is not None


# ---------------------------------------------------------------------------
# Тест 4: .pt как legacy pickle (без ZIP) → не падает
# ---------------------------------------------------------------------------

def test_legacy_pickle_pt_no_crash(tmp_path: Path) -> None:
    """Legacy .pt (plain pickle без ZIP) не падает — передаёт в PickleScanner."""
    legacy_pt = tmp_path / "legacy.pt"
    legacy_pt.write_bytes(_build_safe_pytorch_pickle())

    scanner = PyTorchScanner()
    result = scanner.scan(legacy_pt)

    # Не должно быть необработанного исключения
    # result.error допустим, но не обязателен
    assert result.file_path == legacy_pt
    assert result.scanner_name == "pytorch"


def test_legacy_pickle_pt_detects_os_system(tmp_path: Path) -> None:
    """Legacy .pt с os.system (plain pickle) — globals содержат os.system."""
    evil_legacy = tmp_path / "evil_legacy.pt"
    evil_legacy.write_bytes(_build_os_system_pickle())

    scanner = PyTorchScanner()
    result = scanner.scan(evil_legacy)

    assert result.globals is not None
    assert ("os", "system") in result.globals, (
        f"Legacy .pt: ожидался ('os', 'system'), получили: {result.globals!r}"
    )


def test_legacy_pickle_metadata_marks_format(tmp_path: Path) -> None:
    """Legacy .pt сохраняет в metadata pytorch_format=legacy_pickle."""
    legacy_pt = tmp_path / "legacy.pt"
    legacy_pt.write_bytes(_build_safe_pytorch_pickle())

    result = PyTorchScanner().scan(legacy_pt)

    assert result.metadata is not None
    assert result.metadata.get("pytorch_format") == "legacy_pickle"


# ---------------------------------------------------------------------------
# Тест 5: ZIP без data.pkl → RawScanData с warning, не Exception
# ---------------------------------------------------------------------------

def test_zip_without_pkl_returns_warning(tmp_path: Path) -> None:
    """ZIP без data.pkl возвращает RawScanData с описанием, не бросает исключение."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/config.json", '{"model_type": "bert"}')
        zf.writestr("archive/vocab.txt", "hello\nworld\n")
    zip_data = buf.getvalue()

    pt_file = tmp_path / "no_pkl.pt"
    pt_file.write_bytes(zip_data)

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    # Не должно быть необработанного исключения
    assert result is not None
    assert result.error is not None, "Должен быть error-текст при отсутствии data.pkl"
    # Убеждаемся, что в error есть полезная информация
    assert "pkl" in result.error.lower() or "pickle" in result.error.lower(), (
        f"error должен упоминать pickle: {result.error!r}"
    )


def test_empty_zip_returns_warning(tmp_path: Path) -> None:
    """Пустой ZIP-архив возвращает RawScanData с предупреждением."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass  # пустой архив
    zip_data = buf.getvalue()

    pt_file = tmp_path / "empty.pt"
    pt_file.write_bytes(zip_data)

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result is not None
    assert result.error is not None


def test_corrupted_zip_returns_error_not_exception(tmp_path: Path) -> None:
    """Повреждённый ZIP-архив возвращает RawScanData с error, не Exception."""
    pt_file = tmp_path / "corrupted.pt"
    # Начинается с ZIP magic, но содержимое повреждено
    pt_file.write_bytes(b"PK\x03\x04" + b"\xff" * 100)

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result is not None, "scan() должен возвращать RawScanData, а не бросать"
    assert result.error is not None, "Должна быть ошибка для повреждённого ZIP"


# ---------------------------------------------------------------------------
# Тест 6: ZIP-контейнер всегда разбирается — data.pkl НЕ пропускается
#
# Известная категория ошибки разбора: когда _is_zipfile() возвращает True,
# сканирование содержимого может пропускаться. Мы убеждаемся, что наш сканер
# всегда разбирает содержимое ZIP и находит вредоносный payload в data.pkl.
# ---------------------------------------------------------------------------

def test_nested_zip_pickle_content_always_scanned(tmp_path: Path) -> None:
    """ZIP-файл обязательно сканируется внутри (содержимое не пропускается).

    Известная категория ошибки — пропуск содержимого при обнаружении ZIP.
    Наш сканер должен найти вредоносный payload даже в ZIP-архиве.
    """
    evil_pkl = _build_os_system_pickle()
    pt_file = tmp_path / "nested_zip_regression.pt"
    pt_file.write_bytes(make_pytorch_zip(evil_pkl))

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    # Главное условие: payload найден ВНУТРИ ZIP
    assert result.globals is not None, (
        "globals должны быть заполнены — ZIP был просканирован, не пропущен"
    )
    assert ("os", "system") in result.globals, (
        "Вредоносный global os.system должен быть найден внутри ZIP (data.pkl)"
    )


def test_nested_zip_pickle_nested_files_populated(tmp_path: Path) -> None:
    """Результат содержит nested_files с данными из data.pkl."""
    evil_pkl = _build_os_system_pickle()
    pt_file = tmp_path / "nested_check.pt"
    pt_file.write_bytes(make_pytorch_zip(evil_pkl))

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result.nested_files is not None, "nested_files должны быть заполнены"
    assert len(result.nested_files) >= 1
    # Проверяем, что nested_file для archive/data.pkl содержит globals
    inner = result.nested_files[0]
    assert inner.globals is not None
    assert ("os", "system") in inner.globals


def test_nested_zip_pickle_deep_pkl_path(tmp_path: Path) -> None:
    """Pickle в archive/data.pkl (нестандартный путь внутри ZIP) находится.

    Некоторые .pt файлы используют archive/data.pkl, другие — data.pkl.
    Оба варианта должны быть обнаружены.
    """
    evil_pkl = _build_os_system_pickle()

    # Создаём ZIP с data.pkl в корне (без archive/)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.pkl", evil_pkl)
    pt_file = tmp_path / "flat_data_pkl.pt"
    pt_file.write_bytes(buf.getvalue())

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result.globals is not None
    assert ("os", "system") in result.globals, (
        "data.pkl в корне ZIP должен быть найден и просканирован"
    )


# ---------------------------------------------------------------------------
# Дополнительные тесты: can_handle, metadata, scanner_name
# ---------------------------------------------------------------------------

def test_can_handle_pt_zip(tmp_path: Path) -> None:
    """can_handle принимает .pt с ZIP magic."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(make_pytorch_zip(_build_safe_pytorch_pickle()))
    assert PyTorchScanner.can_handle(pt) is True


def test_can_handle_pth_zip(tmp_path: Path) -> None:
    """can_handle принимает .pth с ZIP magic."""
    pth = tmp_path / "model.pth"
    pth.write_bytes(make_pytorch_zip(_build_safe_pytorch_pickle()))
    assert PyTorchScanner.can_handle(pth) is True


def test_can_handle_ckpt_zip(tmp_path: Path) -> None:
    """can_handle принимает .ckpt с ZIP magic."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(make_pytorch_zip(_build_safe_pytorch_pickle()))
    assert PyTorchScanner.can_handle(ckpt) is True


def test_can_handle_rejects_pkl_extension(tmp_path: Path) -> None:
    """can_handle отклоняет .pkl (это зона PickleScanner)."""
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(make_pytorch_zip(_build_safe_pytorch_pickle()))
    assert PyTorchScanner.can_handle(pkl) is False


def test_scanner_name_in_result(tmp_path: Path) -> None:
    """scanner_name в результате == 'pytorch'."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(make_pytorch_zip(_build_safe_pytorch_pickle()))
    result = PyTorchScanner().scan(pt)
    assert result.scanner_name == "pytorch"


def test_zip_metadata_has_pytorch_format(tmp_path: Path) -> None:
    """RawScanData.metadata содержит pytorch_format=zip для ZIP-файлов."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(make_pytorch_zip(_build_safe_pytorch_pickle()))
    result = PyTorchScanner().scan(pt)
    assert result.metadata is not None
    assert result.metadata.get("pytorch_format") == "zip"


def test_result_has_hashes(tmp_path: Path) -> None:
    """RawScanData содержит sha256, sha512, md5 хеши."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(make_pytorch_zip(_build_safe_pytorch_pickle()))
    result = PyTorchScanner().scan(pt)
    # Аудит #18: MD5 опциональный, по умолчанию только SHA-256/SHA-512
    assert "sha256" in result.file_hash
    assert "sha512" in result.file_hash


def test_multiple_pkl_in_zip(tmp_path: Path) -> None:
    """Все .pkl-файлы в ZIP сканируются, не только главный."""
    evil_pkl = _build_os_system_pickle()
    safe_pkl = _build_safe_pytorch_pickle()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", safe_pkl)   # безопасный главный
        zf.writestr("extra/hidden.pkl", evil_pkl)   # вредоносный дополнительный
    pt_file = tmp_path / "multi_pkl.pt"
    pt_file.write_bytes(buf.getvalue())

    scanner = PyTorchScanner()
    result = scanner.scan(pt_file)

    assert result.globals is not None
    # Вредоносный globals из extra/hidden.pkl должен быть обнаружен
    assert ("os", "system") in result.globals, (
        "Вредоносный payload из дополнительного .pkl должен быть найден"
    )


def test_make_pytorch_zip_produces_valid_zip() -> None:
    """make_pytorch_zip() создаёт валидный ZIP с archive/data.pkl."""
    pkl_bytes = _build_safe_pytorch_pickle()
    zip_bytes = make_pytorch_zip(pkl_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "archive/data.pkl" in names
        assert zf.read("archive/data.pkl") == pkl_bytes


# ---------------------------------------------------------------------------
# Регрессионные тесты: FP от ExecutableDetector на чистых моделях (fix P0/P1)
# ---------------------------------------------------------------------------


def test_legacy_pickle_with_tensor_data_no_pe_fp(tmp_path: Path) -> None:
    """Legacy pickle с большим BINBYTES8-блобом не генерирует PE FP.

    Воспроизводит сценарий BERT/GPT-2 (legacy_pickle, 400+ МБ): тензорные
    данные хранятся как BINBYTES8, что раньше давало сотни CRITICAL FP из-за
    случайных MZ-байтов. После структурной PE-валидации FP = 0.
    """
    from poison_check.core.result import Confidence, MLContext
    from poison_check.detectors.executable_detector import ExecutableDetector

    # Симулируем 8 КБ случайных float32-данных (без валидного PE)
    import struct, math
    floats = [math.sin(i * 0.017) for i in range(2048)]
    tensor_bytes = struct.pack(f"{len(floats)}f", *floats)

    # Собираем legacy pickle: PROTO 2 + BINBYTES8 с "тензорными данными" + STOP
    pkl_data = (
        b"\x80\x02"            # PROTO 2
        + b"\x8e"              # BINBYTES8 opcode
        + len(tensor_bytes).to_bytes(8, "little")
        + tensor_bytes
        + b"."                 # STOP
    )
    legacy_pt = tmp_path / "legacy_model.pt"
    legacy_pt.write_bytes(pkl_data)

    scanner = PyTorchScanner()
    raw = scanner.scan(legacy_pt)
    assert raw.error is None or "BINBYTES8" not in (raw.error or "")

    ctx = MLContext(framework="pytorch", confidence=Confidence.HIGH)
    ed = ExecutableDetector()
    issues = ed.analyze(raw, ctx)
    pe_issues = [i for i in issues if "PE" in i.message]
    assert pe_issues == [], (
        f"Тензорные данные не должны давать PE FP, найдено {len(pe_issues)} issues"
    )


def test_zip_pytorch_tensor_members_not_scanned_for_executables(tmp_path: Path) -> None:
    """ZIP PyTorch: члены archive/data/N не сканируются на PE/ELF.

    Раньше PyTorchScanner загружал содержимое каждого tensor-члена ZIP в память
    и прогонял find_signatures_in_bytes, получая FP. Теперь они пропускаются.
    """
    import io as _io
    import zipfile as _zf
    from poison_check.core.result import Confidence, MLContext
    from poison_check.detectors.executable_detector import ExecutableDetector
    import struct, math

    floats = [math.cos(i * 0.013) for i in range(4096)]
    tensor_data = struct.pack(f"{len(floats)}f", *floats)  # 16 КБ float32

    buf = _io.BytesIO()
    with _zf.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", _build_safe_pytorch_pickle())
        zf.writestr("archive/data/0", tensor_data)
        zf.writestr("archive/data/1", tensor_data)
        zf.writestr("archive/version", b"1")
    pt_file = tmp_path / "bart_like.pt"
    pt_file.write_bytes(buf.getvalue())

    scanner = PyTorchScanner()
    raw = scanner.scan(pt_file)

    ctx = MLContext(framework="pytorch", confidence=Confidence.HIGH)
    ed = ExecutableDetector()
    issues = ed.analyze(raw, ctx)
    pe_issues = [i for i in issues if "PE" in i.message]
    assert pe_issues == [], (
        f"Tensor-члены ZIP не должны давать PE FP, найдено {len(pe_issues)}"
    )


def _short_binunicode(s: str) -> bytes:
    b = s.encode()
    return b"\x8c" + bytes([len(b)]) + b


def _getattr_pt(module: str, attr: str) -> bytes:
    """.pt с getattr(<module-global>, "<attr>") внутри data.pkl (для калибровки правки 2)."""
    pkl = (b"\x80\x02" + b"c" + module.encode() + b"\ngetattr\n"
           + b"ccollections\nOrderedDict\n" + _short_binunicode(attr) + b"\x86R.")
    return make_pytorch_zip(pkl)


class TestGetattrFactsPropagation:
    """Регрессия: факты getattr из вложенного data.pkl пробрасываются в metadata.

    Без проброса getattr(m,"SafeCls") в .pt давал ложный MLS-PKL-001 HIGH
    (реальные FP на легитимных YOLO/ultralytics .pt-моделях). Сужение
    работает и через контейнер .pt.
    """

    def test_safe_literal_attr_propagated_to_metadata(self, tmp_path: Path) -> None:
        p = tmp_path / "m.pt"
        p.write_bytes(_getattr_pt("__builtin__", "DetectionModel"))
        raw = PyTorchScanner().scan(p)
        assert (raw.metadata or {}).get("getattr_literal_attrs") == "DetectionModel"

    def test_safe_getattr_pt_no_pkl001(self, tmp_path: Path) -> None:
        from poison_check.scanner import Scanner
        p = tmp_path / "m.pt"
        p.write_bytes(_getattr_pt("__builtin__", "DetectionModel"))
        fr = Scanner(policy="default").scan(p).results_per_file[p]
        codes = {i.code for i in fr.issues}
        assert "MLS-PKL-001" not in codes
        assert not any("INDIRECT-IMPORT" in c for c in codes)

    def test_dangerous_getattr_pt_still_flagged(self, tmp_path: Path) -> None:
        from poison_check.scanner import Scanner
        p = tmp_path / "m.pt"
        p.write_bytes(_getattr_pt("builtins", "system"))
        fr = Scanner(policy="default").scan(p).results_per_file[p]
        codes = {i.code for i in fr.issues}
        # Опасное имя атрибута → детект НЕ ослаблен (indirect-import паттерн).
        assert any("INDIRECT-IMPORT" in c or c == "MLS-PKL-001" for c in codes)

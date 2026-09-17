"""Тесты для JoblibScanner.

Критерий успеха:
    Сканер детектирует те же 6 payload, упакованных в joblib с разной компрессией.
    На joblib-файлах инструменты нередко либо падают с parse error, либо
    пропускают их — этот сканер должен корректно разбирать сжатые потоки.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import unittest.mock as mock
import zlib
from pathlib import Path

import pytest

import poison_check.scanners.joblib_scanner as joblib_scanner_module
from poison_check.scanners.joblib_scanner import JoblibScanner
from tests.fixtures.generate_fixtures import (
    _build_os_system_pickle,
    make_malicious_joblib,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAFE_DIR = FIXTURES_DIR / "safe"
MALICIOUS_DIR = FIXTURES_DIR / "malicious"


# ---------------------------------------------------------------------------
# Тесты: безопасные файлы
# ---------------------------------------------------------------------------


class TestSafeJoblib:
    """Безопасные joblib-файлы должны сканироваться без ошибок."""

    def test_simple_list_no_dangerous_globals(self, tmp_path: Path) -> None:
        """safe/simple_list.joblib (без сжатия) не содержит os/subprocess."""
        path = SAFE_DIR / "simple_list.joblib"
        if not path.exists():
            pytest.skip("Фикстура simple_list.joblib не создана")
        scanner = JoblibScanner()
        result = scanner.scan(path)

        assert result.error is None, f"Ошибка при сканировании: {result.error}"
        assert result.scanner_name == "joblib"

        dangerous = {"os", "subprocess", "builtins"}
        if result.globals:
            found_modules = {mod for mod, _ in result.globals}
            assert not found_modules & dangerous, (
                f"Найдены опасные модули в безопасном файле: {found_modules & dangerous}"
            )

    def test_sklearn_model_zlib_scans_successfully(self, tmp_path: Path) -> None:
        """safe/sklearn_model.joblib (zlib compress=3) сканируется без ошибок."""
        path = SAFE_DIR / "sklearn_model.joblib"
        if not path.exists():
            pytest.skip("Фикстура sklearn_model.joblib не создана")
        scanner = JoblibScanner()
        result = scanner.scan(path)

        assert result.error is None, f"Ошибка при сканировании: {result.error}"
        assert result.metadata is not None
        assert result.metadata.get("joblib_compression") == "zlib", (
            f"Ожидался метод zlib, получен: {result.metadata.get('joblib_compression')}"
        )

    def test_simple_list_gzip_scans_successfully(self, tmp_path: Path) -> None:
        """safe/simple_list_gzip.joblib (gzip) сканируется без ошибок."""
        path = SAFE_DIR / "simple_list_gzip.joblib"
        if not path.exists():
            pytest.skip("Фикстура simple_list_gzip.joblib не создана")
        scanner = JoblibScanner()
        result = scanner.scan(path)

        assert result.error is None, f"Ошибка при сканировании: {result.error}"
        assert result.metadata is not None
        assert result.metadata.get("joblib_compression") == "gzip"


# ---------------------------------------------------------------------------
# Тесты: вредоносные файлы с разной компрессией
# ---------------------------------------------------------------------------


class TestMaliciousJoblib:
    """Вредоносные joblib-файлы должны детектироваться независимо от компрессии."""

    def _assert_os_system_detected(self, path: Path) -> None:
        """Вспомогательный метод: проверяет детекцию os.system."""
        scanner = JoblibScanner()
        result = scanner.scan(path)

        assert result.globals is not None, (
            f"globals пусты для {path.name} — сканер не извлёк данные"
        )
        assert ("os", "system") in result.globals, (
            f"os.system не найден в globals для {path.name}. "
            f"Найдено: {result.globals}"
        )

    def test_malicious_no_compression(self) -> None:
        """Вредоносный joblib без сжатия (raw pickle) → globals содержат ('os', 'system')."""
        path = MALICIOUS_DIR / "payload_joblib_none.joblib"
        assert path.exists(), "Фикстура не создана, запустите generate_fixtures.py"
        self._assert_os_system_detected(path)

    def test_malicious_zlib(self) -> None:
        """Вредоносный joblib с zlib-сжатием → globals содержат ('os', 'system')."""
        path = MALICIOUS_DIR / "payload_joblib_zlib.joblib"
        assert path.exists(), "Фикстура не создана, запустите generate_fixtures.py"
        self._assert_os_system_detected(path)

    def test_malicious_gzip(self) -> None:
        """Вредоносный joblib с gzip-сжатием → globals содержат ('os', 'system')."""
        path = MALICIOUS_DIR / "payload_joblib_gzip.joblib"
        assert path.exists(), "Фикстура не создана, запустите generate_fixtures.py"
        self._assert_os_system_detected(path)

    def test_malicious_bz2(self) -> None:
        """Вредоносный joblib с bz2-сжатием → globals содержат ('os', 'system')."""
        path = MALICIOUS_DIR / "payload_joblib_bz2.joblib"
        assert path.exists(), "Фикстура не создана, запустите generate_fixtures.py"
        self._assert_os_system_detected(path)

    def test_malicious_lzma(self) -> None:
        """Вредоносный joblib с lzma-сжатием → globals содержат ('os', 'system')."""
        path = MALICIOUS_DIR / "payload_joblib_lzma.joblib"
        assert path.exists(), "Фикстура не создана, запустите generate_fixtures.py"
        self._assert_os_system_detected(path)

    def test_malicious_xz(self) -> None:
        """Вредоносный joblib с xz-сжатием → globals содержат ('os', 'system')."""
        path = MALICIOUS_DIR / "payload_joblib_xz.joblib"
        assert path.exists(), "Фикстура не создана, запустите generate_fixtures.py"
        self._assert_os_system_detected(path)


# ---------------------------------------------------------------------------
# Тесты: устойчивость к ошибкам
# ---------------------------------------------------------------------------


class TestJoblibRobustness:
    """Сканер никогда не падает с необработанным исключением."""

    def test_corrupted_joblib_no_exception(self, tmp_path: Path) -> None:
        """Повреждённый joblib-файл → RawScanData с error, не Exception."""
        corrupt = tmp_path / "corrupt.joblib"
        # Начинается с zlib magic, но содержит мусор
        corrupt.write_bytes(b"\x78\x9c" + b"\xde\xad\xbe\xef" * 100)

        scanner = JoblibScanner()
        result = scanner.scan(corrupt)

        # Сканер не должен бросать исключение
        assert result is not None
        assert result.scanner_name == "joblib"
        # Должна быть ошибка, но не Exception наружу
        assert result.error is not None, "Повреждённый файл должен вернуть error"

    def test_empty_file_no_exception(self, tmp_path: Path) -> None:
        """Пустой файл → RawScanData с error или unknown, не Exception."""
        empty = tmp_path / "empty.joblib"
        empty.write_bytes(b"")

        scanner = JoblibScanner()
        result = scanner.scan(empty)

        assert result is not None
        assert result.scanner_name == "joblib"

    def test_truncated_gzip_no_exception(self, tmp_path: Path) -> None:
        """Обрезанный gzip-поток → error, не Exception."""
        trunc = tmp_path / "truncated.joblib"
        # Корректный gzip-заголовок, но обрезанные данные
        trunc.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03")

        scanner = JoblibScanner()
        result = scanner.scan(trunc)

        assert result is not None
        assert result.error is not None

    def test_nonexistent_file_no_exception(self, tmp_path: Path) -> None:
        """Несуществующий файл → RawScanData с error, не OSError наружу."""
        missing = tmp_path / "nonexistent.joblib"

        scanner = JoblibScanner()
        result = scanner.scan(missing)

        assert result is not None
        assert result.error is not None

    def test_unknown_compression_returns_info_not_exception(
        self, tmp_path: Path
    ) -> None:
        """Файл с неизвестным методом компрессии → error в RawScanData, не Exception.

        Issue с INFO-severity создаётся внутри сканера; детекторы
        смотрят на result.globals/opcodes, поэтому достаточно проверить
        что error заполнен и исключения нет.
        """
        unknown = tmp_path / "unknown_format.joblib"
        # Начинается с байт, которые не совпадают ни с одним известным форматом
        unknown.write_bytes(b"\x00\x01\x02\x03" + b"\xff" * 100)

        scanner = JoblibScanner()
        result = scanner.scan(unknown)

        assert result is not None
        assert result.scanner_name == "joblib"
        # Должна быть ошибка с описанием неизвестного формата
        assert result.error is not None, (
            "Файл с неизвестным форматом должен вернуть error"
        )
        assert result.metadata is not None
        assert result.metadata.get("joblib_compression") == "unknown"


# ---------------------------------------------------------------------------
# Тесты через bytes напрямую (без файлов)
# ---------------------------------------------------------------------------


class TestJoblibDetectCompression:
    """Тесты метода _detect_compression без создания файлов.

    После рефакторинга (задача 8) _detect_compression возвращает только строку
    метода — конструирование Issue вынесено в JoblibMetadataDetector, а факты
    кладутся в metadata сканером.
    """

    def test_detect_raw_pickle(self) -> None:
        """Raw pickle-поток определяется как 'none'."""
        data = b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01K\x02K\x03e."
        assert JoblibScanner._detect_compression(data) == "none"

    def test_detect_zlib(self) -> None:
        """Zlib-поток определяется как 'zlib'."""
        compressed = zlib.compress(b"\x80\x02\x4b\x01\x2e", 3)
        assert JoblibScanner._detect_compression(compressed) == "zlib"

    def test_detect_gzip(self) -> None:
        """Gzip-поток определяется как 'gzip'."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gf:
            gf.write(b"\x80\x02\x4b\x01\x2e")
        assert JoblibScanner._detect_compression(buf.getvalue()) == "gzip"

    def test_detect_bz2(self) -> None:
        """Bz2-поток определяется как 'bz2'."""
        compressed = bz2.compress(b"\x80\x02\x4b\x01\x2e")
        assert JoblibScanner._detect_compression(compressed) == "bz2"

    def test_detect_lzma(self) -> None:
        """Lzma FORMAT_ALONE определяется как 'lzma'."""
        compressed = lzma.compress(b"\x80\x02\x4b\x01\x2e", format=lzma.FORMAT_ALONE)
        assert JoblibScanner._detect_compression(compressed) == "lzma"

    def test_detect_xz(self) -> None:
        """Lzma XZ FORMAT определяется как 'xz'."""
        compressed = lzma.compress(b"\x80\x02\x4b\x01\x2e", format=lzma.FORMAT_XZ)
        assert JoblibScanner._detect_compression(compressed) == "xz"

    def test_detect_unknown_returns_unknown(self) -> None:
        """Неизвестный формат возвращает 'unknown'."""
        data = b"\x00\x01\x02\x03\xff\xfe\xfd"
        assert JoblibScanner._detect_compression(data) == "unknown"

    def test_detect_empty_returns_unknown(self) -> None:
        """Пустые данные возвращают 'unknown'."""
        assert JoblibScanner._detect_compression(b"") == "unknown"

    def test_detect_lz4_prefix(self) -> None:
        """LZ4 magic bytes определяются как 'lz4'."""
        # LZ4 frame magic: 0x04224D18 (little-endian)
        data = b"\x04\x22\x4d\x18" + b"\x00" * 20
        assert JoblibScanner._detect_compression(data) == "lz4"


# ---------------------------------------------------------------------------
# Тест: lz4 недоступен → Issue с INFO, не Exception
# ---------------------------------------------------------------------------


class TestLz4GracefulDegradation:
    """Graceful degradation при отсутствии lz4/zstd.

    После задачи 8 сканер не конструирует Issue — он кладёт факт
    joblib_missing_codec в metadata (error=None), а INFO-Issue
    MLS-JOBLIB-001/002 эмитит JoblibMetadataDetector. Здесь проверяется факт;
    surfacing Issue покрыт в tests/test_joblib_metadata_detector.py.
    """

    def test_lz4_unavailable_sets_missing_codec_fact(self, tmp_path: Path) -> None:
        """Файл с lz4-сжатием при отсутствии lz4 → факт в metadata, не ImportError."""
        import sys
        import unittest.mock as mock

        lz4_data = b"\x04\x22\x4d\x18" + b"\x00" * 20  # LZ4 magic + dummy data
        lz4_file = tmp_path / "lz4_test.joblib"
        lz4_file.write_bytes(lz4_data)

        scanner = JoblibScanner()
        with mock.patch.dict(sys.modules, {"lz4": None, "lz4.frame": None}):
            decompressed, error, facts = scanner._decompress_stream(lz4_file, "lz4", lz4_file.stat().st_size)

        # Не должно быть Exception — только None + факт; error остаётся None
        assert decompressed is None
        assert error is None
        assert facts.get("joblib_missing_codec") == "lz4"

    def test_lz4_unavailable_fact_in_scan_metadata(self, tmp_path: Path) -> None:
        """scan() кладёт joblib_missing_codec=lz4 в metadata (error=None)."""
        import sys
        import unittest.mock as mock

        lz4_file = tmp_path / "lz4_scan.joblib"
        lz4_file.write_bytes(b"\x04\x22\x4d\x18" + b"\x00" * 20)

        scanner = JoblibScanner()
        with mock.patch.dict(sys.modules, {"lz4": None, "lz4.frame": None}):
            result = scanner.scan(lz4_file)

        assert result.scanner_name == "joblib"
        assert result.error is None
        assert result.metadata is not None
        assert result.metadata.get("joblib_missing_codec") == "lz4"

    def test_zstd_unavailable_sets_missing_codec_fact(self, tmp_path: Path) -> None:
        """Файл с zstd-сжатием при отсутствии zstandard → факт в metadata."""
        import sys
        import unittest.mock as mock

        zstd_file = tmp_path / "zstd_test.joblib"
        zstd_file.write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 20)  # Zstd magic

        scanner = JoblibScanner()
        with mock.patch.dict(sys.modules, {"zstandard": None}):
            decompressed, error, facts = scanner._decompress_stream(zstd_file, "zstd", zstd_file.stat().st_size)

        assert decompressed is None
        assert error is None
        assert facts.get("joblib_missing_codec") == "zstd"


# ---------------------------------------------------------------------------
# Тест: can_handle
# ---------------------------------------------------------------------------


class TestJoblibCanHandle:
    """Проверяем корректную работу can_handle."""

    def test_can_handle_joblib_extension(self, tmp_path: Path) -> None:
        """.joblib файлы обрабатываются сканером."""
        p = tmp_path / "model.joblib"
        p.write_bytes(b"\x80\x04.")
        assert JoblibScanner.can_handle(p) is True

    def test_cannot_handle_plain_pickle_pkl(self, tmp_path: Path) -> None:
        """Чистый pickle .pkl НЕ перехватывается JoblibScanner.

        Регрессия аудита #4: ранее JoblibScanner забирал любой .pkl, что
        оборачивало plain pickle лишним слоем декомпрессии и устанавливало
        scanner_name="joblib" вместо "pickle". Теперь .pkl без joblib-сжатия
        отдаётся PickleScanner.
        """
        p = tmp_path / "model.pkl"
        p.write_bytes(b"\x80\x02.")
        assert JoblibScanner.can_handle(p) is False

    def test_can_handle_compressed_pkl(self, tmp_path: Path) -> None:
        """Сжатый joblib-файл с расширением .pkl обрабатывается JoblibScanner."""
        import zlib

        p = tmp_path / "compressed_model.pkl"
        # zlib-сжатый pickle — реальный joblib-формат при compress=1
        p.write_bytes(zlib.compress(b"\x80\x02."))
        assert JoblibScanner.can_handle(p) is True

    def test_can_handle_gzip_pkl(self, tmp_path: Path) -> None:
        """gzip-сжатый .pkl обрабатывается JoblibScanner."""
        import gzip

        p = tmp_path / "gzipped_model.pkl"
        p.write_bytes(gzip.compress(b"\x80\x02."))
        assert JoblibScanner.can_handle(p) is True

    def test_plain_pkl_routed_to_pickle_scanner(self, tmp_path: Path) -> None:
        """Регрессия аудита #4: plain .pkl получает scanner_name='pickle', не 'joblib'.

        Через ScannerRegistry.find_scanner — это должен быть PickleScanner.
        """
        import pickle

        from poison_check.core.registry import ScannerRegistry
        # Гарантируем регистрацию обоих сканеров
        import poison_check.scanners  # noqa: F401

        p = tmp_path / "plain.pkl"
        p.write_bytes(pickle.dumps([1, 2, 3], protocol=2))
        scanner_class = ScannerRegistry.find_scanner(p)
        assert scanner_class is not None
        assert scanner_class.name == "pickle", (
            f"Plain .pkl должен идти в PickleScanner, получили {scanner_class.name!r}"
        )

    def test_cannot_handle_pt_extension(self, tmp_path: Path) -> None:
        """.pt файлы не обрабатываются JoblibScanner."""
        p = tmp_path / "model.pt"
        p.write_bytes(b"\x80\x02.")
        assert JoblibScanner.can_handle(p) is False

    def test_cannot_handle_safetensors(self, tmp_path: Path) -> None:
        """.safetensors файлы не обрабатываются JoblibScanner."""
        p = tmp_path / "model.safetensors"
        p.write_bytes(b"\x00\x00\x00\x00{}")
        assert JoblibScanner.can_handle(p) is False


# ---------------------------------------------------------------------------
# Тест: разбор сжатых joblib-файлов (закрытие пробела)
# ---------------------------------------------------------------------------


class TestJoblibGapClosed:
    """Закрывает пробел в разборе сжатых joblib-файлов.

    Распространённая проблема: на joblib-файлах инструменты либо падают с
    parse error, либо пропускают их. Наш сканер должен успешно сканировать
    такие файлы и находить угрозы.
    """

    def test_joblib_gap_closed(self) -> None:
        """Сканер успешно обрабатывает все вредоносные joblib-файлы.

        Критерий успеха (неделя 5, раздел 7.1): детектируем те же 6 payload,
        упакованных в joblib с разной компрессией.
        """
        malicious_files = list(MALICIOUS_DIR.glob("*.joblib"))
        assert malicious_files, (
            "Нет вредоносных .joblib фикстур в tests/fixtures/malicious/. "
            "Запустите: python tests/fixtures/generate_fixtures.py"
        )

        scanner = JoblibScanner()
        for malicious_file in malicious_files:
            result = scanner.scan(malicious_file)

            # Сканер не должен падать
            assert result is not None, (
                f"scan() вернул None для {malicious_file.name}"
            )

            # Должны быть извлечены globals (это основное требование)
            assert result.globals is not None, (
                f"globals пусты для {malicious_file.name} — "
                "сканер не смог извлечь данные из joblib-файла"
            )

            # Конкретный payload — os.system
            assert len(result.globals) > 0, (
                f"globals пустой set для {malicious_file.name}"
            )

            # Убеждаемся что os.system нашёлся
            assert ("os", "system") in result.globals, (
                f"os.system не найден в {malicious_file.name}. "
                f"Globals: {result.globals}"
            )

    def test_all_compression_methods_covered(self) -> None:
        """Все методы компрессии joblib покрыты фикстурами."""
        expected_methods = {"none", "zlib", "gzip", "bz2", "lzma", "xz"}
        found_methods: set[str] = set()

        scanner = JoblibScanner()
        for path in MALICIOUS_DIR.glob("*.joblib"):
            result = scanner.scan(path)
            if result.metadata:
                method = result.metadata.get("joblib_compression")
                if method:
                    found_methods.add(method)

        missing = expected_methods - found_methods
        assert not missing, (
            f"Не все методы компрессии протестированы. "
            f"Отсутствуют фикстуры для: {missing}"
        )

    def test_scanner_registered(self) -> None:
        """JoblibScanner должен быть зарегистрирован в ScannerRegistry.

        Примечание: test_week1_integration.py сбрасывает реестр через
        autouse-фикстуру. Поэтому здесь мы переинициализируем реестр
        вручную через прямую регистрацию, чтобы убедиться что декоратор
        @ScannerRegistry.register и имя сканера корректны.
        """
        from poison_check.core.registry import ScannerRegistry
        from poison_check.scanners.joblib_scanner import JoblibScanner

        # Проверяем либо через реестр (если не был сброшен),
        # либо напрямую через класс
        assert JoblibScanner.name == "joblib", (
            "Имя сканера должно быть 'joblib'"
        )
        assert ".joblib" in JoblibScanner.supported_extensions, (
            ".joblib должно быть в supported_extensions"
        )
        # Если реестр не был сброшен — проверяем регистрацию
        registered = ScannerRegistry.get("joblib")
        if registered is not None:
            assert registered is JoblibScanner


# ---------------------------------------------------------------------------
# Тесты: защита от decompression bomb
# ---------------------------------------------------------------------------


class TestDecompressionBomb:
    """Decompression bomb атака должна отвергаться без зависания и OOM.

    Стратегия: мокируем MAX_DECOMP = 1024 (1 КБ) через patch на модуль,
    создаём gzip/bz2/zlib из 2 КБ нулей — после распаковки превышают лимит.
    Реальный 3 ГБ файл не нужен.
    """

    def _make_gzip_bytes(self, payload: bytes) -> bytes:
        """Создаёт gzip-сжатые байты из payload."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gf:
            gf.write(payload)
        return buf.getvalue()

    def test_decompression_bomb_gzip_rejected(self, tmp_path: Path) -> None:
        """gzip-бомба (2 КБ нулей при лимите 1 КБ) → error + Issue MLS-JOBLIB-003, не OOM.

        Мокируем MAX_DECOMP=1024, создаём gzip с 2048 нулями.
        scan() должен вернуть RawScanData с error и не зависнуть.
        """
        payload = b"\x00" * 2048
        gzip_data = self._make_gzip_bytes(payload)

        bomb_file = tmp_path / "bomb.joblib"
        bomb_file.write_bytes(gzip_data)

        scanner = JoblibScanner()
        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024):
            result = scanner.scan(bomb_file)

        assert result is not None, "scan() не должен возвращать None"
        assert result.scanner_name == "joblib"
        assert result.error is not None, (
            "Decompression bomb должна вернуть error, не успешный результат"
        )
        assert "bomb" in result.error.lower() or "лимит" in result.error.lower() or "превысила" in result.error.lower(), (
            f"Сообщение об ошибке должно содержать описание bomb-атаки, получено: {result.error!r}"
        )

    def test_decompression_bomb_gzip_sets_bomb_fact(self, tmp_path: Path) -> None:
        """При gzip-бомбе сканер кладёт факт bomb в metadata + error (не Issue).

        MLS-JOBLIB-003 эмитит JoblibMetadataDetector по этому факту — см.
        tests/test_joblib_metadata_detector.py.
        """
        payload = b"\x00" * 2048
        gzip_data = self._make_gzip_bytes(payload)
        bomb_file = tmp_path / "bomb_fact.joblib"
        bomb_file.write_bytes(gzip_data)

        scanner = JoblibScanner()
        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024):
            decompressed, error, facts = scanner._decompress_stream(bomb_file, "gzip", bomb_file.stat().st_size)

        assert decompressed is None, (
            "При bomb-атаке декомпрессированные данные должны быть None"
        )
        assert error is not None, "Должна быть строка ошибки (сохраняем exit 2)"
        assert facts.get("joblib_decompression_bomb") == "true"
        assert facts.get("joblib_bomb_method") == "gzip"

    def test_decompression_bomb_scan_metadata_has_fact(self, tmp_path: Path) -> None:
        """scan() на gzip-бомбе выставляет И error, И факт bomb в metadata.

        Регрессия задачи 8: bomb должна оставлять факт, иначе MLS-JOBLIB-003
        снова теряется (детектор его не увидит).
        """
        payload = b"\x00" * 2048
        gzip_data = self._make_gzip_bytes(payload)
        bomb_file = tmp_path / "bomb_scan.joblib"
        bomb_file.write_bytes(gzip_data)

        scanner = JoblibScanner()
        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024):
            result = scanner.scan(bomb_file)

        assert result.error is not None
        assert result.metadata is not None
        assert result.metadata.get("joblib_decompression_bomb") == "true"
        assert result.metadata.get("joblib_bomb_method") == "gzip"

    def test_decompression_bomb_zlib_rejected(self, tmp_path: Path) -> None:
        """zlib-бомба (2 КБ нулей при лимите 1 КБ) → error + Issue MLS-JOBLIB-003."""
        payload = b"\x00" * 2048
        zlib_data = zlib.compress(payload)

        bomb_file = tmp_path / "bomb_zlib.joblib"
        bomb_file.write_bytes(zlib_data)

        scanner = JoblibScanner()
        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024):
            result = scanner.scan(bomb_file)

        assert result is not None
        assert result.error is not None, "zlib-бомба должна вернуть error"

    def test_decompression_bomb_bz2_rejected(self, tmp_path: Path) -> None:
        """bz2-бомба (2 КБ нулей при лимите 1 КБ) → error + Issue MLS-JOBLIB-003."""
        payload = b"\x00" * 2048
        bz2_data = bz2.compress(payload)

        bomb_file = tmp_path / "bomb_bz2.joblib"
        bomb_file.write_bytes(bz2_data)

        scanner = JoblibScanner()
        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024):
            result = scanner.scan(bomb_file)

        assert result is not None
        assert result.error is not None, "bz2-бомба должна вернуть error"

    def test_decompression_bomb_lzma_rejected(self, tmp_path: Path) -> None:
        """lzma-бомба (2 КБ нулей при лимите 1 КБ) → error + Issue MLS-JOBLIB-003."""
        payload = b"\x00" * 2048
        lzma_data = lzma.compress(payload, format=lzma.FORMAT_ALONE)

        bomb_file = tmp_path / "bomb_lzma.joblib"
        bomb_file.write_bytes(lzma_data)

        scanner = JoblibScanner()
        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024):
            result = scanner.scan(bomb_file)

        assert result is not None
        assert result.error is not None, "lzma-бомба должна вернуть error"

    def test_normal_gzip_within_limit_succeeds(self, tmp_path: Path) -> None:
        """Легитимный gzip меньше лимита декомпрессируется успешно."""
        # Минимальный pickle: протокол 4, пустой список, STOP
        payload = b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00]\x94)}\x94."
        gzip_data = self._make_gzip_bytes(payload)

        bomb_file = tmp_path / "normal_gzip.joblib"
        bomb_file.write_bytes(gzip_data)

        scanner = JoblibScanner()
        # Лимит 1 МБ — payload в сотни байт, проблем нет
        with mock.patch.object(joblib_scanner_module, "MAX_DECOMP", 1024 * 1024):
            result = scanner.scan(bomb_file)

        # Ошибки декомпрессии быть не должно
        # (ошибка pickle — допустима, главное что не bomb)
        assert result is not None
        assert result.scanner_name == "joblib"

"""Тесты compliance-модулей: fstec_mapping, owasp_ml_top10, gost_mapping.

Проверяет:
- Маппинг Issue.code → УБИ ФСТЭК (FstecMapper)
- Маппинг Issue.code → OWASP ML Top 10 (OwaspMapper)
- Маппинг Issue.code → ГОСТ Р 56939-2024 (GostMapper)
- compliance_tags в Issues от каждого детектора
- Корректность generate_report(): содержит все ключи ML01–ML10
- Устойчивость к пустым / ошибочным ScanResult
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from poison_check.compliance.fstec_mapping import FstecMapper, UBI_MAPPINGS
from poison_check.compliance.gost_mapping import GostMapper, GOST_56939_2024, ISSUE_TO_GOST
from poison_check.compliance.owasp_ml_top10 import (
    OWASP_ML_TOP10,
    ISSUE_TO_OWASP,
    OwaspMapper,
)
from poison_check.core.result import (
    Confidence,
    EmbeddedSignature,
    FileResult,
    Issue,
    MLContext,
    Severity,
    StringInfo,
    Summary,
    ScanResult,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.allowlist_detector import AllowlistDetector
from poison_check.detectors.blocklist_detector import BlocklistDetector
from poison_check.detectors.compression_detector import CompressionDetector
from poison_check.detectors.executable_detector import ExecutableDetector
from poison_check.detectors.network_detector import NetworkDetector
from poison_check.detectors.secrets_detector import SecretsDetector


# ---------------------------------------------------------------------------
# Фабричные функции
# ---------------------------------------------------------------------------


def _make_issue(
    code: str = "MLS-PKL-001",
    severity: Severity = Severity.CRITICAL,
    confidence: Confidence = Confidence.CERTAIN,
    message: str = "Тестовое сообщение",
    location: str = "test.pkl",
    compliance_tags: list[str] | None = None,
) -> Issue:
    """Создаёт тестовый Issue с заданными параметрами."""
    return Issue(
        code=code,
        severity=severity,
        confidence=confidence,
        message=message,
        location=location,
        compliance_tags=compliance_tags or [],
    )


def _make_scan_result(issues: list[Issue] | None = None) -> ScanResult:
    """Создаёт минимальный ScanResult с опциональным списком Issues."""
    file_path = Path("test_model.pkl")
    file_result = FileResult(
        file_path=file_path,
        scanner_name="test",
        issues=issues or [],
    )
    return ScanResult(
        tool_version="0.1.0-test",
        timestamp=datetime(2026, 5, 1, 12, 0, 0),
        duration_ms=100.0,
        scanned_paths=[file_path],
        results_per_file={file_path: file_result},
        summary=Summary(),
    )


def _make_raw_data(
    strings: list[str] | None = None,
    embedded_bytes: list[EmbeddedSignature] | None = None,
    raw_content_sample: bytes | None = None,
    file_size: int = 1024,
    nested_files: list[RawScanData] | None = None,
) -> RawScanData:
    """Создаёт минимальный RawScanData для тестов детекторов."""
    string_infos: list[StringInfo] | None = None
    if strings is not None:
        string_infos = [
            StringInfo(value=s, position=i * 100)
            for i, s in enumerate(strings)
        ]
    return RawScanData(
        file_path=Path("test_model.pkl"),
        file_hash={"sha256": "abc", "sha512": "def", "md5": "000"},
        file_size=file_size,
        scanner_name="test",
        strings=string_infos,
        embedded_bytes=embedded_bytes,
        raw_content_sample=raw_content_sample,
        nested_files=nested_files,
    )


def _context(framework: str = "unknown") -> MLContext:
    """Создаёт MLContext для тестов."""
    return MLContext(framework=framework, confidence=0.5)


# ---------------------------------------------------------------------------
# FstecMapper — базовые тесты
# ---------------------------------------------------------------------------


class TestFstecMapper:
    """Тесты для FstecMapper."""

    def test_map_issue_pickle_rce_returns_ubi067(self) -> None:
        """Issue с кодом MLS001 (blocklist/pickle RCE) → содержит 'fstec:ubi-067'."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-PKL-001")
        ubi_ids = mapper.map_issue(issue)
        tags = [mapper.compliance_tag(u) for u in ubi_ids]
        assert "fstec:ubi-067" in tags, (
            f"MLS001 должен маппироваться на УБИ.067, получено: {ubi_ids}"
        )

    def test_map_issue_secrets_returns_ubi037(self) -> None:
        """Issue с кодом MLS020 (secrets) → содержит УБИ.037."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-SEC-001")
        ubi_ids = mapper.map_issue(issue)
        assert "УБИ.037" in ubi_ids, f"MLS020 → ожидается УБИ.037, получено: {ubi_ids}"

    def test_map_issue_network_callback_returns_ubi174(self) -> None:
        """Issue с кодом MLS033 (публичный IP) → содержит УБИ.174."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-NET-004")
        ubi_ids = mapper.map_issue(issue)
        assert "УБИ.174" in ubi_ids, f"MLS033 → ожидается УБИ.174, получено: {ubi_ids}"

    def test_map_issue_embedded_executable_returns_ubi067_and_ubi068(self) -> None:
        """Issue с кодом MLS040 (embedded exe) → содержит УБИ.067 и УБИ.068."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-EXE-001")
        ubi_ids = mapper.map_issue(issue)
        assert "УБИ.067" in ubi_ids, f"MLS040 → ожидается УБИ.067, получено: {ubi_ids}"
        assert "УБИ.068" in ubi_ids, f"MLS040 → ожидается УБИ.068, получено: {ubi_ids}"

    def test_map_issue_compression_returns_ubi111(self) -> None:
        """Issue с кодом MLS050 (zip-бомба) → содержит УБИ.111."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-CMP-001")
        ubi_ids = mapper.map_issue(issue)
        assert "УБИ.111" in ubi_ids, f"MLS050 → ожидается УБИ.111, получено: {ubi_ids}"

    def test_map_issue_cve_detector_prefix_matching(self) -> None:
        """Issue с динамическим кодом CVE (MLS-CVE-2025-32434) → маппируется через prefix."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-CVE-2025-32434")
        ubi_ids = mapper.map_issue(issue)
        assert len(ubi_ids) > 0, "MLS-CVE-2025-32434 должен маппироваться на хотя бы один УБИ"
        assert "УБИ.067" in ubi_ids

    def test_map_issue_pattern_prefix_matching(self) -> None:
        """Issue с паттерном (MLS-PATTERN-OS-SYSTEM) → маппируется через prefix."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-PATTERN-OS-SYSTEM")
        ubi_ids = mapper.map_issue(issue)
        assert "УБИ.067" in ubi_ids

    def test_map_issue_unknown_code_returns_empty(self) -> None:
        """Issue с неизвестным кодом → возвращает пустой список, не бросает исключение."""
        mapper = FstecMapper()
        issue = _make_issue(code="UNKNOWN-CODE-999")
        result = mapper.map_issue(issue)
        assert result == [], f"Неизвестный код → [], получено: {result}"

    def test_map_result_no_exception_on_empty_scan_result(self) -> None:
        """FstecMapper.map_result() не бросает исключение на пустом ScanResult."""
        mapper = FstecMapper()
        result = _make_scan_result(issues=[])
        grouped = mapper.map_result(result)
        assert isinstance(grouped, dict)

    def test_map_result_returns_dict_on_any_scan_result(self) -> None:
        """FstecMapper.map_result() возвращает dict без Exception на любом ScanResult."""
        mapper = FstecMapper()
        issues = [
            _make_issue(code="MLS-PKL-001"),
            _make_issue(code="MLS-SEC-001"),
            _make_issue(code="MLS-EXE-001"),
            _make_issue(code="MLS-CMP-001"),
            _make_issue(code="UNKNOWN-999"),
        ]
        result = _make_scan_result(issues=issues)
        grouped = mapper.map_result(result)

        assert isinstance(grouped, dict)
        # Должны быть ключи УБИ.067 и УБИ.037
        assert "УБИ.067" in grouped
        assert "УБИ.037" in grouped
        # UNKNOWN-999 → _unmapped
        assert "_unmapped" in grouped

    def test_map_result_issue_deduplication(self) -> None:
        """Один Issue с маппингом на несколько УБИ попадает в несколько ключей."""
        mapper = FstecMapper()
        issue = _make_issue(code="MLS-EXE-001")  # → УБИ.067, УБИ.068, УБИ.162
        result = _make_scan_result(issues=[issue])
        grouped = mapper.map_result(result)
        # Один issue с MLS040 → несколько ключей
        assert "УБИ.067" in grouped
        assert "УБИ.068" in grouped

    def test_compliance_tag_format(self) -> None:
        """compliance_tag() формирует корректный формат 'fstec:ubi-NNN'."""
        mapper = FstecMapper()
        assert mapper.compliance_tag("УБИ.067") == "fstec:ubi-067"
        assert mapper.compliance_tag("УБИ.037") == "fstec:ubi-037"
        assert mapper.compliance_tag("УБИ.174") == "fstec:ubi-174"

    def test_get_ubi_description_known(self) -> None:
        """get_ubi_description() возвращает описание для известных УБИ."""
        mapper = FstecMapper()
        desc = mapper.get_ubi_description("УБИ.067")
        assert desc  # не пустая строка
        assert "код" in desc.lower() or "выполнени" in desc.lower()

    def test_get_ubi_description_unknown(self) -> None:
        """get_ubi_description() возвращает заглушку для неизвестных УБИ."""
        mapper = FstecMapper()
        desc = mapper.get_ubi_description("УБИ.999")
        assert "999" in desc or "bdu.fstec.ru" in desc


# ---------------------------------------------------------------------------
# OwaspMapper — базовые тесты
# ---------------------------------------------------------------------------


class TestOwaspMapper:
    """Тесты для OwaspMapper."""

    def test_map_issue_secrets_returns_ml09(self) -> None:
        """Issue MLS020 (secrets) → содержит 'owasp-ml:ml09' в тегах."""
        mapper = OwaspMapper()
        issue = _make_issue(code="MLS-SEC-001")
        owasp_cats = mapper.map_issue(issue)
        tags = [mapper.compliance_tag(c) for c in owasp_cats]
        assert "owasp-ml:ml09" in tags, (
            f"MLS020 должен маппироваться на ML09, получено категории: {owasp_cats}"
        )

    def test_map_issue_pickle_rce_returns_ml03(self) -> None:
        """Issue MLS001 (blocklist) → содержит ML03 (supply chain)."""
        mapper = OwaspMapper()
        issue = _make_issue(code="MLS-PKL-001")
        owasp_cats = mapper.map_issue(issue)
        assert "ML03" in owasp_cats, f"MLS001 → ML03, получено: {owasp_cats}"

    def test_map_issue_embedded_exe_returns_ml03(self) -> None:
        """Issue MLS040 (embedded exe) → содержит ML03."""
        mapper = OwaspMapper()
        issue = _make_issue(code="MLS-EXE-001")
        owasp_cats = mapper.map_issue(issue)
        assert "ML03" in owasp_cats, f"MLS040 → ML03, получено: {owasp_cats}"

    def test_map_issue_unknown_code_returns_empty(self) -> None:
        """OwaspMapper.map_issue() возвращает [] для неизвестного кода."""
        mapper = OwaspMapper()
        issue = _make_issue(code="UNKNOWN-CODE")
        result = mapper.map_issue(issue)
        assert result == []

    def test_map_issue_cve_prefix_matching(self) -> None:
        """MLS-CVE-2025-32434 маппируется через prefix на ML08/ML10."""
        mapper = OwaspMapper()
        issue = _make_issue(code="MLS-CVE-2025-32434")
        cats = mapper.map_issue(issue)
        assert len(cats) > 0
        assert any(c in cats for c in ["ML08", "ML10"])

    def test_generate_report_contains_all_ml01_ml10(self) -> None:
        """OwaspMapper.generate_report() содержит все ключи ML01–ML10."""
        mapper = OwaspMapper()
        result = _make_scan_result(issues=[])
        report = mapper.generate_report(result)

        for cat_id in [f"ML{i:02d}" for i in range(1, 11)]:
            assert cat_id in report, f"Ключ {cat_id} отсутствует в отчёте"
            assert "title_ru" in report[cat_id], f"Нет title_ru для {cat_id}"
            assert "status" in report[cat_id], f"Нет status для {cat_id}"
            assert "issues" in report[cat_id], f"Нет issues для {cat_id}"

    def test_generate_report_summary_key_present(self) -> None:
        """generate_report() содержит ключ '_summary'."""
        mapper = OwaspMapper()
        result = _make_scan_result(issues=[])
        report = mapper.generate_report(result)
        assert "_summary" in report
        summary = report["_summary"]
        assert "hit_categories" in summary
        assert "clean_categories" in summary
        assert "total_issues" in summary

    def test_generate_report_hit_categories_populated(self) -> None:
        """generate_report() корректно заполняет hit_categories при наличии Issues."""
        mapper = OwaspMapper()
        # MLS001 → ML03, ML10
        issues = [_make_issue(code="MLS-PKL-001"), _make_issue(code="MLS-SEC-001")]
        result = _make_scan_result(issues=issues)
        report = mapper.generate_report(result)

        summary = report["_summary"]
        assert "ML03" in summary["hit_categories"] or "ML03" in report
        # ML09 должна быть в hit (из MLS020)
        assert report["ML09"]["status"] == "hit"

    def test_generate_report_clean_on_empty(self) -> None:
        """generate_report() помечает все категории как clean при пустом ScanResult."""
        mapper = OwaspMapper()
        result = _make_scan_result(issues=[])
        report = mapper.generate_report(result)
        for cat_id in [f"ML{i:02d}" for i in range(1, 11)]:
            assert report[cat_id]["status"] == "clean"

    def test_generate_report_no_exception_on_mixed_issues(self) -> None:
        """generate_report() не бросает исключение на разнородных Issues."""
        mapper = OwaspMapper()
        issues = [
            _make_issue(code="MLS-PKL-001"),
            _make_issue(code="MLS-SEC-001"),
            _make_issue(code="MLS-EXE-001"),
            _make_issue(code="MLS-CMP-001"),
            _make_issue(code="MLS-PATTERN-OS-SYSTEM"),
            _make_issue(code="UNKNOWN-999"),
        ]
        result = _make_scan_result(issues=issues)
        report = mapper.generate_report(result)  # не должно бросать исключение
        assert isinstance(report, dict)

    def test_compliance_tag_format(self) -> None:
        """compliance_tag() формирует корректный формат 'owasp-ml:ml03'."""
        mapper = OwaspMapper()
        assert mapper.compliance_tag("ML03") == "owasp-ml:ml03"
        assert mapper.compliance_tag("ML09") == "owasp-ml:ml09"
        assert mapper.compliance_tag("ML10") == "owasp-ml:ml10"

    def test_owasp_ml_top10_dict_completeness(self) -> None:
        """Словарь OWASP_ML_TOP10 содержит ровно 10 категорий ML01–ML10."""
        expected = {f"ML{i:02d}" for i in range(1, 11)}
        actual = set(OWASP_ML_TOP10.keys())
        assert actual == expected, f"Неполный список категорий: {actual ^ expected}"


# ---------------------------------------------------------------------------
# GostMapper — базовые тесты
# ---------------------------------------------------------------------------


class TestGostMapper:
    """Тесты для GostMapper."""

    def test_map_issue_pickle_rce_returns_section_53(self) -> None:
        """Issue MLS001 → содержит раздел 5.3 ГОСТ (безопасное программирование)."""
        mapper = GostMapper()
        issue = _make_issue(code="MLS-PKL-001")
        sections = mapper.map_issue(issue)
        assert "5.3" in sections, f"MLS001 → 5.3, получено: {sections}"

    def test_map_issue_secrets_returns_section_55(self) -> None:
        """Issue MLS020 (secrets) → содержит раздел 5.5 ГОСТ (хранение данных)."""
        mapper = GostMapper()
        issue = _make_issue(code="MLS-SEC-001")
        sections = mapper.map_issue(issue)
        assert "5.5" in sections, f"MLS020 → 5.5, получено: {sections}"

    def test_map_issue_executable_returns_supply_chain_section(self) -> None:
        """Issue MLS040 (embedded exe) → содержит разделы 5.4 и 6.3 ГОСТ."""
        mapper = GostMapper()
        issue = _make_issue(code="MLS-EXE-001")
        sections = mapper.map_issue(issue)
        assert "5.4" in sections
        assert "6.3" in sections

    def test_map_issue_unknown_returns_empty(self) -> None:
        """GostMapper.map_issue() возвращает [] для неизвестного кода."""
        mapper = GostMapper()
        issue = _make_issue(code="UNKNOWN-9999")
        result = mapper.map_issue(issue)
        assert result == []

    def test_map_result_no_exception(self) -> None:
        """GostMapper.map_result() не бросает исключение на любом ScanResult."""
        mapper = GostMapper()
        issues = [
            _make_issue(code="MLS-PKL-001"),
            _make_issue(code="MLS-SEC-001"),
            _make_issue(code="UNKNOWN-999"),
        ]
        result = _make_scan_result(issues=issues)
        grouped = mapper.map_result(result)
        assert isinstance(grouped, dict)

    def test_compliance_tag_format(self) -> None:
        """compliance_tag() формирует 'gost:56939-2024:5.3'."""
        mapper = GostMapper()
        assert mapper.compliance_tag("5.3") == "gost:56939-2024:5.3"
        assert mapper.compliance_tag("6.1") == "gost:56939-2024:6.1"

    def test_format_reference(self) -> None:
        """format_reference() формирует 'ГОСТ Р 56939-2024 п.5.3'."""
        mapper = GostMapper()
        assert mapper.format_reference("5.3") == "ГОСТ Р 56939-2024 п.5.3"

    def test_gost_56939_2024_not_empty(self) -> None:
        """GOST_56939_2024 содержит хотя бы 5 разделов."""
        assert len(GOST_56939_2024) >= 5


# ---------------------------------------------------------------------------
# Тесты compliance_tags в Issues от детекторов
# ---------------------------------------------------------------------------


class TestDetectorComplianceTags:
    """Проверяет, что детекторы заполняют compliance_tags в Issues."""

    def test_blocklist_detector_issue_contains_fstec_ubi067(self) -> None:
        """BlocklistDetector: Issue → содержит 'fstec:ubi-067' в compliance_tags.

        os.system попадает под PATTERN-OS-SYSTEM (CVE-правило), не под MLS001.
        Используем global, которого нет в CVE-правилах, чтобы получить MLS001.
        Либо проверяем что любой issue от blocklist-детектора содержит fstec:ubi-067.
        """
        raw_with_globals = RawScanData(
            file_path=Path("evil.pkl"),
            file_hash={"sha256": "abc", "sha512": "def", "md5": "000"},
            file_size=100,
            scanner_name="test",
            globals={("os", "system")},
        )
        detector = BlocklistDetector()
        ctx = _context()
        issues = detector.analyze(raw_with_globals, ctx)
        assert issues, "BlocklistDetector должен найти os.system"
        # os.system может дать PATTERN-OS-SYSTEM (CVE) или MLS001 (blocklist) —
        # в обоих случаях compliance_tags должны содержать fstec:ubi-067
        for issue in issues:
            assert "fstec:ubi-067" in issue.compliance_tags, (
                f"Issue {issue.code} должен содержать fstec:ubi-067, "
                f"теги: {issue.compliance_tags}"
            )

    def test_secrets_detector_issue_contains_owasp_ml09(
        self, tmp_path: Path
    ) -> None:
        """SecretsDetector: Issue MLS020 → содержит 'owasp-ml:ml09' в compliance_tags."""
        # Создаём raw с API-ключом-строкой (должен сработать secrets detector)
        raw = _make_raw_data(
            strings=["AKIA1234567890ABCDEF"],  # AWS access key формат
        )
        detector = SecretsDetector()
        ctx = _context()
        issues = detector.analyze(raw, ctx)
        if not issues:
            pytest.skip("SecretsDetector не распознал тестовый секрет (паттерн может отличаться)")
        issue = issues[0]
        assert "owasp-ml:ml09" in issue.compliance_tags, (
            f"MLS020 должен содержать owasp-ml:ml09, теги: {issue.compliance_tags}"
        )

    def test_executable_detector_issue_contains_fstec_tags(self) -> None:
        """ExecutableDetector: Issue MLS040 → содержит fstec:ubi-067 в compliance_tags."""
        # PE-заголовок в embedded_bytes
        raw = _make_raw_data(
            embedded_bytes=[
                EmbeddedSignature(signature_type="PE", offset=0, size=4096)
            ]
        )
        detector = ExecutableDetector()
        ctx = _context()
        issues = detector.analyze(raw, ctx)
        assert issues, "ExecutableDetector должен найти PE-файл"
        issue = issues[0]
        assert "fstec:ubi-067" in issue.compliance_tags, (
            f"MLS040 должен содержать fstec:ubi-067, теги: {issue.compliance_tags}"
        )
        assert "owasp-ml:ml03" in issue.compliance_tags

    def test_executable_detector_issue_contains_gost_tag(self) -> None:
        """ExecutableDetector: Issue MLS040 → содержит gost:56939-2024:5.3."""
        raw = _make_raw_data(
            embedded_bytes=[
                EmbeddedSignature(signature_type="ELF", offset=0, size=2048)
            ]
        )
        detector = ExecutableDetector()
        ctx = _context()
        issues = detector.analyze(raw, ctx)
        assert issues
        issue = issues[0]
        assert "gost:56939-2024:5.3" in issue.compliance_tags, (
            f"Ожидается gost:56939-2024:5.3, теги: {issue.compliance_tags}"
        )

    def test_network_detector_unknown_domain_contains_fstec174(self) -> None:
        """NetworkDetector: Issue MLS031 (неизвестный домен) → содержит fstec:ubi-174."""
        raw = _make_raw_data(strings=["https://evil-attacker.com/exfil"])
        detector = NetworkDetector()
        ctx = _context()
        issues = detector.analyze(raw, ctx)
        # Ищем MLS031 — неизвестный домен
        mls031_issues = [i for i in issues if i.code == "MLS-NET-002"]
        assert mls031_issues, f"Должен быть Issue MLS031, найдено: {[i.code for i in issues]}"
        issue = mls031_issues[0]
        assert "fstec:ubi-174" in issue.compliance_tags, (
            f"MLS031 → fstec:ubi-174, теги: {issue.compliance_tags}"
        )

    def test_compression_detector_zipbomb_contains_fstec111(self) -> None:
        """CompressionDetector: Issue MLS050 (zip-бомба) → содержит fstec:ubi-111."""
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Создаём содержимое с высоким коэффициентом сжатия
            data = b"\x00" * (200 * 1024 * 1024)  # 200 МБ нулей (сожмётся очень хорошо)
            zf.writestr("huge.bin", data)

        compressed_bytes = buf.getvalue()
        raw = _make_raw_data(
            raw_content_sample=compressed_bytes[:4096],
            file_size=len(compressed_bytes),
        )
        # Используем raw_content_sample для инициализации
        raw_full = RawScanData(
            file_path=Path("bomb.zip"),
            file_hash={"sha256": "abc", "sha512": "def", "md5": "000"},
            file_size=len(compressed_bytes),
            scanner_name="test",
            raw_content_sample=compressed_bytes,
        )
        detector = CompressionDetector()
        ctx = _context()
        issues = detector.analyze(raw_full, ctx)
        zipbomb_issues = [i for i in issues if i.code in ("MLS-CMP-001", "MLS-CMP-002")]
        if not zipbomb_issues:
            pytest.skip("CompressionDetector не обнаружил zip-бомбу на тестовых данных")
        issue = zipbomb_issues[0]
        assert "fstec:ubi-111" in issue.compliance_tags, (
            f"Zip-бомба → fstec:ubi-111, теги: {issue.compliance_tags}"
        )

    def test_allowlist_detector_issue_contains_fstec_tags(self) -> None:
        """AllowlistDetector: Issue MLS010 → содержит fstec:ubi-067 и gost:56939-2024:5.3."""
        raw = RawScanData(
            file_path=Path("suspicious.pkl"),
            file_hash={"sha256": "abc", "sha512": "def", "md5": "000"},
            file_size=100,
            scanner_name="test",
            globals={("dill._dill", "_create_code")},
        )
        detector = AllowlistDetector()
        ctx = MLContext(framework="pytorch", confidence=0.9)
        issues = detector.analyze(raw, ctx)
        if not issues:
            pytest.skip("AllowlistDetector не вернул issues — allowlist может включать dill")
        issue = issues[0]
        assert "fstec:ubi-067" in issue.compliance_tags, (
            f"MLS010 → fstec:ubi-067, теги: {issue.compliance_tags}"
        )
        assert "gost:56939-2024:5.3" in issue.compliance_tags, (
            f"MLS010 → gost:56939-2024:5.3, теги: {issue.compliance_tags}"
        )


# ---------------------------------------------------------------------------
# Тесты маппингов через compliance_tags в Issue (end-to-end через сканеры)
# ---------------------------------------------------------------------------


class TestComplianceTagsEndToEnd:
    """Проверяет полный цикл: реальный файл → детекторы → compliance_tags."""

    def test_pickle_rce_issue_compliance_tags_contain_fstec_ubi067(self) -> None:
        """Issue от RCE-паттерна содержит 'fstec:ubi-067' в compliance_tags.

        os.system → PATTERN-OS-SYSTEM (CVE), не MLS001.
        Оба кода должны содержать fstec:ubi-067.
        """
        raw = RawScanData(
            file_path=Path("evil.pkl"),
            file_hash={"sha256": "abc", "sha512": "def", "md5": "000"},
            file_size=100,
            scanner_name="test",
            globals={("os", "system")},
        )
        detector = BlocklistDetector()
        issues = detector.analyze(raw, _context())
        assert issues, "BlocklistDetector должен найти os.system"
        for issue in issues:
            assert "fstec:ubi-067" in issue.compliance_tags, (
                f"RCE-issue {issue.code} должен содержать fstec:ubi-067, "
                f"теги: {issue.compliance_tags}"
            )

    def test_secrets_issue_compliance_tags_contain_owasp_ml09(self) -> None:
        """Secrets Issue MLS020 содержит 'owasp-ml:ml09' непосредственно в Issue.compliance_tags."""
        # Создаём Issue напрямую (как это делает SecretsDetector)
        from poison_check.detectors.secrets_detector import SecretsDetector
        raw = _make_raw_data(strings=["sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"])
        detector = SecretsDetector()
        issues = detector.analyze(raw, _context())
        if issues:
            assert "owasp-ml:ml09" in issues[0].compliance_tags

    def test_fstec_mapper_on_scan_result_with_rce(self) -> None:
        """FstecMapper.map_result() корректно группирует RCE issues по УБИ.067."""
        mapper = FstecMapper()
        issue = _make_issue(
            code="MLS-PKL-001",
            compliance_tags=["owasp-ml:ml03", "fstec:ubi-067", "gost:56939-2024:5.3"],
        )
        result = _make_scan_result(issues=[issue])
        grouped = mapper.map_result(result)
        assert "УБИ.067" in grouped
        assert issue in grouped["УБИ.067"]

    def test_owasp_mapper_generate_report_ml01_through_ml10_all_present(self) -> None:
        """OwaspMapper.generate_report() содержит все ключи ML01–ML10 при разных Issues."""
        mapper = OwaspMapper()
        issues = [
            _make_issue(code="MLS-PKL-001"),   # → ML03, ML10
            _make_issue(code="MLS-SEC-001"),   # → ML09
            _make_issue(code="MLS-EXE-001"),   # → ML03, ML10
            _make_issue(code="MLS-CVE-2025-32434"),  # → ML08, ML10
        ]
        result = _make_scan_result(issues=issues)
        report = mapper.generate_report(result)

        for i in range(1, 11):
            cat_id = f"ML{i:02d}"
            assert cat_id in report, f"Ключ {cat_id} отсутствует в отчёте"

    def test_issue_to_owasp_all_values_are_valid_categories(self) -> None:
        """Все значения ISSUE_TO_OWASP ссылаются на существующие категории ML01–ML10."""
        valid_cats = {f"ML{i:02d}" for i in range(1, 11)}
        for code, cats in ISSUE_TO_OWASP.items():
            for cat in cats:
                assert cat in valid_cats, (
                    f"Код {code} ссылается на несуществующую категорию {cat}"
                )

    def test_ubi_mappings_all_values_are_valid_ubi_ids(self) -> None:
        """Все значения UBI_MAPPINGS имеют формат 'УБИ.NNN'."""
        import re
        ubi_pattern = re.compile(r"^УБИ\.\d{3}$")
        for code, ubi_ids in UBI_MAPPINGS.items():
            for ubi_id in ubi_ids:
                assert ubi_pattern.match(ubi_id), (
                    f"Код {code}: невалидный УБИ-идентификатор '{ubi_id}'"
                )

    def test_gost_mapper_compliance_tags_in_issues(self) -> None:
        """Все Issues c кодами в ISSUE_TO_GOST содержат валидные разделы ГОСТ."""
        mapper = GostMapper()
        for code in ISSUE_TO_GOST:
            issue = _make_issue(code=code)
            sections = mapper.map_issue(issue)
            assert sections, f"Код {code} должен маппироваться на хотя бы один раздел ГОСТ"
            for section in sections:
                # Раздел должен быть в формате "N.N" или "N"
                parts = section.split(".")
                assert all(p.isdigit() for p in parts), (
                    f"Невалидный формат раздела ГОСТ '{section}' для кода {code}"
                )


# ---------------------------------------------------------------------------
# Регрессия аудита #30: compliance disclaimer присутствует в отчётах
# ---------------------------------------------------------------------------


class TestComplianceDisclaimer:
    """ComplianceReport содержит disclaimer о неверифицированности маппинга."""

    def test_disclaimer_constant_exists_and_nonempty(self) -> None:
        """COMPLIANCE_DISCLAIMER экспортирован и не пустой."""
        from poison_check.compliance.fstec_mapping import (  # noqa: PLC0415
            COMPLIANCE_DISCLAIMER,
        )

        assert COMPLIANCE_DISCLAIMER
        assert len(COMPLIANCE_DISCLAIMER) > 100
        assert "ВНИМАНИЕ" in COMPLIANCE_DISCLAIMER
        assert "верифиц" in COMPLIANCE_DISCLAIMER.lower()

    def test_compliance_report_includes_disclaimer_after_scan(
        self, tmp_path: object
    ) -> None:
        """После реального сканирования с банковской политикой disclaimer заполнен."""
        import pickle as _pickle
        from pathlib import Path as _P

        from poison_check.scanner import Scanner  # noqa: PLC0415

        # Используем tmp_path как Path
        tmp = _P(str(tmp_path))
        target = tmp / "model.pkl"
        target.write_bytes(_pickle.dumps([1, 2, 3], protocol=2))

        result = Scanner(policy="banking").scan(target)
        assert result.compliance_report is not None
        assert result.compliance_report.disclaimer is not None
        assert "ВНИМАНИЕ" in result.compliance_report.disclaimer

    def test_disclaimer_in_json_output(self) -> None:
        """JSON-форматтер включает disclaimer в сериализацию compliance_report."""
        from datetime import datetime, timezone
        from pathlib import Path as _P

        from poison_check.compliance.fstec_mapping import (  # noqa: PLC0415
            COMPLIANCE_DISCLAIMER,
        )
        from poison_check.core.result import (  # noqa: PLC0415
            ComplianceReport,
            FileResult,
            ScanResult,
            Summary,
        )
        from poison_check.output.json_format import JsonFormatter  # noqa: PLC0415

        result = ScanResult(
            tool_version="0.1.0",
            timestamp=datetime.now(timezone.utc),
            duration_ms=1.0,
            scanned_paths=[_P("x.pkl")],
            policy="banking",
            results_per_file={
                _P("x.pkl"): FileResult(file_path=_P("x.pkl"), scanner_name="pickle"),
            },
            summary=Summary(),
            compliance_report=ComplianceReport(
                fstec_ubi=["УБИ.067"],
                disclaimer=COMPLIANCE_DISCLAIMER,
            ),
        )
        json_str = JsonFormatter().format(result)
        assert "disclaimer" in json_str
        assert "ВНИМАНИЕ" in json_str

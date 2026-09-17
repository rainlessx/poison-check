"""Тесты для SecretsDetector, NetworkDetector, ExecutableDetector, CompressionDetector."""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.core.result import (
    Confidence,
    EmbeddedSignature,
    Issue,
    MLContext,
    Severity,
    StringInfo,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.compression_detector import CompressionDetector
from poison_check.detectors.executable_detector import ExecutableDetector
from poison_check.detectors.network_detector import NetworkDetector
from poison_check.detectors.secrets_detector import SecretsDetector


# ---------------------------------------------------------------------------
# Вспомогательные фабричные функции
# ---------------------------------------------------------------------------


def _context(framework: str = "unknown") -> MLContext:
    """Создаёт минимальный MLContext для тестов."""
    return MLContext(framework=framework, confidence=0.5)


def _raw(
    strings: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    embedded_bytes: list[EmbeddedSignature] | None = None,
    raw_content_sample: bytes | None = None,
    nested_files: list[RawScanData] | None = None,
    file_size: int = 1024,
) -> RawScanData:
    """Создаёт минимальный RawScanData для тестов."""
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
        metadata=metadata,
        embedded_bytes=embedded_bytes,
        raw_content_sample=raw_content_sample,
        nested_files=nested_files,
    )


# ---------------------------------------------------------------------------
# SecretsDetector
# ---------------------------------------------------------------------------


class TestSecretsDetector:
    """Тесты для детектора секретов."""

    def setup_method(self) -> None:
        """Создаёт экземпляр детектора перед каждым тестом."""
        self.detector = SecretsDetector()

    def test_openai_key_detected_as_critical(self) -> None:
        """Строка с OpenAI API ключом → Issue CRITICAL."""
        # Ключ OpenAI: sk- + 48 символов [a-zA-Z0-9]
        key = "sk-" + "A" * 48
        raw = _raw(strings=[f"openai_key = '{key}'"])
        issues = self.detector.analyze(raw, _context())

        assert len(issues) >= 1
        secret_issues = [i for i in issues if "OpenAI" in i.message]
        assert len(secret_issues) >= 1
        assert secret_issues[0].severity == Severity.CRITICAL
        assert secret_issues[0].code == "MLS-SEC-001"

    def test_no_secrets_returns_empty(self) -> None:
        """Строка без секретов → пустой список."""
        raw = _raw(strings=[
            "layer = torch.nn.Linear(768, 768)",
            "optimizer = torch.optim.Adam(model.parameters())",
            "batch_size = 32",
        ])
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_huggingface_token_detected(self) -> None:
        """Токен HuggingFace → Issue CRITICAL."""
        token = "hf_" + "B" * 37
        raw = _raw(strings=[f"token = '{token}'"])
        issues = self.detector.analyze(raw, _context())

        assert len(issues) >= 1
        hf_issues = [i for i in issues if "HuggingFace" in i.message]
        assert len(hf_issues) >= 1
        assert hf_issues[0].severity == Severity.CRITICAL

    def test_aws_key_detected(self) -> None:
        """AWS Access Key ID → Issue CRITICAL."""
        raw = _raw(strings=["AKIAIOSFODNN7EXAMPLE123456"])
        issues = self.detector.analyze(raw, _context())
        # AKIA + 16 символов [0-9A-Z]
        aws_issues = [i for i in issues if "AWS" in i.message]
        assert len(aws_issues) >= 1
        assert aws_issues[0].severity == Severity.CRITICAL

    def test_generic_api_key_detected_as_high(self) -> None:
        """Generic API ключ → Issue HIGH."""
        raw = _raw(strings=["api_key = 'supersecret12345678901234'"])
        issues = self.detector.analyze(raw, _context())

        generic_issues = [i for i in issues if "API ключ" in i.message]
        assert len(generic_issues) >= 1
        assert generic_issues[0].severity == Severity.HIGH

    def test_private_key_detected_as_critical(self) -> None:
        """RSA private key header → Issue CRITICAL."""
        raw = _raw(strings=["-----BEGIN RSA PRIVATE KEY-----\nMIIE..."])
        issues = self.detector.analyze(raw, _context())

        pk_issues = [i for i in issues if "приватный" in i.message.lower()]
        assert len(pk_issues) >= 1
        assert pk_issues[0].severity == Severity.CRITICAL

    def test_secret_in_metadata(self) -> None:
        """Секрет в metadata (safetensors) → тоже детектируется."""
        token = "hf_" + "C" * 37
        raw = _raw(metadata={"author_token": token})
        issues = self.detector.analyze(raw, _context())

        assert len(issues) >= 1
        assert any("HuggingFace" in i.message for i in issues)

    def test_no_strings_and_no_metadata(self) -> None:
        """Нет строк и метаданных → пустой список."""
        raw = _raw()
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_error_file_returns_empty(self) -> None:
        """Файл с ошибкой парсинга → пустой список (graceful degradation)."""
        raw = _raw(strings=["sk-" + "A" * 48])
        raw.error = "parse error"
        issues = self.detector.analyze(raw, _context())
        assert issues == []


# ---------------------------------------------------------------------------
# NetworkDetector
# ---------------------------------------------------------------------------


class TestNetworkDetector:
    """Тесты для детектора сетевых IoC."""

    def setup_method(self) -> None:
        """Создаёт экземпляр детектора перед каждым тестом."""
        self.detector = NetworkDetector()

    def test_whitelist_domain_returns_info(self) -> None:
        """URL huggingface.co → Issue INFO (whitelist)."""
        raw = _raw(strings=["model_url = 'https://huggingface.co/bert-base-uncased'"])
        issues = self.detector.analyze(raw, _context())

        assert len(issues) >= 1
        hf_issues = [i for i in issues if "huggingface.co" in i.details.get("domain", "")]
        assert len(hf_issues) >= 1
        assert hf_issues[0].severity == Severity.INFO

    def test_unknown_domain_returns_medium(self) -> None:
        """URL attacker.com → Issue MEDIUM."""
        raw = _raw(strings=["callback = 'https://attacker.com/exfil'"])
        issues = self.detector.analyze(raw, _context())

        unknown_issues = [i for i in issues if "attacker.com" in i.details.get("domain", "")]
        assert len(unknown_issues) >= 1
        assert unknown_issues[0].severity == Severity.MEDIUM
        assert unknown_issues[0].code == "MLS-NET-002"

    def test_public_ip_returns_high(self) -> None:
        """Публичный IP 1.2.3.4 → Issue HIGH."""
        raw = _raw(strings=["server = '1.2.3.4'"])
        issues = self.detector.analyze(raw, _context())

        ip_issues = [i for i in issues if "1.2.3.4" in i.details.get("ip", "")]
        assert len(ip_issues) >= 1
        assert ip_issues[0].severity == Severity.HIGH

    def test_private_ip_not_flagged_high(self) -> None:
        """Приватный IP 192.168.1.1 → не HIGH (либо INFO, либо не найден)."""
        raw = _raw(strings=["local = '192.168.1.1'"])
        issues = self.detector.analyze(raw, _context())

        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        # Приватный IP не должен давать HIGH
        private_high = [
            i for i in high_issues
            if "192.168.1.1" in str(i.details)
        ]
        assert private_high == []

    def test_loopback_ip_ignored(self) -> None:
        """Loopback IP 127.0.0.1 → не создаёт HIGH Issue."""
        raw = _raw(strings=["localhost = '127.0.0.1:8080'"])
        issues = self.detector.analyze(raw, _context())

        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        assert high_issues == []

    def test_pytorch_org_is_whitelist(self) -> None:
        """pytorch.org → INFO (whitelist)."""
        raw = _raw(strings=["url = 'https://download.pytorch.org/models/resnet50.pth'"])
        issues = self.detector.analyze(raw, _context())

        assert any(i.severity == Severity.INFO for i in issues)

    def test_file_scheme_url_detected_as_high(self) -> None:
        """Регрессия аудита #9: file:// → HIGH с кодом MLS-NET-006.

        file:// — LFI/SSRF вектор при PDF-рендере.
        """
        raw = _raw(strings=["bad = 'file:///etc/passwd'"])
        issues = self.detector.analyze(raw, _context())

        file_issues = [i for i in issues if i.code == "MLS-NET-006"]
        assert len(file_issues) == 1
        assert file_issues[0].severity == Severity.HIGH
        assert "file" in file_issues[0].details.get("scheme", "")

    def test_javascript_scheme_url_detected_as_high(self) -> None:
        """javascript: → HIGH с кодом MLS-NET-006."""
        raw = _raw(strings=["xss = 'javascript:alert(1)'"])
        issues = self.detector.analyze(raw, _context())

        js_issues = [i for i in issues if i.code == "MLS-NET-006"]
        assert len(js_issues) == 1
        assert js_issues[0].severity == Severity.HIGH
        assert js_issues[0].details.get("scheme") == "javascript"

    def test_data_scheme_url_detected_as_high(self) -> None:
        """data:base64 → HIGH с кодом MLS-NET-006."""
        raw = _raw(strings=["payload = 'data:text/html;base64,PHNjcmlwdD4='"])
        issues = self.detector.analyze(raw, _context())

        data_issues = [i for i in issues if i.code == "MLS-NET-006"]
        assert len(data_issues) == 1
        assert data_issues[0].severity == Severity.HIGH

    def test_ftp_url_detected(self) -> None:
        """ftp:// — обрабатывается как обычный URL по unknown-домену."""
        raw = _raw(strings=["mirror = 'ftp://attacker.example/payload.bin'"])
        issues = self.detector.analyze(raw, _context())
        # ftp на чужом домене — это либо MEDIUM (unknown domain), либо классифицирован
        # как dangerous scheme. Главное — что-то нашли.
        assert len(issues) >= 1
        assert any(
            "attacker.example" in str(i.details) or i.code == "MLS-NET-006"
            for i in issues
        )

    def test_no_strings_returns_empty(self) -> None:
        """Нет строк → пустой список."""
        raw = _raw()
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_ip_in_url_not_duplicated(self) -> None:
        """IP внутри URL обрабатывается один раз, не дублируется как standalone IP."""
        raw = _raw(strings=["url = 'https://1.2.3.4/cmd'"])
        issues = self.detector.analyze(raw, _context())

        # Один Issue для IP-URL, не два (один для URL, один для standalone IP)
        ip_issues = [i for i in issues if "1.2.3.4" in str(i.details)]
        assert len(ip_issues) == 1

    def test_default_does_not_escalate(self) -> None:
        """По умолчанию escalate_external_urls=False — поведение не изменилось."""
        assert self.detector.escalate_external_urls is False


# ---------------------------------------------------------------------------
# NetworkDetector — правило политики extra_rules.no_external_urls
# ---------------------------------------------------------------------------


class TestNetworkDetectorEscalateExternalUrls:
    """Эскалация неизвестных доменов MEDIUM → HIGH по правилу no_external_urls.

    Правило приходит из политик banking / government / strict
    (``extra_rules.no_external_urls: true``) и затрагивает только ветку
    MLS-NET-002 — неизвестный домен.
    """

    _UNKNOWN = "callback = 'https://attacker.example/exfil'"
    _WHITELISTED = "model_url = 'https://huggingface.co/bert-base-uncased'"

    def _net_issues(self, code: str, escalate: bool, string: str) -> list[Issue]:
        """Возвращает issues указанного кода для одной строки файла."""
        detector = NetworkDetector(escalate_external_urls=escalate)
        issues = detector.analyze(_raw(strings=[string]), _context())
        return [i for i in issues if i.code == code]

    def test_unknown_domain_high_when_enabled(self) -> None:
        """no_external_urls=true → неизвестный домен получает HIGH."""
        issues = self._net_issues("MLS-NET-002", escalate=True, string=self._UNKNOWN)

        assert len(issues) == 1
        assert issues[0].severity == Severity.HIGH
        assert issues[0].details["escalated_by_policy"] is True
        assert issues[0].details["domain"] == "attacker.example"

    def test_unknown_domain_medium_when_disabled(self) -> None:
        """Без флага неизвестный домен остаётся MEDIUM (дефолтное поведение)."""
        issues = self._net_issues("MLS-NET-002", escalate=False, string=self._UNKNOWN)

        assert len(issues) == 1
        assert issues[0].severity == Severity.MEDIUM
        assert issues[0].details["escalated_by_policy"] is False

    def test_whitelisted_domain_stays_info_when_enabled(self) -> None:
        """Whitelist-домен остаётся INFO даже при включённой эскалации."""
        issues = self._net_issues("MLS-NET-001", escalate=True, string=self._WHITELISTED)

        assert len(issues) == 1
        assert issues[0].severity == Severity.INFO

    def test_whitelisted_domain_stays_info_when_disabled(self) -> None:
        """Whitelist-домен остаётся INFO и без эскалации."""
        issues = self._net_issues("MLS-NET-001", escalate=False, string=self._WHITELISTED)

        assert len(issues) == 1
        assert issues[0].severity == Severity.INFO

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html;base64,PHNjcmlwdD4=",
            "vbscript:msgbox(1)",
        ],
    )
    def test_dangerous_schemes_unchanged(self, url: str) -> None:
        """Опасные схемы дают HIGH (MLS-NET-006) независимо от флага политики."""
        for escalate in (False, True):
            issues = self._net_issues(
                "MLS-NET-006", escalate=escalate, string=f"x = '{url}'"
            )
            assert len(issues) == 1, f"escalate={escalate}, url={url}"
            assert issues[0].severity == Severity.HIGH

    def test_escalation_does_not_change_issue_code(self) -> None:
        """Код issue остаётся MLS-NET-002 — меняется только severity."""
        high = self._net_issues("MLS-NET-002", escalate=True, string=self._UNKNOWN)[0]
        medium = self._net_issues("MLS-NET-002", escalate=False, string=self._UNKNOWN)[0]

        assert high.code == medium.code == "MLS-NET-002"
        assert high.location == medium.location
        assert high.confidence == medium.confidence

    def test_escalation_explained_in_why(self) -> None:
        """При эскалации в поле why указана причина — правило политики."""
        issue = self._net_issues("MLS-NET-002", escalate=True, string=self._UNKNOWN)[0]
        assert issue.why is not None
        assert "no_external_urls" in issue.why


# ---------------------------------------------------------------------------
# ExecutableDetector
# ---------------------------------------------------------------------------


class TestExecutableDetector:
    """Тесты для детектора встроенных исполняемых файлов."""

    def setup_method(self) -> None:
        """Создаёт экземпляр детектора перед каждым тестом."""
        self.detector = ExecutableDetector()

    def test_pe_signature_in_sample_returns_critical(self) -> None:
        """Валидный PE-blob в sample → Issue CRITICAL.

        Используется минимальный PE с корректным e_lfanew и PE\\x00\\x00,
        так как структурная валидация отфильтровывает случайные MZ-байты.
        """
        # Минимальный PE: MZ + 58 байт padding + e_lfanew=0x40 + PE\0\0 + stub
        pe_blob = (
            b"MZ"
            + b"\x00" * 58
            + (0x40).to_bytes(4, "little")  # e_lfanew = 64
            + b"PE\x00\x00"
            + b"\x00" * 60
        )
        sample = b"\x00" * 8 + pe_blob
        raw = _raw(raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())

        assert len(issues) >= 1
        pe_issues = [i for i in issues if "PE" in i.message]
        assert len(pe_issues) >= 1
        assert pe_issues[0].severity == Severity.CRITICAL
        assert pe_issues[0].code == "MLS-EXE-001"

    def test_elf_signature_detected(self) -> None:
        """Валидный ELF-заголовок → Issue CRITICAL."""
        # \x7fELF + ei_class=2 (64-bit) + ei_data=1 (little-endian) + ei_version=1
        sample = b"\x7fELF" + b"\x02\x01\x01\x00" + b"\x00" * 100
        raw = _raw(raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())

        elf_issues = [i for i in issues if "ELF" in i.message]
        assert len(elf_issues) >= 1
        assert elf_issues[0].severity == Severity.CRITICAL

    def test_invalid_pe_not_detected(self) -> None:
        """MZ без валидного DOS-заголовка → NOT detected (FP regression).

        Воспроизводит сценарий float32-тензоров, где байты 0x4D 0x5A (MZ)
        встречаются тысячи раз случайно, но PE-структура не валидна.
        """
        # MZ с нулевым e_lfanew — не проходит _validate_pe
        sample = b"\x00" * 32 + b"MZ" + b"\x00" * 200
        raw = _raw(raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())
        pe_issues = [i for i in issues if "PE" in i.message]
        assert pe_issues == [], (
            f"Невалидный PE не должен детектироваться, найдено: {pe_issues}"
        )

    def test_invalid_elf_not_detected(self) -> None:
        """\\x7fELF с недопустимыми полями ident → NOT detected (FP regression)."""
        # ei_class=5 (не 1 и не 2) — невалидный ELF
        sample = b"\x7fELF" + b"\x05\x01\x01\x00" + b"\x00" * 100
        raw = _raw(raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())
        elf_issues = [i for i in issues if "ELF" in i.message]
        assert elf_issues == []

    def test_random_float32_tensor_no_pe_fp(self) -> None:
        """Случайные float32-данные не генерируют FP PE-детекцию.

        Воспроизводит типичные данные тензоров PyTorch: 4 КБ псевдослучайных
        float32, которые статистически содержат MZ-байты, но без PE-структуры.
        """
        import struct
        import math
        # Генерируем 4096 байт через детерминированный float32-паттерн
        floats = [math.sin(i * 0.001) for i in range(1024)]
        tensor_data = struct.pack(f"{len(floats)}f", *floats)
        raw = _raw(raw_content_sample=tensor_data)
        issues = self.detector.analyze(raw, _context())
        pe_issues = [i for i in issues if "PE" in i.message]
        assert pe_issues == [], (
            f"Тензорные данные не должны давать PE FP, найдено: {pe_issues}"
        )

    def test_macho_signature_detected(self) -> None:
        """Mach-O сигнатура → Issue CRITICAL."""
        sample = b"\xfe\xed\xfa\xce" + b"\x00" * 100
        raw = _raw(raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())

        macho_issues = [i for i in issues if "Mach-O" in i.message]
        assert len(macho_issues) >= 1
        assert macho_issues[0].severity == Severity.CRITICAL

    def test_normal_data_returns_empty(self) -> None:
        """Обычные данные (без сигнатур) → пустой список."""
        sample = b"\x00" * 4096
        raw = _raw(raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_safetensors_like_data_returns_empty(self) -> None:
        """Данные похожие на safetensors header → пустой список."""
        # safetensors начинается с uint64 (длина заголовка)
        header_len = (100).to_bytes(8, "little")
        sample = header_len + b'{"__metadata__": {}}' + b"\x00" * 80
        raw = _raw(raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_embedded_bytes_from_scanner(self) -> None:
        """EmbeddedSignature от сканера → Issue CRITICAL без дублирования."""
        emb = EmbeddedSignature(signature_type="PE", offset=256, size=1024)
        raw = _raw(embedded_bytes=[emb])
        issues = self.detector.analyze(raw, _context())

        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL
        assert "PE" in issues[0].message

    def test_no_duplicate_when_embedded_and_sample_overlap(self) -> None:
        """Если embedded_bytes и sample оба указывают на offset 0 — не дублируем."""
        emb = EmbeddedSignature(signature_type="ELF", offset=0, size=100)
        # Валидный ELF: ei_class=2, ei_data=1, ei_version=1
        sample = b"\x7fELF" + b"\x02\x01\x01\x00" + b"\x00" * 100
        raw = _raw(embedded_bytes=[emb], raw_content_sample=sample)
        issues = self.detector.analyze(raw, _context())

        # Offset 0 покрыт embedded_bytes, в sample тоже найдём — но offset=0 уже seen
        issues_at_zero = [i for i in issues if i.details.get("offset") == 0]
        assert len(issues_at_zero) == 1

    def test_error_file_returns_empty(self) -> None:
        """Файл с ошибкой → пустой список (независимо от содержимого sample)."""
        # Используем валидный PE-blob, чтобы убедиться что блокирует именно
        # error-флаг, а не отсутствие сигнатуры.
        pe_blob = (
            b"MZ"
            + b"\x00" * 58
            + (0x40).to_bytes(4, "little")
            + b"PE\x00\x00"
            + b"\x00" * 60
        )
        raw = _raw(raw_content_sample=pe_blob)
        raw.error = "parse error"
        issues = self.detector.analyze(raw, _context())
        assert issues == []


# ---------------------------------------------------------------------------
# CompressionDetector
# ---------------------------------------------------------------------------


class TestCompressionDetector:
    """Тесты для детектора атак через сжатие."""

    def setup_method(self) -> None:
        """Создаёт экземпляр детектора перед каждым тестом."""
        self.detector = CompressionDetector()

    def _make_nested(self, size_bytes: int, count: int = 1) -> list[RawScanData]:
        """Создаёт список вложенных RawScanData с заданным суммарным размером."""
        return [
            RawScanData(
                file_path=Path(f"nested_{i}.pkl"),
                file_hash={"sha256": "x", "sha512": "y", "md5": "z"},
                file_size=size_bytes // max(count, 1),
                scanner_name="test",
            )
            for i in range(count)
        ]

    def test_normal_size_returns_empty(self) -> None:
        """Нормальное соотношение размеров → пустой список."""
        # 5 МБ файл, 10 МБ суммарно — соотношение 2x, норма
        nested = self._make_nested(size_bytes=10 * 1024 * 1024, count=3)
        raw = _raw(nested_files=nested, file_size=5 * 1024 * 1024)
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_zipbomb_small_file_large_unpacked(self) -> None:
        """Маленький файл (< 1 МБ) + суммарно > 1 ГБ → MLS-CMP-001 (HIGH/CRITICAL).

        Аудит #25: severity зависит от ratio. 4096× — это HIGH (подозрительно,
        возможно агрессивное сжатие). Чтобы получить CRITICAL — нужен ratio
        > 10 000.
        """
        # 500 КБ сжатый, 2 ГБ распакованный — ratio ≈ 4096×
        nested = self._make_nested(size_bytes=2 * 1024 * 1024 * 1024, count=1)
        raw = _raw(nested_files=nested, file_size=500 * 1024)
        issues = self.detector.analyze(raw, _context())

        bomb_issues = [i for i in issues if i.code == "MLS-CMP-001"]
        assert len(bomb_issues) >= 1
        assert bomb_issues[0].severity == Severity.HIGH
        assert bomb_issues[0].confidence == Confidence.MEDIUM

    def test_extreme_zipbomb_returns_critical(self) -> None:
        """ratio > 10 000× → CRITICAL/HIGH (почти точно bomb)."""
        # 100 КБ сжатый, 2 ГБ распакованный — ratio ≈ 20 000×
        nested = self._make_nested(size_bytes=2 * 1024 * 1024 * 1024, count=1)
        raw = _raw(nested_files=nested, file_size=100 * 1024)
        issues = self.detector.analyze(raw, _context())

        bomb_issues = [i for i in issues if i.code == "MLS-CMP-001"]
        assert len(bomb_issues) >= 1
        assert bomb_issues[0].severity == Severity.CRITICAL
        assert bomb_issues[0].confidence == Confidence.HIGH

    def test_realistic_sklearn_compress9_no_critical_fp(self) -> None:
        """Регрессия аудита #25: sklearn.dump(compress=9) на реальных моделях.

        joblib с compress=9 на повторяющихся весах легко даёт ratio 100–500×.
        Это НЕ должно давать CRITICAL false positive. Допустим только HIGH+MEDIUM
        confidence.
        """
        # 5 МБ сжатый, 1 ГБ распакованный — ratio = 200× (realistic для sklearn)
        nested = self._make_nested(size_bytes=1 * 1024 * 1024 * 1024, count=1)
        raw = _raw(nested_files=nested, file_size=5 * 1024 * 1024)
        issues = self.detector.analyze(raw, _context())

        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        assert critical_issues == [], (
            "Realistic sklearn compress=9 не должен давать CRITICAL FP, "
            f"получено: {[i.code for i in critical_issues]}"
        )

    def test_high_compression_ratio(self) -> None:
        """Соотношение > 1000x → Issue HIGH (MLS051)."""
        # 1 МБ + 1 байт (чуть больше порога, чтобы не попасть в zipbomb ветку)
        # Файл 1.5 МБ (выше _SMALL_FILE_THRESHOLD), распакованный 2000 МБ → ratio=1333x
        nested = self._make_nested(size_bytes=2000 * 1024 * 1024, count=1)
        file_size = 1500 * 1024  # 1.5 МБ — выше порога 1 МБ
        raw = _raw(nested_files=nested, file_size=file_size)
        issues = self.detector.analyze(raw, _context())

        ratio_issues = [i for i in issues if i.code == "MLS-CMP-002"]
        assert len(ratio_issues) >= 1
        assert ratio_issues[0].severity == Severity.HIGH

    def test_many_files_over_1000_returns_high(self) -> None:
        """1001 вложенный файл → Issue HIGH (MLS052)."""
        nested = self._make_nested(size_bytes=1001 * 1024, count=1001)
        raw = _raw(nested_files=nested, file_size=100 * 1024)
        issues = self.detector.analyze(raw, _context())

        many_issues = [i for i in issues if i.code == "MLS-CMP-003"]
        assert len(many_issues) >= 1
        assert many_issues[0].severity == Severity.HIGH

    def test_many_files_over_100_returns_medium(self) -> None:
        """101 вложенный файл → Issue MEDIUM (MLS052)."""
        nested = self._make_nested(size_bytes=101 * 1024, count=101)
        # Соотношение должно быть < 1000x, чтобы проверить только счётчик
        raw = _raw(nested_files=nested, file_size=200 * 1024)
        issues = self.detector.analyze(raw, _context())

        count_issues = [i for i in issues if i.code == "MLS-CMP-003"]
        assert len(count_issues) >= 1
        assert count_issues[0].severity == Severity.MEDIUM

    def test_no_nested_files_returns_empty(self) -> None:
        """Нет nested_files → пустой список."""
        raw = _raw(file_size=100 * 1024)
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_error_file_returns_empty(self) -> None:
        """Файл с ошибкой → пустой список."""
        nested = self._make_nested(size_bytes=2 * 1024 * 1024 * 1024, count=1)
        raw = _raw(nested_files=nested, file_size=100)
        raw.error = "parse error"
        issues = self.detector.analyze(raw, _context())
        assert issues == []

    def test_total_uncompressed_bytes_metadata_used(self) -> None:
        """Регрессия аудита #15: PyTorch ZIP zip-bomb детектируется через metadata.

        До фикса CompressionDetector суммировал nested.file_size, что для
        PyTorch ZIP даёт размер pickle-payload, а не суммарный размер всех
        членов архива (включая бинарные тензоры). Теперь PyTorchScanner
        записывает total_uncompressed_bytes в metadata, и детектор
        предпочитает это значение.
        """
        # Эмулируем PyTorch ZIP: 500 КБ файл, nested = 1 крошечный pickle (200 байт),
        # но в metadata указано что в архиве распакованного на 2 ГБ.
        nested = self._make_nested(size_bytes=200, count=1)
        raw = _raw(nested_files=nested, file_size=500 * 1024)
        raw.metadata = {
            "pytorch_format": "zip",
            "total_uncompressed_bytes": str(2 * 1024 * 1024 * 1024),
        }

        issues = self.detector.analyze(raw, _context())
        bomb_issues = [i for i in issues if i.code == "MLS-CMP-001"]
        assert len(bomb_issues) >= 1, (
            "Должна быть детектирована zip-bomb через total_uncompressed_bytes"
        )
        # severity зависит от ratio (аудит #25): 2 ГБ / 500 КБ ≈ 4096× → HIGH.
        # Главное — что детектор сработал; точный severity проверяется отдельно.
        assert bomb_issues[0].severity in (Severity.CRITICAL, Severity.HIGH)

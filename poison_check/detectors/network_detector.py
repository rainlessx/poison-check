"""Детектор сетевых индикаторов компрометации (IoC) в ML-файлах.

Ищет URL и IP-адреса в строках файла.
Whitelist-домены → INFO (информационно).
Неизвестные домены → MEDIUM (подозрительно), либо HIGH при включённом
правиле политики ``extra_rules.no_external_urls`` (banking/government/strict).
Публичные IP-адреса (не RFC-1918, не loopback) → HIGH.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from typing import ClassVar
from urllib.parse import ParseResult, urlparse

from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Severity,
)
from poison_check.core.scanner_base import RawScanData

logger = logging.getLogger(__name__)

# Регулярное выражение для поиска URL.
# Покрывает (аудит #9):
#   * http(s)://
#   * ftp://, ftps://
#   * file://    — может тянуть локальные ресурсы при PDF-рендере / parse
#   * git+https://, git+ssh://, git+http://  — supply-chain атаки в HF metadata
#   * javascript:, data:, vbscript:  — вредоносные scheme в HTML/PDF контексте
# Скобки/кавычки/управляющие символы в теле URL отвергаются (типичный stop-set).
_URL_PATTERN: re.Pattern[str] = re.compile(
    r"(?:https?|ftps?|file|vbscript|javascript|data|git\+(?:https?|ssh))"
    r":(?://)?[^\s'\"<>\[\]{}|\\^`\x00-\x1f\x7f]+",
    re.IGNORECASE,
)

# Регулярное выражение для IP-адресов (с необязательным портом)
_IP_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
    r"(?::\d{1,5})?\b"
)


@DetectorRegistry.register
class NetworkDetector(BaseDetector):
    """Детектор сетевых индикаторов компрометации (Network IoC).

    Проверяет raw_data.strings на наличие URL и IP-адресов.

    Классификация по severity:
    - Whitelist-домены (huggingface.co, pytorch.org, etc.) → INFO
    - Неизвестные домены → MEDIUM (или HIGH при escalate_external_urls=True)
    - Публичные IP-адреса (не 127.x, не RFC-1918, не link-local) → HIGH

    IP-адреса в URL не дублируются: обрабатываются один раз в рамках URL.
    """

    name: ClassVar[str] = "network"
    description: ClassVar[str] = "Детектор сетевых IoC: URL и IP-адреса в ML-файлах"
    severity_range: ClassVar[tuple[Severity, Severity]] = (Severity.INFO, Severity.HIGH)

    def __init__(self, escalate_external_urls: bool = False) -> None:
        """Создаёт детектор сетевых IoC.

        :param escalate_external_urls: Если True — URL к домену вне whitelist
            получает severity HIGH вместо MEDIUM (MLS-NET-002). Значение
            приходит из ``extra_rules.no_external_urls`` политики banking /
            government / strict: для этих отраслей любой внешний адрес внутри
            модели считается признаком C2-канала или утечки данных.
            Whitelist-домены (INFO) и опасные схемы file:/javascript:/data:/
            vbscript: (HIGH) флаг не затрагивает.
        """
        self.escalate_external_urls = escalate_external_urls

    # Домены, которые легитимны для ML-экосистемы
    WHITELIST_DOMAINS: ClassVar[frozenset[str]] = frozenset({
        "huggingface.co",
        "pytorch.org",
        "github.com",
        "githubusercontent.com",
        "pypi.org",
        "python.org",
        "tensorflow.org",
        "keras.io",
        "scikit-learn.org",
        "numpy.org",
        "scipy.org",
        "conda.io",
        "anaconda.com",
        "gitlab.com",
        "arxiv.org",
        "paperswithcode.com",
        "ultralytics.com",
    })

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Ищет URL и IP-адреса в строках файла.

        Алгоритм:
        1. Собирает строки из raw_data.strings.
        2. Ищет URL — classифицирует по домену (whitelist vs. неизвестный;
           неизвестный эскалируется до HIGH при ``escalate_external_urls``).
        3. Ищет standalone IP (не входящие в уже найденные URL) — проверяет публичность.
        4. Возвращает список Issues с соответствующим severity.

        Дедупликация: одно значение (URL или IP) → один Issue.
        """
        if raw_data.error is not None:
            return []

        if not raw_data.strings:
            return []

        issues: list[Issue] = []
        seen_urls: set[str] = set()
        seen_ips: set[str] = set()
        # IP-адреса, уже вошедшие в состав URL — не создаём повторный Issue
        ips_in_urls: set[str] = set()

        for string_info in raw_data.strings:
            value = string_info.value
            location = f"{raw_data.file_path} (offset {string_info.position})"

            # --- URL ---
            for url_match in _URL_PATTERN.finditer(value):
                url = url_match.group(0).rstrip(".,;)")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Опасные scheme — всегда HIGH, без учёта whitelist (аудит #9).
                # file://, javascript:, data:, vbscript: — это либо чтение
                # локальных ресурсов (SSRF/LFI вектор при PDF-рендере), либо
                # выполнение скрипта в HTML-контексте.
                lower_url = url.lower()
                if lower_url.startswith(("file:", "javascript:", "data:", "vbscript:")):
                    # data:image/... — preview-картинка в metadata LoRA/safetensors,
                    # не несёт угрозы. Прочие data:-схемы остаются HIGH.
                    if lower_url.startswith("data:image/"):
                        issues.append(_make_data_image_issue(url, location))
                    else:
                        issues.append(_make_dangerous_scheme_issue(url, location))
                    continue

                parsed = _safe_parse_url(url)
                if parsed is None:
                    continue

                hostname = parsed.hostname or ""

                # Если hostname — IP-адрес, отмечаем его как обработанный
                if _is_ip_address(hostname):
                    ips_in_urls.add(hostname)
                    issue = _classify_ip_url(raw_data, url, hostname, location)
                else:
                    issue = _classify_url(
                        raw_data,
                        url,
                        hostname,
                        location,
                        self.WHITELIST_DOMAINS,
                        escalate_external=self.escalate_external_urls,
                    )

                if issue is not None:
                    issues.append(issue)

            # --- Standalone IP-адреса ---
            for ip_match in _IP_PATTERN.finditer(value):
                raw_ip = ip_match.group(0)
                # Отделяем порт от IP для проверки
                ip_str = raw_ip.split(":")[0]
                if ip_str in seen_ips or ip_str in ips_in_urls:
                    continue
                seen_ips.add(ip_str)

                issue = _classify_standalone_ip(raw_data, raw_ip, ip_str, location)
                if issue is not None:
                    issues.append(issue)

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _safe_parse_url(url: str) -> ParseResult | None:
    """Разбирает URL через urllib.parse. Возвращает None при ошибке."""
    try:
        return urlparse(url)
    except ValueError:
        return None


def _is_ip_address(hostname: str) -> bool:
    """Возвращает True если hostname — валидный IPv4-адрес."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _is_private_or_loopback(ip_str: str) -> bool:
    """Возвращает True если IP является приватным, loopback или link-local.

    RFC-1918 приватные диапазоны:
    - 10.0.0.0/8
    - 172.16.0.0/12
    - 192.168.0.0/16

    Также: loopback 127.0.0.0/8, link-local 169.254.0.0/16.
    """
    try:
        addr = ipaddress.ip_address(ip_str)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return True  # При неразборчивом адресе — не поднимаем тревогу


def _is_whitelisted_domain(hostname: str, whitelist: frozenset[str]) -> bool:
    """Проверяет, является ли hostname или его родительский домен whitelisted."""
    hostname = hostname.lower().rstrip(".")
    if hostname in whitelist:
        return True
    # Проверяем поддомены: cdn.huggingface.co → huggingface.co
    parts = hostname.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in whitelist:
            return True
    return False


def _classify_url(
    raw_data: RawScanData,
    url: str,
    hostname: str,
    location: str,
    whitelist: frozenset[str],
    escalate_external: bool = False,
) -> Issue | None:
    """Создаёт Issue для URL в зависимости от домена.

    :param escalate_external: Поднимает severity неизвестного домена
        MEDIUM → HIGH (правило политики ``no_external_urls``). На
        whitelist-домены не влияет.
    """
    if _is_whitelisted_domain(hostname, whitelist):
        return Issue(
            code="MLS-NET-001",
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            message=f"URL к доверенному сервису: {hostname}",
            location=location,
            details={"url": url, "domain": hostname, "whitelisted": True},
            why=(
                "Обнаружен URL к известному ML-сервису. "
                "Сам по себе не является угрозой, но стоит убедиться, "
                "что сетевые запросы при загрузке модели ожидаемы."
            ),
            remediation=(
                "Проверьте, выполняет ли модель сетевые запросы при загрузке — "
                "это может быть нежелательным поведением."
            ),
            compliance_tags=[
                "owasp-ml:ml03",
                "fstec:ubi-179",
                "gost:56939-2024:5.4",
            ],
        )

    # Неизвестный домен — подозрительно. Политика с no_external_urls
    # (banking / government / strict) поднимает уровень до HIGH.
    why_text = (
        "Неизвестный URL в ML-файле может указывать на передачу данных "
        "злоумышленнику (data exfiltration) или загрузку вредоносного кода."
    )
    if escalate_external:
        why_text += (
            " Политика сканирования запрещает внешние URL "
            "(extra_rules.no_external_urls), поэтому уровень повышен до HIGH."
        )

    return Issue(
        code="MLS-NET-002",
        severity=Severity.HIGH if escalate_external else Severity.MEDIUM,
        confidence=Confidence.MEDIUM,
        message=f"Обнаружен URL к неизвестному домену: {hostname}",
        location=location,
        details={
            "url": url,
            "domain": hostname,
            "whitelisted": False,
            "escalated_by_policy": escalate_external,
        },
        why=why_text,
        remediation=(
            "Проверьте, легитимен ли этот домен для данной модели. "
            "Если модель не должна делать сетевые запросы — это признак атаки."
        ),
        compliance_tags=[
            "owasp-ml:ml10",
            "owasp-ml:ml03",
            "fstec:ubi-174",
            "fstec:ubi-179",
            "gost:56939-2024:5.5",
            "gost:56939-2024:6.1",
        ],
    )


def _classify_ip_url(
    raw_data: RawScanData,
    url: str,
    ip_str: str,
    location: str,
) -> Issue | None:
    """Создаёт Issue для URL с IP-адресом вместо домена."""
    if _is_private_or_loopback(ip_str):
        return Issue(
            code="MLS-NET-003",
            severity=Severity.INFO,
            confidence=Confidence.MEDIUM,
            message=f"URL к локальному/приватному IP: {ip_str}",
            location=location,
            details={"url": url, "ip": ip_str, "is_private": True},
            why="URL к приватному IP обычно означает обращение к локальному сервису.",
            remediation=(
                "Проверьте, ожидается ли это обращение. "
                "В production-среде это может указывать на SSRF-атаку."
            ),
            compliance_tags=[
                "owasp-ml:ml10",
                "fstec:ubi-174",
                "gost:56939-2024:5.5",
            ],
        )

    return Issue(
        code="MLS-NET-004",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=f"URL к публичному IP-адресу: {ip_str}",
        location=location,
        details={"url": url, "ip": ip_str, "is_private": False},
        why=(
            "URL к публичному IP-адресу (не имя хоста) является нестандартным "
            "и может указывать на command-and-control инфраструктуру атакующего "
            "или утечку данных."
        ),
        remediation=(
            "Блокируйте исходящий трафик к этому IP. "
            "Проверьте, не является ли модель вредоносной."
        ),
        compliance_tags=[
            "owasp-ml:ml10",
            "fstec:ubi-174",
            "fstec:ubi-067",
            "gost:56939-2024:5.5",
            "gost:56939-2024:6.1",
        ],
    )


def _make_data_image_issue(url: str, location: str) -> Issue:
    """Создаёт Issue INFO для data:image/… URL — встроенное превью-изображение.

    В metadata LoRA/safetensors-файлов такие URL легитимны: авторы встраивают
    JPEG/PNG для предварительного просмотра. Опасности не несут.
    """
    return Issue(
        code="MLS-NET-007",
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        message=f"Встроенное изображение (data:image) в metadata файла: {url[:80]}…",
        location=location,
        details={"url": url[:200], "scheme": "data:image"},
        why=(
            "URL data:image/ содержит base64-кодированное изображение — "
            "обычно превью модели или LoRA. Угрозы не представляет."
        ),
        remediation=(
            "Встроенные изображения в metadata допустимы. "
            "Убедитесь, что файл получен из доверенного источника."
        ),
        compliance_tags=[
            "owasp-ml:ml03",
        ],
    )


def _make_dangerous_scheme_issue(url: str, location: str) -> Issue:
    """Создаёт Issue HIGH для опасных URL-схем: file/javascript/data/vbscript.

    Появляются (аудит #9) для защиты от SSRF / LFI через PDF-рендер и от
    XSS-style payload в HTML-контексте отчётов.
    """
    lower = url.lower()
    if lower.startswith("file:"):
        scheme = "file"
        why = (
            "URL со схемой file:// внутри ML-файла указывает на попытку чтения "
            "локальных ресурсов сервера. При рендере PDF-отчёта или другой "
            "обработке URL атакующий может получить содержимое файлов вроде "
            "/etc/passwd (LFI / SSRF)."
        )
    elif lower.startswith("javascript:"):
        scheme = "javascript"
        why = (
            "URL со схемой javascript: позволяет выполнить код в HTML-контексте. "
            "При рендере отчёта это XSS / RCE на стороне аудитора."
        )
    elif lower.startswith("data:"):
        scheme = "data"
        why = (
            "URL со схемой data: содержит inline-payload (часто base64). "
            "Используется для встраивания скриптов и обхода CSP-политик в HTML."
        )
    else:  # vbscript
        scheme = "vbscript"
        why = (
            "URL со схемой vbscript: позволяет выполнить VBScript в IE/HTA-контексте. "
            "Признак вредоносного payload."
        )
    return Issue(
        code="MLS-NET-006",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=f"Опасная URL-схема {scheme}: {url[:120]}",
        location=location,
        details={"url": url, "scheme": scheme},
        why=why,
        remediation=(
            "Не используйте эту модель в production. "
            "Если источник доверенный — попросите пересохранить без таких URL."
        ),
        compliance_tags=[
            "owasp-ml:ml03",
            "owasp-ml:ml10",
            "fstec:ubi-067",
            "fstec:ubi-174",
            "gost:56939-2024:5.5",
            "gost:56939-2024:6.1",
        ],
    )


def _classify_standalone_ip(
    raw_data: RawScanData,
    raw_ip: str,
    ip_str: str,
    location: str,
) -> Issue | None:
    """Создаёт Issue для standalone IP-адреса (не входящего в URL).

    Приватные и loopback адреса игнорируются — они легитимны в конфигурациях.
    """
    if _is_private_or_loopback(ip_str):
        return None

    return Issue(
        code="MLS-NET-005",
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        message=f"Обнаружен публичный IP-адрес: {ip_str}",
        location=location,
        details={"ip": raw_ip, "ip_clean": ip_str, "is_private": False},
        why=(
            "Публичный IP-адрес внутри ML-файла может указывать на "
            "жёстко закодированную точку C2-сервера или цель для утечки данных."
        ),
        remediation=(
            "Проверьте, откуда этот IP. "
            "Если модель не должна обращаться к внешним адресам — это признак атаки."
        ),
        compliance_tags=[
            "owasp-ml:ml10",
            "fstec:ubi-174",
            "gost:56939-2024:5.5",
            "gost:56939-2024:6.1",
        ],
    )

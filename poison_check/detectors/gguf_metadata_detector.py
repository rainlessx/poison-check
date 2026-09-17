"""Детектор подозрительных паттернов в metadata GGUF-файлов.

Извлечён из GGUFScanner: сканеры знают формат,
детекторы знают угрозы. Раньше сканер сам собирал Issues и записывал их
в metadata как строки — пользователь их не видел.

Анализирует:
- ключи metadata, содержащие подозрительные слова (script/code/exec/eval/cmd);
- значения, содержащие URL или IP-адреса;
- аномально длинные строковые значения (потенциальный base64-payload).

URL и IP отчасти дублируют NetworkDetector; здесь оставлены, потому что
NetworkDetector работает с raw_data.strings, а GGUF-метаданные имеют
имя ключа — это даёт более информативное сообщение.
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar

from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Severity,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.gguf_scanner import (
    META_BAD_MAGIC,
    META_BAD_MAGIC_HEX,
    META_EXPECTED_MAGIC,
    META_HEADER_VALID,
)

logger = logging.getLogger(__name__)

# Паттерны для детекции URL и IP-адресов (синхронизированы с GGUFScanner v0.1)
_URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://[^\s\"'<>]{3,}|ftp://[^\s\"'<>]{3,}",
    re.IGNORECASE,
)
_IP_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b",
)

# Подозрительные ключи, указывающие на возможный исполняемый код
_SUSPICIOUS_KEY_PATTERNS: tuple[str, ...] = (
    "script",
    "code",
    "exec",
    "eval",
    "cmd",
    "command",
)

# Системные ключи, добавляемые сканером (начинаются с подчёркивания)
_SYSTEM_KEY_PREFIX: str = "_"

# Версионные ключи, в которых IP-подобные строки (1.2.3.4) — это версии, а не IoC
_VERSION_KEYS: frozenset[str] = frozenset(
    {"version", "model_version", "gguf_version", "general.version"}
)

# Префикс файлов, к которым применим детектор. Без префикса детектор был бы
# вынужден угадывать формат, что нарушает разделение слоёв.
_GGUF_SCANNER_NAME: str = "gguf"

# Поддерживаемые версии формата GGUF (синхронизировано с GGUFScanner)
_SUPPORTED_GGUF_VERSIONS: frozenset[int] = frozenset({1, 2, 3})

# Системные ключи, добавляемые сканером как факты заголовка — не анализируются
# как обычные metadata-значения (не URL/IP/подозрительные ключи).
_SYSTEM_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "gguf_version",
        "tensor_count",
        "kv_count",
        META_HEADER_VALID,
        META_BAD_MAGIC,
        META_BAD_MAGIC_HEX,
        META_EXPECTED_MAGIC,
    }
)


@DetectorRegistry.register
class GGUFMetadataDetector(BaseDetector):
    """Анализирует metadata GGUF-файлов на подозрительные паттерны.

    Срабатывает только для RawScanData, полученных от GGUFScanner
    (проверяется по scanner_name). Это намеренно: GGUF — единственный
    формат, в котором структурированные пары ключ-значение могут содержать
    URL/exec/script-payload как часть формата.
    """

    name: ClassVar[str] = "gguf_metadata"
    description: ClassVar[str] = (
        "Детектор подозрительных паттернов в metadata GGUF-файлов"
    )
    severity_range: ClassVar[tuple[Severity, Severity]] = (
        Severity.LOW,
        Severity.MEDIUM,
    )

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Возвращает список Issues по metadata GGUF-файла.

        :param raw_data: Результат сканирования.
        :param context: ML-контекст (не используется — GGUF не привязан к фреймворку).
        :return: Список Issues, пустой если scanner_name != 'gguf' или metadata пустые.
        """
        if raw_data.scanner_name != _GGUF_SCANNER_NAME:
            return []

        if raw_data.error is not None and not raw_data.metadata:
            return []

        if not raw_data.metadata:
            return []

        issues: list[Issue] = []
        location = str(raw_data.file_path)

        # Аномалия заголовка: неверные magic-байты (подделка/искажение формата).
        # Симметрично проверке версии ниже — обе аномалии заголовка идут одним
        # путём «сканер зафиксировал факт → детектор эмитит Issue».
        if raw_data.metadata.get(META_HEADER_VALID) == "false":
            issues.append(_make_bad_magic_issue(raw_data.metadata, location))

        # Проверка версии формата
        version_str = raw_data.metadata.get("gguf_version")
        if version_str is not None:
            try:
                version_int = int(version_str)
                if version_int not in _SUPPORTED_GGUF_VERSIONS:
                    issues.append(
                        Issue(
                            code="MLS-GGUF-005",
                            severity=Severity.INFO,
                            confidence=Confidence.MEDIUM,
                            message=(
                                f"Неизвестная версия GGUF: {version_int}. "
                                f"Поддерживаемые: {sorted(_SUPPORTED_GGUF_VERSIONS)}."
                            ),
                            location=location,
                            details={"gguf_version": version_int},
                            why=(
                                "Неизвестная версия формата может означать "
                                "будущую версию спецификации либо намеренное "
                                "искажение."
                            ),
                            remediation=(
                                "Проверьте происхождение файла и обновите "
                                "poison-check до актуальной версии."
                            ),
                        )
                    )
            except ValueError:
                pass

        for key, value in raw_data.metadata.items():
            # Пропускаем системные ключи, добавленные сканером (например, gguf_version)
            if key.startswith(_SYSTEM_KEY_PREFIX):
                continue
            # Системные не-_-префиксные ключи сканера тоже пропускаем
            if key in _SYSTEM_METADATA_KEYS:
                continue

            issues.extend(_analyze_key(key, value, location))
            issues.extend(_analyze_value_urls(key, value, location))
            issues.extend(_analyze_value_ips(key, value, location))
            issues.extend(_analyze_value_length(key, value, location))

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def _make_bad_magic_issue(metadata: dict[str, str], location: str) -> Issue:
    """Строит Issue MLS-GGUF-006 для неверных magic-байтов GGUF.

    Severity СРЕДНИЙ (обоснование): заголовок GGUF обязан начинаться с
    фиксированной сигнатуры b'GGUF'. Её подмена — более сильный сигнал искажения
    формата, чем неизвестная версия (MLS-GGUF-005, ИНФО): файл либо не является
    GGUF, либо намеренно замаскирован под GGUF по расширению. Этого достаточно,
    чтобы не пропустить файл в пайплайн без внимания аналитика, но не CRITICAL —
    доказанного RCE-payload здесь нет (только искажение сигнатуры). Отсюда
    Severity.MEDIUM (≥ уровня MLS-GGUF-005). Confidence.HIGH — сам факт
    несовпадения magic достоверен (байты прочитаны).

    В message и details пишутся СЫРЫЕ прочитанные байты (repr + hex) и ожидаемое
    значение — forensic-инвариант, ничего не нормализуется.
    """
    raw_magic = metadata.get(META_BAD_MAGIC, "?")
    raw_magic_hex = metadata.get(META_BAD_MAGIC_HEX, "")
    expected = metadata.get(META_EXPECTED_MAGIC, repr(b"GGUF"))
    hex_part = f" (hex {raw_magic_hex})" if raw_magic_hex else ""
    return Issue(
        code="MLS-GGUF-006",
        severity=Severity.MEDIUM,
        confidence=Confidence.HIGH,
        message=(
            f"Неверные magic-байты GGUF: {raw_magic}{hex_part}, "
            f"ожидалось {expected}. Возможна подделка или искажение формата."
        ),
        location=location,
        details={
            "bad_magic": raw_magic,
            "bad_magic_hex": raw_magic_hex,
            "expected_magic": expected,
        },
        why=(
            "Заголовок GGUF обязан начинаться с сигнатуры b'GGUF'. Неверные "
            "magic-байты означают, что файл не является корректным GGUF: либо "
            "повреждён, либо намеренно искажён (маскировка под GGUF по "
            "расширению). Такой файл нельзя считать доверенной моделью."
        ),
        remediation=(
            "Не загружайте файл в Ollama/llama.cpp. Проверьте происхождение и "
            "целостность файла; при сомнении в подлинности — отклоните его."
        ),
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )


def _key_has_word(key_lower: str, keyword: str) -> bool:
    """True если keyword является отдельной частью ключа (разделитель: . _ -).

    Например: 'clip.has_vision_encoder' → parts=['clip','has','vision','encoder']
    Слово 'code' НЕ матчит 'encoder' и НЕ матчит 'decoder', только точное совпадение.
    """
    return keyword in re.split(r"[.\-_]", key_lower)


def _analyze_key(key: str, value: str, location: str) -> list[Issue]:
    """Возвращает Issue если имя ключа содержит подозрительный паттерн как отдельное слово."""
    key_lower = key.lower()
    for suspicious_keyword in _SUSPICIOUS_KEY_PATTERNS:
        if _key_has_word(key_lower, suspicious_keyword):
            return [
                Issue(
                    code="MLS-GGUF-001",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.LOW,
                    message=(
                        f"Подозрительный ключ metadata в GGUF-файле: {key!r}. "
                        f"Ключ содержит паттерн '{suspicious_keyword}', "
                        "возможно указывает на встроенный исполняемый код."
                    ),
                    location=location,
                    details={"key": key, "value_preview": value[:200]},
                    why=(
                        "Ключи вида 'script', 'code', 'exec' в metadata GGUF-файла "
                        "нетипичны для легитимных LLM-моделей и могут указывать "
                        "на попытку внедрения исполняемого кода."
                    ),
                    remediation=(
                        "Проверьте источник файла. Не запускайте этот файл "
                        "в Ollama/llama.cpp без подтверждения подлинности."
                    ),
                    compliance_tags=[
                        "owasp-ml:ml03",
                        "fstec:ubi-067",
                        "gost:56939-2024:5.3",
                    ],
                )
            ]
    return []


def _analyze_value_urls(key: str, value: str, location: str) -> list[Issue]:
    """Возвращает Issues для каждого URL, найденного в значении."""
    issues: list[Issue] = []
    for url in _URL_PATTERN.findall(value):
        issues.append(
            Issue(
                code="MLS-GGUF-002",
                severity=Severity.LOW,
                confidence=Confidence.HIGH,
                message=(
                    f"URL обнаружен в metadata GGUF-файла (ключ: {key!r}): {url}"
                ),
                location=location,
                details={"key": key, "url": url},
                why=(
                    "Наличие URL в metadata LLM-модели нетипично для легитимных "
                    "файлов. Может быть индикатором C2-сервера или утечки данных."
                ),
                remediation=(
                    "Изучите URL вручную. При подозрении на вредоносность — "
                    "не используйте модель в production-окружении."
                ),
                compliance_tags=[
                    "owasp-ml:ml10",
                    "fstec:ubi-174",
                    "gost:56939-2024:5.5",
                ],
            )
        )
    return issues


def _analyze_value_ips(key: str, value: str, location: str) -> list[Issue]:
    """Возвращает Issues для каждого IP, найденного в значении.

    Версионные ключи пропускаются: 'version: 1.2.3.4' — это версия, не IoC.
    """
    if key.lower() in _VERSION_KEYS:
        return []
    issues: list[Issue] = []
    for ip in _IP_PATTERN.findall(value):
        issues.append(
            Issue(
                code="MLS-GGUF-003",
                severity=Severity.LOW,
                confidence=Confidence.MEDIUM,
                message=(
                    f"IP-адрес обнаружен в metadata GGUF-файла "
                    f"(ключ: {key!r}): {ip}"
                ),
                location=location,
                details={"key": key, "ip": ip},
                why=(
                    "IP-адрес в metadata LLM-модели может указывать "
                    "на C2-инфраструктуру."
                ),
                remediation=(
                    "Проверьте IP-адрес через threat intelligence. "
                    "При сомнениях — не используйте модель."
                ),
                compliance_tags=[
                    "owasp-ml:ml10",
                    "fstec:ubi-174",
                    "gost:56939-2024:5.5",
                ],
            )
        )
    return issues


def _analyze_value_length(key: str, value: str, location: str) -> list[Issue]:
    """Возвращает Issue если значение аномально длинное (>10 000 символов)."""
    if len(value) <= 10_000:
        return []
    return [
        Issue(
            code="MLS-GGUF-004",
            severity=Severity.LOW,
            confidence=Confidence.LOW,
            message=(
                f"Аномально длинная строка в metadata GGUF-файла "
                f"(ключ: {key!r}, длина: {len(value)} символов)."
            ),
            location=location,
            details={"key": key, "value_length": len(value)},
            why=(
                "Аномально длинные строки в metadata нетипичны для легитимных "
                "моделей. Это может быть признаком встроенных данных или обфускации."
            ),
            remediation=(
                "Изучите содержимое строки. "
                "Возможно, это base64-кодированный payload."
            ),
            compliance_tags=[
                "owasp-ml:ml10",
                "fstec:ubi-067",
                "gost:56939-2024:5.5",
            ],
        )
    ]

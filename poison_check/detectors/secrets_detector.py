"""Детектор секретов и чувствительных данных в ML-файлах.

Ищет API-ключи, токены, пароли и другие секреты в строках и метаданных файла.
Паттерны загружаются из rules/secrets/patterns.yaml — обновляются независимо от кода.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml  # type: ignore[import-untyped]

from poison_check._paths import RULES_DIR
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

_DEFAULT_PATTERNS_PATH = RULES_DIR / "secrets" / "patterns.yaml"

# Отображение строкового severity из YAML на Severity enum
_SEVERITY_MAP: dict[str, Severity] = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "info": Severity.INFO,
}


@dataclass
class _SecretPattern:
    """Скомпилированный паттерн для обнаружения секрета."""

    name: str
    regex: re.Pattern[str]
    severity: Severity
    description_ru: str
    why_ru: str
    remediation_ru: str


def _load_patterns(path: Path) -> list[_SecretPattern]:
    """Загружает паттерны из YAML-файла и компилирует регулярные выражения.

    При отсутствии файла или ошибке — возвращает пустой список (graceful degradation).
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info(
            "Файл паттернов секретов не найден: %s — SecretsDetector не будет срабатывать",
            path,
        )
        return []
    except OSError as exc:
        logger.warning("Не удалось прочитать файл паттернов: %s: %s", path, exc)
        return []
    except yaml.YAMLError as exc:
        logger.warning("Ошибка разбора YAML паттернов: %s: %s", path, exc)
        return []

    if not isinstance(raw, dict) or "patterns" not in raw:
        logger.warning(
            "Неожиданный формат файла паттернов %s: ожидается dict с ключом 'patterns'",
            path,
        )
        return []

    entries = raw["patterns"]
    if not isinstance(entries, list):
        logger.warning("Поле 'patterns' в %s должно быть списком", path)
        return []

    result: list[_SecretPattern] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            pattern = _parse_pattern_entry(entry)
            result.append(pattern)
        except (KeyError, ValueError, re.error) as exc:
            rule_name = entry.get("name", "?") if isinstance(entry, dict) else "?"
            logger.warning("Пропускаем некорректный паттерн '%s': %s", rule_name, exc)

    logger.debug("Загружено паттернов секретов: %d из %s", len(result), path)
    return result


def _parse_pattern_entry(entry: dict[str, object]) -> _SecretPattern:
    """Разбирает одну запись из YAML в _SecretPattern.

    Бросает KeyError, ValueError или re.error если запись некорректна.
    """
    name = str(entry["name"])
    raw_pattern = str(entry["pattern"])
    severity_str = str(entry.get("severity", "high")).lower()
    case_sensitive = entry.get("case_sensitive", True)

    if severity_str not in _SEVERITY_MAP:
        raise ValueError(f"Неизвестный severity: {severity_str!r}")

    flags = re.MULTILINE
    if not case_sensitive:
        flags |= re.IGNORECASE

    regex = re.compile(raw_pattern, flags)

    return _SecretPattern(
        name=name,
        regex=regex,
        severity=_SEVERITY_MAP[severity_str],
        description_ru=str(entry.get("description_ru", f"Обнаружен секрет: {name}")),
        why_ru=str(entry.get("why_ru", "")),
        remediation_ru=str(entry.get("remediation_ru", "")),
    )


@DetectorRegistry.register
class SecretsDetector(BaseDetector):
    """Детектор секретов — ищет API-ключи, токены, пароли в ML-файлах.

    Проверяет:
    - raw_data.strings — строки, извлечённые сканером из pickle/GGUF/etc.
    - raw_data.metadata — метаданные safetensors-файлов (строковые значения).

    Паттерны загружаются из rules/secrets/patterns.yaml.
    Каждый паттерн может иметь severity critical/high/medium.
    """

    name: ClassVar[str] = "secrets"
    description: ClassVar[str] = "Детектор секретов: API-ключи, токены, пароли в ML-файлах"
    severity_range: ClassVar[tuple[Severity, Severity]] = (Severity.MEDIUM, Severity.CRITICAL)

    def __init__(self, patterns_path: Path | None = None) -> None:
        """Загружает паттерны из YAML.

        patterns_path: путь к YAML с паттернами. По умолчанию — rules/secrets/patterns.yaml.
        """
        path = patterns_path if patterns_path is not None else _DEFAULT_PATTERNS_PATH
        self._patterns: list[_SecretPattern] = _load_patterns(path)

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Ищет секреты в строках и метаданных файла.

        Алгоритм:
        1. Собирает все текстовые значения из raw_data.strings и raw_data.metadata.
        2. Для каждого значения проверяет все паттерны.
        3. При совпадении создаёт Issue с соответствующим severity.

        Дедупликация: один паттерн в одном значении — один Issue (не дублируется
        при множественных совпадениях одного паттерна в одной строке).
        """
        if raw_data.error is not None:
            return []

        issues: list[Issue] = []
        seen: set[tuple[str, str]] = set()  # (pattern_name, matched_value_preview)

        # --- raw_data.strings ---
        if raw_data.strings:
            for string_info in raw_data.strings:
                for pattern in self._patterns:
                    match = pattern.regex.search(string_info.value)
                    if match:
                        matched = match.group(0)
                        dedup_key = (pattern.name, _redact(matched))
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)
                        location = f"{raw_data.file_path} (offset {string_info.position})"
                        issues.append(
                            _make_secret_issue(raw_data, pattern, matched, location)
                        )

        # --- raw_data.metadata (safetensors и другие форматы с метаданными) ---
        if raw_data.metadata:
            for meta_key, meta_value in raw_data.metadata.items():
                if not isinstance(meta_value, str):   # ← guard against non-string metadata
                    continue
                for pattern in self._patterns:
                    match = pattern.regex.search(meta_value)
                    if match:
                        matched = match.group(0)
                        dedup_key = (pattern.name, _redact(matched))
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)
                        location = f"{raw_data.file_path} (metadata key: {meta_key!r})"
                        issues.append(
                            _make_secret_issue(raw_data, pattern, matched, location)
                        )

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _redact(value: str) -> str:
    """Возвращает частично замаскированное значение для дедупликации и логов.

    Показывает первые 6 и последние 4 символа, остальное заменяет на ***.
    """
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}***{value[-4:]}"


def _make_secret_issue(
    raw_data: RawScanData,
    pattern: _SecretPattern,
    matched: str,
    location: str,
) -> Issue:
    """Создаёт Issue для найденного секрета (MLS020)."""
    return Issue(
        code="MLS-SEC-001",
        severity=pattern.severity,
        confidence=Confidence.HIGH,
        message=pattern.description_ru,
        location=location,
        details={
            "pattern_name": pattern.name,
            "matched_preview": _redact(matched),
            "file": str(raw_data.file_path),
        },
        why=pattern.why_ru or None,
        remediation=pattern.remediation_ru or None,
        compliance_tags=[
            "owasp-ml:ml09",
            "fstec:ubi-037",
            "gost:56939-2024:5.5",
        ],
    )

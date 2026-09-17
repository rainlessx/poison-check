"""Детектор на основе allowlist — всё вне разрешённого списка подозрительно."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import yaml  # type: ignore[import-untyped]

from poison_check._paths import RULES_DIR
from poison_check.core.detector_base import BaseDetector
from poison_check.core.known_dangerous import (
    DANGEROUS_FUNCTION_NAMES,
    KNOWN_DANGEROUS_GLOBALS,
    normalize_global,
)
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Severity,
)
from poison_check.core.scanner_base import RawScanData

logger = logging.getLogger(__name__)

_ALLOWLIST_DIR = RULES_DIR / "allowlist"


def _load_allowlist_dir(
    directory: Path,
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, frozenset[str]]]:
    """Загружает все YAML-файлы из директории allowlist.

    Возвращает пару:
      - allowlists: framework → set[(module, name)]
      - trusted_prefixes: framework → frozenset[prefix]

    Ошибки разбора отдельных файлов логируются и не останавливают загрузку.
    """
    allowlists: dict[str, set[tuple[str, str]]] = {}
    trusted_prefixes: dict[str, frozenset[str]] = {}

    if not directory.is_dir():
        logger.warning(
            "Директория allowlist не найдена: %s — AllowlistDetector будет работать "
            "без allowlist (все глобалы будут подозрительны)",
            directory,
        )
        return allowlists, trusted_prefixes

    for yaml_path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("Не удалось прочитать allowlist %s: %s", yaml_path, exc)
            continue
        except yaml.YAMLError as exc:
            logger.warning("Ошибка разбора allowlist %s: %s", yaml_path, exc)
            continue

        if not isinstance(raw, dict):
            logger.warning(
                "Некорректный формат allowlist %s: ожидается dict, получен %s",
                yaml_path,
                type(raw).__name__,
            )
            continue

        framework = raw.get("framework")
        if not isinstance(framework, str) or not framework:
            logger.warning(
                "Allowlist %s: отсутствует или пустое поле 'framework' — пропускаем",
                yaml_path,
            )
            continue

        globals_list = raw.get("globals")
        if not isinstance(globals_list, list):
            logger.warning(
                "Allowlist %s: поле 'globals' не является списком — пропускаем",
                yaml_path,
            )
            continue

        pairs: set[tuple[str, str]] = set()
        for entry in globals_list:
            if not isinstance(entry, dict):
                continue
            module = entry.get("module")
            name = entry.get("name")
            if isinstance(module, str) and isinstance(name, str):
                pairs.add((module, name))
            else:
                logger.debug(
                    "Allowlist %s: пропускаем запись без module/name: %s",
                    yaml_path,
                    entry,
                )

        if framework in allowlists:
            allowlists[framework] |= pairs
        else:
            allowlists[framework] = pairs

        # Разбираем trusted_prefixes (необязательное поле)
        raw_prefixes = raw.get("trusted_prefixes")
        if isinstance(raw_prefixes, list):
            prefixes = frozenset(p for p in raw_prefixes if isinstance(p, str) and p)
            if framework in trusted_prefixes:
                trusted_prefixes[framework] = trusted_prefixes[framework] | prefixes
            else:
                trusted_prefixes[framework] = prefixes

        logger.debug(
            "Загружен allowlist %s: фреймворк=%s, записей=%d, trusted_prefixes=%d",
            yaml_path.name,
            framework,
            len(pairs),
            len(trusted_prefixes.get(framework, frozenset())),
        )

    return allowlists, trusted_prefixes


@DetectorRegistry.register
class AllowlistDetector(BaseDetector):
    """Allowlist-first детектор — всё вне allowlist подозрительно.

    Загружает разрешённые (module, name)-пары из YAML-файлов в rules/allowlist/.
    Для каждого глобала, отсутствующего в allowlist, создаёт Issue с
    severity=MEDIUM и confidence=LOW.

    Не дублирует находки BlocklistDetector: если глобал уже покрыт
    hardcoded blocklist, AllowlistDetector его пропускает — пользователь
    получит один Issue вместо двух.

    Если фреймворк не определён (MLContext.framework == "unknown") —
    используется объединение всех загруженных allowlists (консервативная стратегия).
    """

    name: ClassVar[str] = "allowlist"
    description: ClassVar[str] = (
        "Allowlist-first детектор — всё вне allowlist подозрительно"
    )
    severity_range: ClassVar[tuple[Severity, Severity]] = (Severity.INFO, Severity.MEDIUM)

    # Множество заведомо опасных globals из core/known_dangerous.py.
    # AllowlistDetector пропускает эти пары — BlocklistDetector уже создаст Issue,
    # дублировать не нужно. Нет прямой зависимости от BlocklistDetector.
    _BLOCKLIST: ClassVar[frozenset[tuple[str, str]]] = KNOWN_DANGEROUS_GLOBALS

    _DANGEROUS_NAMES: ClassVar[frozenset[str]] = DANGEROUS_FUNCTION_NAMES

    def __init__(self, allowlist_dir: Path | None = None) -> None:
        """Загружает все YAML из директории allowlist и строит индекс.

        allowlist_dir: путь к директории с YAML-файлами. По умолчанию —
        rules/allowlist/ относительно корня проекта.
        """
        directory = allowlist_dir if allowlist_dir is not None else _ALLOWLIST_DIR
        self._allowlists, self._trusted_prefixes = _load_allowlist_dir(directory)
        # Объединение всех allowlists — используется для unknown framework
        self._combined: set[tuple[str, str]] = set()
        for pairs in self._allowlists.values():
            self._combined |= pairs
        # Объединение всех trusted_prefixes — используется для unknown framework
        self._combined_prefixes: frozenset[str] = frozenset(
            p for prefixes in self._trusted_prefixes.values() for p in prefixes
        )

    def _get_allowlist(self, context: MLContext) -> set[tuple[str, str]]:
        """Возвращает allowlist для фреймворка из context.

        Особый случай ``transformers`` (аудит #28): HF-модели всегда используют
        и transformers.*, и torch.* — поэтому при detection transformers
        возвращаем объединение transformers ⊕ pytorch ⊕ numpy. Без этого
        на каждой HF-модели был бы шум по torch.nn.* классам.

        Если фреймворк неизвестен или не загружен — возвращает объединение
        всех allowlists (консервативная стратегия: меньше false positive).
        """
        framework = context.framework.lower()

        # transformers → объединение нескольких allowlist'ов
        if framework == "transformers":
            combined = set(self._allowlists.get("transformers", set()))
            combined |= self._allowlists.get("pytorch", set())
            combined |= self._allowlists.get("numpy", set())
            return combined

        if framework in self._allowlists:
            return self._allowlists[framework]
        # unknown или нераспознанный фреймворк → объединённый allowlist
        if framework != "unknown":
            logger.debug(
                "Фреймворк '%s' не найден в allowlists — используется объединённый",
                framework,
            )
        return self._combined

    def _get_trusted_prefixes(self, context: MLContext) -> frozenset[str]:
        """Возвращает trusted_prefixes для фреймворка из context.

        transformers использует объединение нескольких наборов префиксов
        (аналогично _get_allowlist).
        """
        framework = context.framework.lower()

        if framework == "transformers":
            combined: set[str] = set(self._trusted_prefixes.get("transformers", frozenset()))
            combined |= self._trusted_prefixes.get("pytorch", frozenset())
            combined |= self._trusted_prefixes.get("numpy", frozenset())
            return frozenset(combined)

        if framework in self._trusted_prefixes:
            return self._trusted_prefixes[framework]
        if framework != "unknown":
            logger.debug(
                "Фреймворк '%s' не найден в trusted_prefixes — используется объединённый",
                framework,
            )
        return self._combined_prefixes

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Проверяет globals против allowlist и возвращает список подозрительных.

        Алгоритм:
        1. Если globals отсутствуют — нечего проверять, возвращаем [].
        2. Вычитаем allowlist из globals — получаем подозрительные пары.
        3. Пары, уже покрытые BlocklistDetector, пропускаем (нет дубликатов).
        4. Модуль входит в trusted_prefixes → Issue severity=INFO (код MLS-ALW-002).
        5. Иначе → Issue severity=MEDIUM (код MLS-ALW-001).
        """
        if raw_data.globals is None:
            return []

        allowlist = self._get_allowlist(context)
        trusted_prefixes = self._get_trusted_prefixes(context)
        suspicious = raw_data.globals - allowlist

        issues: list[Issue] = []
        for module, name in sorted(suspicious):  # sorted для детерминированного порядка
            # Нормализуем Py2/legacy алиасы: __builtin__.__import__ должен
            # уступить BlocklistDetector'у так же, как builtins.__import__.
            if normalize_global(module, name) in self._BLOCKLIST:
                # BlocklistDetector уже поймает это — не дублируем
                continue
            is_trusted = any(module.startswith(p) for p in trusted_prefixes)
            if is_trusted and name in self._DANGEROUS_NAMES:
                # trusted_prefix не защищает опасные имена функций:
                # "torch.utils.evil.system" опасен так же, как "os.system"
                severity = Severity.HIGH
            elif is_trusted:
                severity = Severity.INFO
            else:
                severity = Severity.MEDIUM
            issues.append(
                _make_allowlist_issue(raw_data, module, name, context.framework, severity)
            )

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _make_allowlist_issue(
    raw_data: RawScanData,
    module: str,
    name: str,
    framework: str,
    severity: Severity = Severity.MEDIUM,
) -> Issue:
    """Создаёт Issue для глобала, отсутствующего в allowlist.

    MLS-ALW-001 (MEDIUM): модуль из незнакомого пространства имён.
    MLS-ALW-002 (INFO): модуль из доверенного пространства имён (sklearn.*, torch.* и др.),
    но конкретный класс не занесён в allowlist — вероятно, просто новый алгоритм.
    """
    if severity == Severity.INFO:
        code = "MLS-ALW-002"
        why = (
            "Класс из доверенного пространства имён отсутствует в allowlist. "
            "Вероятно, это новый алгоритм или версия библиотеки — риск низкий."
        )
        remediation = (
            "Убедитесь, что модель получена из доверенного источника. "
            "Для подавления предупреждения добавьте запись в allowlist."
        )
    else:
        code = "MLS-ALW-001"
        why = (
            "Неизвестный вызов может быть признаком вредоносного кода "
            "или нестандартной сериализации."
        )
        remediation = (
            "Проверьте источник модели. "
            "Если модель доверенная — добавьте запись в allowlist."
        )

    return Issue(
        code=code,
        severity=severity,
        confidence=Confidence.LOW,
        message=(
            f"Глобал {module}.{name} отсутствует в allowlist для {framework}"
        ),
        location=f"{raw_data.file_path}#{module}.{name}",
        details={"module": module, "name": name, "framework": framework},
        why=why,
        remediation=remediation,
        compliance_tags=[
            "owasp-ml:ml03",
            "owasp-ml:ml08",
            "fstec:ubi-067",
            "fstec:ubi-179",
            "gost:56939-2024:5.3",
            "gost:56939-2024:6.2",
            "gost:56939-2024:6.3",
        ],
    )

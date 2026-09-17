"""Загрузчик политик сканирования и РЕЕСТР ключей политики.

Политика задаёт набор активных детекторов, порог внимания к находкам
(``severity_threshold``) и уровень, при достижении которого CLI возвращает
exit code 1 (``fail_on_severity``).

Встроенные политики хранятся в каталоге ``policies/`` в корне проекта.
Пользователь может указать путь к своему YAML-файлу — PolicyLoader его
загрузит так же, как встроенную политику.

**Реестр ключей — защита от «декоративных» ключей политики.**
Дважды в истории проекта случался один и тот же класс дефекта: ключ добавляли
в YAML и в документацию, он исправно парсился, а в коде его никто не читал —
пользователь считал правило работающим, а оно молчало (сначала ``extra_rules``
в banking/government, затем ``severity_threshold``). Точечные починки класс не
закрывали, потому что «ключ есть в файле и парсится» ≠ «ключ влияет на
поведение».

Поэтому каждый ключ политики обязан иметь запись в :data:`POLICY_KEYS` (или
:data:`EXTRA_RULE_KEYS` для секции ``extra_rules``) с явной, машинно-проверяемой
привязкой:

* :attr:`PolicyKeySpec.applied_by` — где ключ читается и влияет на поведение;
* :attr:`PolicyKeySpec.proven_by`  — тест, который это поведение доказывает;
* :attr:`PolicyKeySpec.status`     — ENFORCED / DESCRIPTIVE / RESERVED.

Ссылки записаны как ``"модуль:qualname"`` и резолвятся импортом — мета-тест
``tests/test_policy_keys_enforced.py`` проверяет, что каждая из них существует,
и что в политиках нет ни одного незарегистрированного ключа. Неучтённый ключ =
красный тест, а не тихо игнорируемая настройка.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from poison_check._paths import POLICIES_DIR as _BUILTIN_POLICIES_DIR
from poison_check.core.detector_base import BaseDetector
from poison_check.core.result import PolicyThresholds, Severity

# Имена встроенных политик (без расширения)
BUILTIN_POLICY_NAMES = frozenset({"default", "banking", "government", "strict"})

#: Множитель ГБ → байты. Совпадает с тем, что использует CLI-флаг
#: ``--max-file-size`` (десятичный ГБ, не гибибайт).
_BYTES_PER_GB = 1_000_000_000

# Значения по умолчанию, если ключ в YAML отсутствует
_DEFAULTS: dict[str, Any] = {
    "name": "default",
    "description": "",
    "enabled_detectors": None,   # None = все детекторы
    "severity_threshold": "medium",
    "fail_on_severity": "critical",
    "compliance": [],
    "extra_rules": {},
}


def _is_relative_to(child: Path, parent: Path) -> bool:
    """Возвращает True если ``child`` лежит внутри ``parent``.

    На Python 3.10 у Path есть метод ``is_relative_to``, но мы используем
    собственную реализацию для большей прозрачности и явного логирования.
    """
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


class PolicyLoader:
    """Загрузчик и валидатор политик сканирования.

    Поддерживает два режима загрузки:
    - по имени встроенной политики: ``PolicyLoader.load("banking")``
    - по пути к пользовательскому YAML: ``PolicyLoader.load("/path/to/custom.yaml")``
    """

    BUILTIN_POLICIES_DIR: Path = _BUILTIN_POLICIES_DIR

    @classmethod
    def load(cls, name: str) -> dict[str, Any]:
        """Загружает политику по имени или пути к YAML-файлу.

        :param name: Имя встроенной политики ("default", "banking", "government",
            "strict") или абсолютный/относительный путь к пользовательскому
            YAML-файлу (должен содержать расширение .yaml или .yml, либо
            существовать как файл).
        :return: Словарь с параметрами политики. Отсутствующие ключи заполняются
            значениями по умолчанию.
        :raises ValueError: Если политика не найдена или содержит ошибки.
        """
        path = cls._resolve_path(name)
        raw = cls._read_yaml(path)
        return cls._apply_defaults(raw)

    @classmethod
    def _resolve_path(cls, name: str) -> Path:
        """Определяет путь к YAML-файлу политики.

        Поддерживает два чётко разделённых режима:

        1. **Имя встроенной политики** (без separator-ов и без расширения):
           ``"banking"``. Ищется в BUILTIN_POLICIES_DIR, после resolve() обязан
           находиться внутри неё — иначе путь вида ``"../../etc/passwd"``
           пытался бы атаковать через директорию политик.

        2. **Явный путь** к пользовательскому YAML (содержит separator или
           расширение .yaml/.yml): ``"./my.yaml"``, ``"/etc/policies/x.yml"``.
           Файл должен существовать; чтение произвольных не-YAML файлов
           через путь без расширения запрещено.

        :raises ValueError: при попытке traversal или несуществующем файле.
        """
        if not name:
            raise ValueError("Имя политики пустое")

        candidate = Path(name)
        looks_like_path = (
            "/" in name
            or "\\" in name
            or candidate.suffix in (".yaml", ".yml")
        )

        if looks_like_path:
            # Явный путь: должен указывать на существующий .yaml/.yml файл.
            if candidate.suffix not in (".yaml", ".yml"):
                raise ValueError(
                    f"Путь к политике должен оканчиваться на .yaml или .yml: "
                    f"'{name}'"
                )
            if not candidate.is_file():
                raise ValueError(
                    f"Файл политики не найден: '{name}'. "
                    f"Проверьте путь или используйте встроенную политику: "
                    f"{sorted(BUILTIN_POLICY_NAMES)}"
                )
            return candidate

        # Имя встроенной политики: запрещаем любые traversal-конструкции.
        # name = "banking" — допустимо.
        # name = ".." — недопустимо: после join будет "policies/...yaml" вне директории.
        if ".." in name or name.startswith("."):
            raise ValueError(
                f"Недопустимое имя политики '{name}': "
                "имя не должно содержать '..' или начинаться с точки"
            )

        builtin_path = cls.BUILTIN_POLICIES_DIR / f"{name}.yaml"

        # Защита от path traversal: после resolve() путь обязан быть внутри
        # BUILTIN_POLICIES_DIR. Если кто-то ухитрился через символьную ссылку
        # или нестандартное имя выйти из неё — отказываем.
        try:
            resolved = builtin_path.resolve(strict=False)
            policies_dir_resolved = cls.BUILTIN_POLICIES_DIR.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"Не удалось разрешить путь к политике '{name}': {exc}"
            ) from exc

        if not _is_relative_to(resolved, policies_dir_resolved):
            raise ValueError(
                f"Путь политики '{name}' указывает за пределы каталога политик "
                f"({policies_dir_resolved}) — отказано"
            )

        if builtin_path.is_file():
            return builtin_path

        raise ValueError(
            f"Политика '{name}' не найдена. "
            f"Встроенные политики: {sorted(BUILTIN_POLICY_NAMES)}. "
            f"Для пользовательской политики укажите путь к YAML-файлу."
        )

    @classmethod
    def _read_yaml(cls, path: Path) -> dict[str, Any]:
        """Читает и парсит YAML-файл политики.

        :raises ValueError: При ошибке чтения или парсинга YAML.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Ошибка чтения файла политики '{path}': {exc}") from exc

        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Ошибка парсинга YAML в файле политики '{path}': {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"Файл политики '{path}' должен содержать YAML-словарь, "
                f"получено: {type(raw).__name__}"
            )

        return raw

    @classmethod
    def _apply_defaults(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Заполняет отсутствующие ключи значениями по умолчанию.

        Не перезаписывает явно указанные значения.
        """
        result: dict[str, Any] = dict(_DEFAULTS)
        result.update(raw)
        return result

    @classmethod
    def list_builtin(cls) -> list[str]:
        """Возвращает список имён встроенных политик."""
        if not cls.BUILTIN_POLICIES_DIR.is_dir():
            return []
        return sorted(
            p.stem
            for p in cls.BUILTIN_POLICIES_DIR.glob("*.yaml")
            if p.stem != ".gitkeep"
        )


# ---------------------------------------------------------------------------
# Аксессоры к extra_rules — единственное место, где эти ключи интерпретируются
# ---------------------------------------------------------------------------


def policy_extra_rules(policy: dict[str, Any]) -> dict[str, Any]:
    """Возвращает секцию ``extra_rules`` политики.

    Политика может прийти из PolicyLoader (где ключ гарантированно есть)
    либо из «сырого» YAML (Scanner-фасад), поэтому отсутствие ключа и
    некорректный тип трактуются как «правил нет».

    :param policy: Загруженная политика.
    :return: Словарь дополнительных правил (пустой, если их нет).
    """
    extra = policy.get("extra_rules")
    if not isinstance(extra, dict):
        return {}
    return cast(dict[str, Any], extra)


def policy_max_file_size_bytes(policy: dict[str, Any]) -> int | None:
    """Возвращает лимит размера файла из ``extra_rules.max_file_size_gb``.

    Значение задаётся в десятичных гигабайтах (как и CLI-флаг
    ``--max-file-size``) и переводится в байты.

    :param policy: Загруженная политика.
    :return: Лимит в байтах или ``None``, если правило не задано либо
        значение некорректно (не число / не положительное).
    """
    raw = policy_extra_rules(policy).get("max_file_size_gb")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if raw <= 0:
        return None
    return int(raw * _BYTES_PER_GB)


def policy_escalate_external_urls(policy: dict[str, Any]) -> bool:
    """Возвращает значение ``extra_rules.no_external_urls``.

    ``true`` означает: URL к домену вне whitelist эскалируется
    MEDIUM → HIGH (NetworkDetector, MLS-NET-002). Для банков и госструктур
    любой внешний URL внутри модели — потенциальный C2-канал.

    :param policy: Загруженная политика.
    :return: True, если эскалация включена.
    """
    return policy_extra_rules(policy).get("no_external_urls") is True


def policy_escalate_unverified(policy: dict[str, Any]) -> bool:
    """Возвращает значение ``extra_rules.no_unverified_files``.

    ``true`` означает: файл, который сканер не смог разобрать из-за отсутствия
    обязательной зависимости, эскалируется MEDIUM → HIGH (KerasThreatDetector,
    MLS-KERAS-003 при неимпортируемом h5py). Для банков и госструктур
    «не проверено» равносильно «не допущено», поэтому такой файл обязан
    валить гейт CI (в этих политиках ``fail_on_severity: high``).

    :param policy: Загруженная политика.
    :return: True, если эскалация включена.
    """
    return policy_extra_rules(policy).get("no_unverified_files") is True


def policy_strict_format_detection(policy: dict[str, Any]) -> bool:
    """Возвращает значение ``extra_rules.strict_format_detection``.

    ``true`` означает: формат файла определяется по СОДЕРЖИМОМУ, и расхождение
    с обещанием расширения даёт находку HIGH (FormatPolicyDetector,
    MLS-FMT-001). Файл, чей формат определить не удалось, расхождением не
    считается — это зона MLS-PARSE-001 / «неподдерживаемый формат».

    :param policy: Загруженная политика.
    :return: True, если строгая проверка формата включена.
    """
    return policy_extra_rules(policy).get("strict_format_detection") is True


def policy_require_safetensors(policy: dict[str, Any]) -> bool:
    """Возвращает значение ``extra_rules.require_safetensors``.

    ``true`` означает: допустимы только форматы, не исполняющие код при
    загрузке (safetensors, GGUF, NumPy без object-dtype). Любой code-bearing
    формат — pickle/joblib/PyTorch/Keras — даёт находку HIGH
    (FormatPolicyDetector, MLS-FMT-002) даже при чистом содержимом: в строгом
    контуре недопустим сам формат, а не только найденный в нём payload.

    :param policy: Загруженная политика.
    :return: True, если требование safetensors включено.
    """
    return policy_extra_rules(policy).get("require_safetensors") is True


def policy_severity_threshold(policy: Mapping[str, Any]) -> Severity | None:
    """Возвращает ``severity_threshold`` политики как :class:`Severity`.

    Это порог ВНИМАНИЯ, а не порог отображения: находки ниже него не исчезают
    из отчёта, а помечаются как подпороговые (см.
    :mod:`poison_check.output.severity_threshold`). Полное скрытие находок
    противоречит инварианту проекта «сигнал не теряется».

    :param policy: Загруженная политика.
    :return: Порог или ``None``, если ключ не задан либо значение некорректно
        (тогда ни одна находка не помечается — поведение «как без порога»).
    """
    return _severity_or_none(policy.get("severity_threshold"))


def policy_fail_on_severity(policy: Mapping[str, Any]) -> Severity:
    """Возвращает ``fail_on_severity`` политики как :class:`Severity`.

    Порог ДЕЙСТВИЯ: с какого уровня CLI возвращает exit code 1. Поддерживается
    устаревший синоним ``exit_on_severity`` — политики, написанные до
    переименования ключа, продолжают валить гейт как раньше.

    :param policy: Загруженная политика.
    :return: Порог; при отсутствии или некорректном значении — CRITICAL.
    """
    value = policy.get("fail_on_severity", policy.get("exit_on_severity"))
    return _severity_or_none(value) or Severity.CRITICAL


def policy_thresholds(policy: Mapping[str, Any]) -> PolicyThresholds:
    """Собирает пороги политики для передачи в :class:`ScanResult`.

    Единая точка, из которой и CLI, и Python API кладут пороги в результат
    сканирования — благодаря этому оба пути форматируют отчёт одинаково.

    :param policy: Загруженная политика.
    :return: Пороги внимания и действия.
    """
    return PolicyThresholds(
        severity_threshold=policy_severity_threshold(policy),
        fail_on_severity=policy_fail_on_severity(policy),
    )


def _severity_or_none(value: Any) -> Severity | None:
    """Преобразует значение из YAML в :class:`Severity` или ``None``.

    Некорректное значение (опечатка, число, ``None``) трактуется как
    «порог не задан»: молча подставлять чужой уровень опаснее, чем не
    применять порог вовсе.
    """
    if not isinstance(value, str):
        return None
    try:
        return Severity(value.strip().lower())
    except ValueError:
        return None


def detector_kwargs_for(policy: dict[str, Any], detector_name: str) -> dict[str, Any]:
    """Возвращает kwargs конструктора детектора, продиктованные политикой.

    Единая точка, из которой и CLI (``_get_cached_detectors``), и Python API
    (``Scanner._get_detectors``) получают параметры детекторов — благодаря
    этому оба пути ведут себя одинаково.

    :param policy: Загруженная политика.
    :param detector_name: Значение ``BaseDetector.name``.
    :return: Словарь именованных аргументов (пустой для детекторов без настроек).
    """
    if detector_name == "network":
        return {"escalate_external_urls": policy_escalate_external_urls(policy)}
    if detector_name == "keras":
        return {"escalate_unverified": policy_escalate_unverified(policy)}
    if detector_name == "format_policy":
        return {
            "strict_format_detection": policy_strict_format_detection(policy),
            "require_safetensors": policy_require_safetensors(policy),
        }
    return {}


def instantiate_detectors(
    detector_classes: list[type[BaseDetector]],
    policy: dict[str, Any],
) -> list[BaseDetector]:
    """Создаёт экземпляры детекторов с учётом параметров из политики.

    :param detector_classes: Классы детекторов (обычно из
        ``DetectorRegistry.enabled_for_policy``).
    :param policy: Загруженная политика.
    :return: Список готовых экземпляров.
    """
    instances: list[BaseDetector] = []
    for cls in detector_classes:
        # cast нужен, потому что BaseDetector не объявляет __init__ —
        # конкретные детекторы принимают собственные именованные аргументы.
        factory = cast(Callable[..., BaseDetector], cls)
        instances.append(factory(**detector_kwargs_for(policy, cls.name)))
    return instances


# ---------------------------------------------------------------------------
# Реестр ключей политики: ключ → точка применения → тест
# ---------------------------------------------------------------------------


class PolicyKeyStatus(Enum):
    """Статус ключа политики.

    * ``ENFORCED``    — ключ влияет на сканирование, отчёт или exit code.
    * ``DESCRIPTIVE`` — метаданные политики; на решения не влияют, но
      обязаны где-то показываться пользователю (иначе это мёртвый ключ).
    * ``RESERVED``    — ключ намеренно не применяется. Требует обоснования и
      маркера ``not enforced yet`` рядом в YAML, чтобы не создавать у
      пользователя ложного ощущения работающего правила.
    """

    ENFORCED = "enforced"
    DESCRIPTIVE = "descriptive"
    RESERVED = "reserved"


@dataclass(frozen=True)
class PolicyKeySpec:
    """Декларация одного ключа политики и его точки применения.

    ``applied_by`` и ``proven_by`` — ссылки вида ``"модуль:qualname"``
    (``"poison_check.cli:_compute_exit_code"``,
    ``"tests.test_policies:TestBankingPolicy.test_high_issue_causes_exit1"``).
    Мета-тест ``tests/test_policy_keys_enforced.py`` резолвит каждую ссылку
    импортом: «точка применения» перестаёт быть знанием в голове и становится
    проверяемым фактом. Ссылка, которая перестала существовать (функцию
    переименовали, тест удалили) — красный тест.

    :ivar key: Имя ключа в YAML.
    :ivar status: Статус (:class:`PolicyKeyStatus`).
    :ivar summary: Что ключ означает — одна строка для отчётов и документации.
    :ivar applied_by: Точки в коде, где ключ читается и влияет на поведение.
    :ivar proven_by: Тесты, доказывающие это влияние.
    :ivar rationale: Обоснование для ``RESERVED`` (почему пока не применяется).
    """

    key: str
    status: PolicyKeyStatus
    summary: str
    applied_by: tuple[str, ...] = ()
    proven_by: tuple[str, ...] = ()
    rationale: str = ""


#: Ключи ВЕРХНЕГО УРОВНЯ политики. Добавляете ключ в YAML — добавьте строку
#: сюда, иначе мета-тест упадёт (и правильно сделает).
POLICY_KEYS: tuple[PolicyKeySpec, ...] = (
    PolicyKeySpec(
        key="name",
        status=PolicyKeyStatus.DESCRIPTIVE,
        summary="Имя политики — показывается пользователю в CLI.",
        applied_by=(
            "poison_check.cli:scan",
            "poison_check.cli:doctor",
        ),
        proven_by=(
            "tests.test_policy_keys_enforced:TestDescriptiveKeysAreShown"
            ".test_doctor_lists_policy_names",
        ),
    ),
    PolicyKeySpec(
        key="description",
        status=PolicyKeyStatus.DESCRIPTIVE,
        summary="Описание политики — показывается в выводе `poison-check doctor`.",
        applied_by=("poison_check.cli:doctor",),
        proven_by=(
            "tests.test_policy_keys_enforced:TestDescriptiveKeysAreShown"
            ".test_doctor_lists_policy_descriptions",
        ),
    ),
    PolicyKeySpec(
        key="enabled_detectors",
        status=PolicyKeyStatus.ENFORCED,
        summary="Список активных детекторов; None — все зарегистрированные.",
        applied_by=("poison_check.core.registry:DetectorRegistry.enabled_for_policy",),
        proven_by=(
            "tests.test_policies:TestDetectorRegistryPolicy"
            ".test_registry_enabled_for_policy_with_real_detectors",
        ),
    ),
    PolicyKeySpec(
        key="disabled_detectors",
        status=PolicyKeyStatus.ENFORCED,
        summary=(
            "Устаревший blocklist-формат выбора детекторов; "
            "приоритет у enabled_detectors."
        ),
        applied_by=("poison_check.core.registry:DetectorRegistry.enabled_for_policy",),
        proven_by=(
            "tests.test_core_registry:test_enabled_for_policy_excludes_disabled",
        ),
    ),
    PolicyKeySpec(
        key="severity_threshold",
        status=PolicyKeyStatus.ENFORCED,
        summary=(
            "Порог ВНИМАНИЯ: находки ниже него остаются в отчёте, но помечаются "
            "как подпороговые во всех форматах (console/JSON/SARIF/SBOM)."
        ),
        applied_by=(
            "poison_check.policies:policy_severity_threshold",
            "poison_check.output.severity_threshold:partition_issues",
        ),
        proven_by=(
            "tests.test_severity_threshold:TestThresholdInAllFormats"
            ".test_below_threshold_marked_in_json",
            "tests.test_severity_threshold:TestThresholdInAllFormats"
            ".test_below_threshold_marked_in_sarif",
            "tests.test_severity_threshold:TestThresholdInAllFormats"
            ".test_below_threshold_marked_in_sbom",
            "tests.test_severity_threshold:TestThresholdInAllFormats"
            ".test_below_threshold_separated_in_console",
        ),
    ),
    PolicyKeySpec(
        key="fail_on_severity",
        status=PolicyKeyStatus.ENFORCED,
        summary="Порог ДЕЙСТВИЯ: с какого severity CLI возвращает exit code 1.",
        applied_by=(
            "poison_check.policies:policy_fail_on_severity",
            "poison_check.cli:_compute_exit_code",
        ),
        proven_by=(
            "tests.test_policies:TestBankingPolicy.test_high_issue_causes_exit1",
        ),
    ),
    PolicyKeySpec(
        key="exit_on_severity",
        status=PolicyKeyStatus.ENFORCED,
        summary="Устаревший синоним fail_on_severity (обратная совместимость).",
        applied_by=("poison_check.policies:policy_fail_on_severity",),
        proven_by=(
            "tests.test_severity_threshold:TestThresholdVsFailOnSeverity"
            ".test_legacy_exit_on_severity_still_gates",
        ),
    ),
    PolicyKeySpec(
        key="compliance",
        status=PolicyKeyStatus.ENFORCED,
        summary="Какие compliance-маперы запускать (fstec / owasp_ml / gost_56939_2024).",
        applied_by=(
            "poison_check.cli:_build_compliance_report",
            "poison_check.scanner:_build_compliance_report",
        ),
        proven_by=(
            "tests.test_compliance:TestComplianceDisclaimer"
            ".test_compliance_report_includes_disclaimer_after_scan",
        ),
    ),
    PolicyKeySpec(
        key="extra_rules",
        status=PolicyKeyStatus.ENFORCED,
        summary="Секция отраслевых правил; её ключи описаны в EXTRA_RULE_KEYS.",
        applied_by=("poison_check.policies:policy_extra_rules",),
        proven_by=(
            "tests.test_policies:TestExtraRulesAccessors"
            ".test_max_file_size_gb_converted_to_bytes",
        ),
    ),
)


#: Ключи секции ``extra_rules``. Правила те же, что и для верхнего уровня.
EXTRA_RULE_KEYS: tuple[PolicyKeySpec, ...] = (
    PolicyKeySpec(
        key="max_file_size_gb",
        status=PolicyKeyStatus.ENFORCED,
        summary="Лимит размера файла в десятичных ГБ; CLI-флаг --max-file-size сильнее.",
        applied_by=(
            "poison_check.policies:policy_max_file_size_bytes",
            "poison_check.cli:scan",
            "poison_check.scanner:Scanner._scan_single_file",
        ),
        proven_by=(
            "tests.test_policies:TestExtraRulesAccessors"
            ".test_builtin_policies_expose_size_limit",
        ),
    ),
    PolicyKeySpec(
        key="no_external_urls",
        status=PolicyKeyStatus.ENFORCED,
        summary="Внешний URL в модели эскалируется MEDIUM → HIGH (MLS-NET-002).",
        applied_by=(
            "poison_check.policies:policy_escalate_external_urls",
            "poison_check.policies:detector_kwargs_for",
        ),
        proven_by=(
            "tests.test_policies:TestDetectorInstantiationFromPolicy"
            ".test_cli_and_api_agree_on_escalation",
        ),
    ),
    PolicyKeySpec(
        key="no_unverified_files",
        status=PolicyKeyStatus.ENFORCED,
        summary=(
            "Непроверенный из-за отсутствия зависимости файл эскалируется "
            "MEDIUM → HIGH (MLS-KERAS-003)."
        ),
        applied_by=(
            "poison_check.policies:policy_escalate_unverified",
            "poison_check.policies:detector_kwargs_for",
        ),
        proven_by=(
            "tests.test_policies:TestExtraRulesAccessors"
            ".test_builtin_policies_escalate_unverified",
        ),
    ),
    PolicyKeySpec(
        key="require_safetensors",
        status=PolicyKeyStatus.ENFORCED,
        summary=(
            "Формат, исполняющий код при загрузке (pickle/joblib/PyTorch/Keras), "
            "недопустим → HIGH (MLS-FMT-002) даже при чистом содержимом."
        ),
        applied_by=(
            "poison_check.policies:policy_require_safetensors",
            "poison_check.policies:detector_kwargs_for",
            "poison_check.detectors.format_policy_detector:FormatPolicyDetector.analyze",
        ),
        proven_by=(
            "tests.test_format_policy:TestRequireSafetensors"
            ".test_clean_pickle_flagged_when_enabled",
            "tests.test_format_policy:TestRequireSafetensors"
            ".test_safetensors_clean_under_both_keys",
            "tests.test_format_policy:TestRequireSafetensors"
            ".test_pickle_not_escalated_when_disabled",
        ),
    ),
    PolicyKeySpec(
        key="require_model_signature",
        status=PolicyKeyStatus.RESERVED,
        summary="Требовать подпись или хеш-верификацию модели.",
        rationale=(
            "Дизайн проработан и зафиксирован — docs/design/model_signature.md "
            "(формат подписи, keystore, слои, коды MLS-SIG-*, деградация, "
            "тест-план, поэтапный план). Реализация сознательно отложена: "
            "проверка подписи требует инфраструктуры доверенных ключей, которая "
            "проектируется под контур конкретного заказчика, а регуляторно "
            "значимый результат в ГОСТ-контуре требует сертифицированного СКЗИ, "
            "которое нельзя поставлять вместе с Apache-2.0 пакетом. Пока у "
            "пользователя нет собственного процесса подписания, правило всегда "
            "отвечает одинаково («подписи нет») и информации не несёт. Статус "
            "пересматривается по условиям из §9.3 дизайн-документа. Помечено "
            "«not enforced yet» в YAML."
        ),
    ),
    PolicyKeySpec(
        key="strict_format_detection",
        status=PolicyKeyStatus.ENFORCED,
        summary=(
            "Формат определяется по содержимому; расхождение с расширением → "
            "HIGH (MLS-FMT-001). Неопознанный формат расхождением не считается."
        ),
        applied_by=(
            "poison_check.policies:policy_strict_format_detection",
            "poison_check.policies:detector_kwargs_for",
            "poison_check.detectors.format_policy_detector:FormatPolicyDetector.analyze",
        ),
        proven_by=(
            "tests.test_format_policy:TestStrictFormatDetection"
            ".test_pickle_disguised_as_safetensors_flagged",
            "tests.test_format_policy:TestStrictFormatDetection"
            ".test_format_taken_from_content_not_from_name",
            "tests.test_format_policy:TestStrictFormatDetection"
            ".test_mismatch_not_flagged_when_disabled",
        ),
    ),
)


def policy_key_specs() -> dict[str, PolicyKeySpec]:
    """Возвращает реестр ключей верхнего уровня в виде ``ключ → спецификация``."""
    return {spec.key: spec for spec in POLICY_KEYS}


def extra_rule_key_specs() -> dict[str, PolicyKeySpec]:
    """Возвращает реестр ключей ``extra_rules`` в виде ``ключ → спецификация``."""
    return {spec.key: spec for spec in EXTRA_RULE_KEYS}


def unregistered_policy_keys(policy: Mapping[str, Any]) -> list[str]:
    """Возвращает ключи верхнего уровня политики, отсутствующие в реестре.

    Непустой список означает декоративный ключ: он есть в YAML, но нигде не
    объявлен, а значит с высокой вероятностью нигде и не читается. Используется
    мета-тестом (гейт для встроенных политик) и CLI (предупреждение для
    пользовательских политик — иначе опечатка вроде ``severity_treshold``
    молча отключает настройку).

    :param policy: Загруженная политика.
    :return: Отсортированный список незарегистрированных ключей.
    """
    known = policy_key_specs()
    return sorted(key for key in policy if key not in known)


def unregistered_extra_rule_keys(policy: Mapping[str, Any]) -> list[str]:
    """Возвращает ключи ``extra_rules``, отсутствующие в реестре.

    :param policy: Загруженная политика.
    :return: Отсортированный список незарегистрированных ключей.
    """
    known = extra_rule_key_specs()
    return sorted(key for key in policy_extra_rules(dict(policy)) if key not in known)

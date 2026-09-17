"""Мета-тест класса «декоративный ключ политики».

Класс дефекта, который закрывает этот файл: ключ появляется в YAML-политике и в
документации, исправно парсится — и нигде не читается. Пользователь настраивает
правило, а оно молчит. Так уже было дважды: сначала ``extra_rules`` в
banking/government, потом ``severity_threshold``. Обычные тесты класс не ловят,
потому что «ключ есть в файле и парсится» ≠ «ключ влияет на поведение».

Гейт устроен так:

1. Полный список ключей берётся из ЭТАЛОННЫХ ИСТОЧНИКОВ — четырёх встроенных
   политик (сырой YAML, без ``_apply_defaults``) и словаря умолчаний
   ``policies._DEFAULTS``. Каждый ключ обязан иметь запись в реестре
   ``POLICY_KEYS`` / ``EXTRA_RULE_KEYS``.
2. Для ENFORCED и DESCRIPTIVE ключей резолвятся ссылки ``applied_by``
   (где ключ читается) и ``proven_by`` (тест, доказывающий поведение).
   Ссылка, указывающая в пустоту, — красный тест.
3. RESERVED-ключ обязан нести обоснование и маркер ``not enforced yet`` рядом
   в YAML: пользователь должен видеть, что правило пока не работает.
4. Ловушка: во временную политику добавляется фиктивный ключ, и проверяется,
   что гейт на нём КРАСНЕЕТ. Иначе мета-тест был бы таким же декоративным, как
   ключи, от которых он защищает.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from poison_check.policies import (
    _DEFAULTS,
    EXTRA_RULE_KEYS,
    POLICY_KEYS,
    PolicyKeySpec,
    PolicyKeyStatus,
    PolicyLoader,
    extra_rule_key_specs,
    policy_key_specs,
    unregistered_extra_rule_keys,
    unregistered_policy_keys,
)

_BUILTIN_NAMES = ("default", "banking", "government", "strict")

runner = CliRunner()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _raw_policy(name: str) -> dict[str, Any]:
    """Читает YAML политики БЕЗ подстановки умолчаний.

    Мета-тест обязан видеть ровно то, что написано в файле: ``_apply_defaults``
    добавил бы ключи, которых в политике нет, и скрыл бы лишние.
    """
    path = PolicyLoader.BUILTIN_POLICIES_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"Политика {name}.yaml должна быть словарём"
    return raw


def _resolve(ref: str) -> object:
    """Резолвит ссылку вида ``"модуль:qualname"`` в объект.

    Именно эта функция делает точку применения машинно-проверяемой: строка в
    реестре либо указывает на существующий объект, либо тест падает.

    :raises AssertionError: Если модуль или атрибут не найдены.
    """
    module_name, sep, qualname = ref.partition(":")
    assert sep and qualname, (
        f"Ссылка {ref!r} должна иметь формат 'модуль:qualname'"
    )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover — сообщение важнее ветки
        raise AssertionError(
            f"Точка применения {ref!r}: модуль {module_name!r} не импортируется ({exc})"
        ) from exc

    obj: Any = module
    for part in qualname.split("."):
        assert hasattr(obj, part), (
            f"Точка применения {ref!r} не найдена: у {obj!r} нет атрибута {part!r}. "
            f"Ключ политики объявлен применяемым, но код/тест отсутствует — "
            f"либо реализуйте его, либо переведите ключ в RESERVED с обоснованием."
        )
        obj = getattr(obj, part)
    return obj


def _all_specs() -> list[tuple[str, PolicyKeySpec]]:
    """Все спецификации обоих реестров в виде ``(область, спецификация)``."""
    return [("policy", s) for s in POLICY_KEYS] + [
        ("extra_rules", s) for s in EXTRA_RULE_KEYS
    ]


def _active_specs() -> list[tuple[str, PolicyKeySpec]]:
    """Спецификации, обязанные иметь точку применения (ENFORCED + DESCRIPTIVE)."""
    return [
        (scope, spec)
        for scope, spec in _all_specs()
        if spec.status is not PolicyKeyStatus.RESERVED
    ]


# ---------------------------------------------------------------------------
# 1. Каждый ключ схемы политики зарегистрирован
# ---------------------------------------------------------------------------


class TestEveryPolicyKeyIsRegistered:
    """Ни одного ключа в политиках вне реестра — иначе он декоративный."""

    @pytest.mark.parametrize("name", _BUILTIN_NAMES)
    def test_top_level_keys_registered(self, name: str) -> None:
        """Все ключи верхнего уровня встроенной политики есть в POLICY_KEYS."""
        unknown = unregistered_policy_keys(_raw_policy(name))
        assert not unknown, (
            f"В политике {name}.yaml есть незарегистрированные ключи верхнего "
            f"уровня: {unknown}. Добавьте их в poison_check.policies.POLICY_KEYS "
            f"с точкой применения — либо удалите из YAML."
        )

    @pytest.mark.parametrize("name", _BUILTIN_NAMES)
    def test_extra_rule_keys_registered(self, name: str) -> None:
        """Все ключи extra_rules встроенной политики есть в EXTRA_RULE_KEYS."""
        unknown = unregistered_extra_rule_keys(_raw_policy(name))
        assert not unknown, (
            f"В политике {name}.yaml есть незарегистрированные ключи extra_rules: "
            f"{unknown}. Добавьте их в poison_check.policies.EXTRA_RULE_KEYS."
        )

    def test_defaults_keys_registered(self) -> None:
        """Каждый ключ из _DEFAULTS тоже описан в реестре.

        _DEFAULTS — вторая половина схемы политики: он подставляется в любую
        политику, поэтому декоративный ключ может завестись и здесь.
        """
        unknown = sorted(set(_DEFAULTS) - set(policy_key_specs()))
        assert not unknown, (
            f"Ключи из policies._DEFAULTS вне реестра: {unknown}"
        )

    def test_registry_has_no_duplicate_keys(self) -> None:
        """Один ключ — одна запись, иначе непонятно, какая из них истинна."""
        for scope, specs in (("policy", POLICY_KEYS), ("extra_rules", EXTRA_RULE_KEYS)):
            keys = [spec.key for spec in specs]
            assert len(keys) == len(set(keys)), (
                f"Дубликаты ключей в реестре {scope}: {sorted(keys)}"
            )


# ---------------------------------------------------------------------------
# 2. Заявленные точки применения и тесты существуют
# ---------------------------------------------------------------------------


class TestApplicationPointsResolve:
    """``applied_by`` / ``proven_by`` резолвятся импортом, а не «на словах»."""

    @pytest.mark.parametrize(
        ("scope", "spec"),
        _active_specs(),
        ids=[f"{scope}.{spec.key}" for scope, spec in _active_specs()],
    )
    def test_applied_by_is_declared_and_resolvable(
        self, scope: str, spec: PolicyKeySpec
    ) -> None:
        """У активного ключа есть хотя бы одна существующая точка применения."""
        assert spec.applied_by, (
            f"Ключ {scope}.{spec.key} объявлен как {spec.status.value}, но не "
            f"указывает ни одной точки применения (applied_by). Такой ключ "
            f"неотличим от декоративного."
        )
        for ref in spec.applied_by:
            _resolve(ref)

    @pytest.mark.parametrize(
        ("scope", "spec"),
        _active_specs(),
        ids=[f"{scope}.{spec.key}" for scope, spec in _active_specs()],
    )
    def test_proven_by_is_declared_and_resolvable(
        self, scope: str, spec: PolicyKeySpec
    ) -> None:
        """У активного ключа есть существующий тест, доказывающий поведение."""
        assert spec.proven_by, (
            f"Ключ {scope}.{spec.key} не указывает теста (proven_by). "
            f"«Код читает ключ» без теста не защищает от регрессии."
        )
        for ref in spec.proven_by:
            _resolve(ref)

    @pytest.mark.parametrize(
        ("scope", "spec"),
        _all_specs(),
        ids=[f"{scope}.{spec.key}" for scope, spec in _all_specs()],
    )
    def test_every_spec_has_summary(self, scope: str, spec: PolicyKeySpec) -> None:
        """Каждая запись реестра объясняет, что ключ означает."""
        assert spec.summary.strip(), f"У ключа {scope}.{spec.key} пустой summary"


# ---------------------------------------------------------------------------
# 3. RESERVED-ключи честно помечены
# ---------------------------------------------------------------------------


class TestReservedKeysAreHonest:
    """Неприменяемый ключ обязан быть виден как неприменяемый."""

    @pytest.mark.parametrize(
        ("scope", "spec"),
        [(s, k) for s, k in _all_specs() if k.status is PolicyKeyStatus.RESERVED],
        ids=[
            f"{s}.{k.key}"
            for s, k in _all_specs()
            if k.status is PolicyKeyStatus.RESERVED
        ],
    )
    def test_reserved_has_rationale(self, scope: str, spec: PolicyKeySpec) -> None:
        """RESERVED-ключ несёт обоснование, почему он пока не применяется."""
        assert spec.rationale.strip(), (
            f"Ключ {scope}.{spec.key} помечен RESERVED без обоснования. "
            f"Без него неотличимо «сознательно отложено» от «забыли подключить»."
        )

    @pytest.mark.parametrize(
        ("scope", "spec"),
        [(s, k) for s, k in _all_specs() if k.status is PolicyKeyStatus.RESERVED],
        ids=[
            f"{s}.{k.key}"
            for s, k in _all_specs()
            if k.status is PolicyKeyStatus.RESERVED
        ],
    )
    def test_reserved_has_no_application_point(
        self, scope: str, spec: PolicyKeySpec
    ) -> None:
        """RESERVED-ключ не должен заявлять точку применения (это противоречие)."""
        assert not spec.applied_by, (
            f"Ключ {scope}.{spec.key} помечен RESERVED, но заявляет applied_by="
            f"{spec.applied_by}. Если он применяется — статус должен быть ENFORCED."
        )

    @pytest.mark.parametrize("name", _BUILTIN_NAMES)
    def test_reserved_keys_marked_in_yaml(self, name: str) -> None:
        """Рядом с RESERVED-ключом в YAML стоит маркер «not enforced yet».

        Расширение прежнего TestNotEnforcedRulesMarked: раньше маркер
        проверялся только для extra_rules и по захардкоженному списку
        enforced-ключей; теперь источник истины — реестр, а покрытие включает
        и ключи верхнего уровня.
        """
        path = PolicyLoader.BUILTIN_POLICIES_DIR / f"{name}.yaml"
        lines = path.read_text(encoding="utf-8").splitlines()
        raw = _raw_policy(name)

        reserved = {
            spec.key
            for spec in (*POLICY_KEYS, *EXTRA_RULE_KEYS)
            if spec.status is PolicyKeyStatus.RESERVED
        }
        present = set(raw) | set(raw.get("extra_rules") or {})

        for key in sorted(reserved & present):
            idx = next(
                i for i, line in enumerate(lines) if line.strip().startswith(f"{key}:")
            )
            preceding = "\n".join(lines[max(0, idx - 3):idx])
            assert "not enforced yet" in preceding, (
                f"Ключ '{key}' в {name}.yaml не помечен как «not enforced yet»"
            )


# ---------------------------------------------------------------------------
# 4. Ловушка: фиктивный ключ обязан ронять гейт
# ---------------------------------------------------------------------------


class TestTrapCatchesDecorativeKey:
    """Доказательство, что мета-тест работает, а не проходит вхолостую."""

    def test_fictitious_top_level_key_is_caught(self, tmp_path: Path) -> None:
        """Неучтённый ключ верхнего уровня в политике → гейт краснеет."""
        custom = tmp_path / "decorative.yaml"
        custom.write_text(
            "name: decorative\n"
            "fail_on_severity: high\n"
            "severity_threshold: medium\n"
            "block_everything_on_friday: true\n",
            encoding="utf-8",
        )
        policy = PolicyLoader.load(str(custom))

        unknown = unregistered_policy_keys(policy)
        assert unknown == ["block_everything_on_friday"], (
            f"Ловушка не сработала: фиктивный ключ не обнаружен, получено {unknown}"
        )

    def test_fictitious_extra_rule_key_is_caught(self, tmp_path: Path) -> None:
        """Неучтённый ключ extra_rules → гейт краснеет."""
        custom = tmp_path / "decorative_extra.yaml"
        custom.write_text(
            "name: decorative_extra\n"
            "extra_rules:\n"
            "  no_external_urls: true\n"
            "  forbid_models_larger_than_the_moon: true\n",
            encoding="utf-8",
        )
        policy = PolicyLoader.load(str(custom))

        unknown = unregistered_extra_rule_keys(policy)
        assert unknown == ["forbid_models_larger_than_the_moon"], (
            f"Ловушка не сработала для extra_rules, получено {unknown}"
        )

    def test_clean_policy_passes_the_trap(self) -> None:
        """Обратная сторона ловушки: на честной политике гейт зелёный."""
        policy = PolicyLoader.load("banking")
        assert unregistered_policy_keys(policy) == []
        assert unregistered_extra_rule_keys(policy) == []

    def test_broken_application_point_is_caught(self) -> None:
        """Ссылка в никуда роняет резолвер — значит, гейт не формальный."""
        with pytest.raises(AssertionError, match="не найдена"):
            _resolve("poison_check.policies:no_such_function_at_all")


# ---------------------------------------------------------------------------
# 5. severity_threshold — активный ключ реестра (результат Части 1)
# ---------------------------------------------------------------------------


class TestSeverityThresholdRegistered:
    """Регрессия исходного дефекта: ключ обязан быть применяемым."""

    def test_severity_threshold_is_enforced(self) -> None:
        """severity_threshold зарегистрирован со статусом ENFORCED."""
        spec = policy_key_specs()["severity_threshold"]
        assert spec.status is PolicyKeyStatus.ENFORCED, (
            f"severity_threshold должен быть ENFORCED, получено {spec.status}"
        )

    def test_severity_threshold_application_points_resolve(self) -> None:
        """Заявленные точки применения severity_threshold существуют."""
        spec = policy_key_specs()["severity_threshold"]
        for ref in spec.applied_by:
            _resolve(ref)
        assert any("output" in ref for ref in spec.applied_by), (
            "severity_threshold обязан применяться в слое Output — "
            "порог влияет на отчёт, а не на сканирование"
        )

    def test_fail_on_severity_is_separate_key(self) -> None:
        """severity_threshold и fail_on_severity — разные ключи с разными ролями."""
        specs = policy_key_specs()
        assert specs["severity_threshold"].applied_by != specs[
            "fail_on_severity"
        ].applied_by, (
            "Если бы точки применения совпадали, один из ключей был бы лишним"
        )


# ---------------------------------------------------------------------------
# 6. DESCRIPTIVE-ключи действительно показываются пользователю
# ---------------------------------------------------------------------------


class TestDescriptiveKeysAreShown:
    """``name`` и ``description`` не украшение YAML — их видно в CLI."""

    def test_doctor_lists_policy_names(self) -> None:
        """`poison-check doctor` перечисляет имена встроенных политик."""
        from poison_check.cli import app  # noqa: PLC0415

        result = runner.invoke(app, ["doctor"])
        for name in _BUILTIN_NAMES:
            assert name in result.output, (
                f"doctor не показал политику {name!r}:\n{result.output}"
            )

    def test_doctor_lists_policy_descriptions(self) -> None:
        """`poison-check doctor` показывает description каждой политики."""
        from poison_check.cli import app  # noqa: PLC0415

        result = runner.invoke(app, ["doctor"])
        for name in _BUILTIN_NAMES:
            description = str(_raw_policy(name).get("description", ""))
            assert description, f"У политики {name} пустой description"
            # Хвост описания достаточно уникален и не страдает от переносов Rich.
            fragment = description.split("—")[0].strip()[:20]
            assert fragment in result.output, (
                f"doctor не показал описание политики {name!r} "
                f"(искали {fragment!r}):\n{result.output}"
            )

    def test_specs_agree_with_registry_lookup(self) -> None:
        """policy_key_specs() и extra_rule_key_specs() отдают тот же набор."""
        assert set(policy_key_specs()) == {spec.key for spec in POLICY_KEYS}
        assert set(extra_rule_key_specs()) == {spec.key for spec in EXTRA_RULE_KEYS}

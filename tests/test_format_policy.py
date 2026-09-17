"""Тесты политики формата: MLS-FMT-001 и MLS-FMT-002.

Закрывают активацию двух ключей ``extra_rules``, которые до этого были
RESERVED — объявлены в banking/government/strict, но на поведение не влияли:

* ``strict_format_detection`` — формат определяется по СОДЕРЖИМОМУ, расхождение
  с расширением даёт MLS-FMT-001 (HIGH);
* ``require_safetensors`` — формат, исполняющий код при загрузке, даёт
  MLS-FMT-002 (HIGH) даже при полностью чистом содержимом.

Все вредоносные фикстуры собираются побайтово из опкодов; ``pickle.dumps`` на
злонамеренном объекте не используется, ``pickle.load`` не вызывается никогда.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from poison_check.core.result import Issue, MLContext, Severity
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.format_policy_detector import FormatPolicyDetector
from poison_check.policies import (
    PolicyKeyStatus,
    PolicyLoader,
    detector_kwargs_for,
    extra_rule_key_specs,
    policy_require_safetensors,
    policy_strict_format_detection,
)
from poison_check.scanner import Scanner

FIXTURES = Path(__file__).parent / "fixtures" / "malicious"

#: Политики, в которых оба ключа обязаны быть включены.
STRICT_POLICIES = ("banking", "government", "strict")

#: Коды, означающие «в содержимом найден вредонос». MLS-FMT-002 не должен
#: появляться по этой причине — он про формат, а не про находку.
_MALWARE_CODE_PREFIXES = ("MLS-PATTERN-", "MLS-CVE-", "MLS-PKL-", "MLS-ALW-")


# ---------------------------------------------------------------------------
# Фикстуры: ручная сборка байт
# ---------------------------------------------------------------------------


def _clean_pickle_bytes() -> bytes:
    """Валидный безвредный pickle: список из двух чисел, без глобалов.

    PROTO 2, EMPTY_LIST, MARK, BININT1 1, BININT1 2, APPENDS, STOP.
    """
    return b"\x80\x02]q\x00(K\x01K\x02e."


def _os_system_pickle_bytes(command: bytes = b"id") -> bytes:
    """Pickle-поток ``os.system(command)``, собранный из опкодов.

    PROTO 2, GLOBAL os system, SHORT_BINSTRING, TUPLE1, REDUCE, STOP.
    """
    return (
        b"\x80\x02"
        + b"c" + b"os\nsystem\n"
        + b"U" + bytes([len(command)]) + command
        + b"\x85"
        + b"R"
        + b"."
    )


def _safetensors_bytes() -> bytes:
    """Минимальный валидный safetensors: заголовок + один тензор F32."""
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x80\x3f"


def _gguf_bytes() -> bytes:
    """Минимальный валидный GGUF: magic, версия 3, ноль тензоров и ноль KV."""
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    """Пишет фикстуру во временный каталог теста."""
    target = tmp_path / name
    target.write_bytes(data)
    return target


def _codes(path: Path, policy: str) -> list[str]:
    """Возвращает коды находок по файлу при указанной политике."""
    result = Scanner(policy=policy).scan(path)
    return [issue.code for issue in result.results_per_file[path].issues]


def _issues(path: Path, policy: str) -> list[Issue]:
    """Возвращает находки по файлу при указанной политике."""
    result = Scanner(policy=policy).scan(path)
    return result.results_per_file[path].issues


def _custom_policy(tmp_path: Path, extra_rules: str) -> Path:
    """Создаёт пользовательскую политику с заданной секцией extra_rules."""
    target = tmp_path / "custom_policy.yaml"
    target.write_text(
        "name: custom\n"
        "description: 'политика для теста'\n"
        "severity_threshold: info\n"
        "fail_on_severity: medium\n"
        f"{extra_rules}",
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# 1. strict_format_detection → MLS-FMT-001
# ---------------------------------------------------------------------------


class TestStrictFormatDetection:
    """Расхождение «расширение ↔ фактический формат» становится находкой."""

    def test_pickle_disguised_as_safetensors_flagged(self, tmp_path: Path) -> None:
        """Pickle с именем .safetensors под banking → MLS-FMT-001 (HIGH)."""
        target = _write(tmp_path, "model.safetensors", _clean_pickle_bytes())

        found = [i for i in _issues(target, "banking") if i.code == "MLS-FMT-001"]

        assert len(found) == 1, (
            f"Ожидался ровно один MLS-FMT-001, получено: "
            f"{[i.code for i in _issues(target, 'banking')]}"
        )
        assert found[0].severity is Severity.HIGH

    def test_message_carries_raw_extension_format_and_basis(
        self, tmp_path: Path
    ) -> None:
        """В message — сырое расширение, фактический формат и признак детекции."""
        target = _write(tmp_path, "model.safetensors", _clean_pickle_bytes())

        issue = next(i for i in _issues(target, "banking") if i.code == "MLS-FMT-001")

        assert ".safetensors" in issue.message, issue.message
        assert "pickle" in issue.message, issue.message
        assert "PROTO" in issue.message, (
            f"Признак определения формата (magic/структура) не попал в message: "
            f"{issue.message}"
        )
        assert issue.details["declared_ext"] == ".safetensors"
        assert issue.details["detected_format"] == "pickle"
        assert issue.details["detection_basis"]

    def test_format_taken_from_content_not_from_name(self, tmp_path: Path) -> None:
        """Обратная подмена: safetensors с именем .pkl определяется по содержимому.

        Если бы формат брался из расширения, файл считался бы pickle и
        расхождения не было бы вовсе.
        """
        target = _write(tmp_path, "model.pkl", _safetensors_bytes())

        issue = next(
            (i for i in _issues(target, "banking") if i.code == "MLS-FMT-001"), None
        )

        assert issue is not None, "Обратная подмена формата не обнаружена"
        assert issue.details["declared_ext"] == ".pkl"
        assert issue.details["detected_format"] == "safetensors", (
            "Формат определён по имени файла, а не по содержимому"
        )

    def test_mismatch_not_flagged_when_disabled(self, tmp_path: Path) -> None:
        """default: расхождение не эскалируется в отдельную находку."""
        target = _write(tmp_path, "model.safetensors", _clean_pickle_bytes())

        assert "MLS-FMT-001" not in _codes(target, "default")

    def test_scanner_choice_unchanged_when_disabled(self, tmp_path: Path) -> None:
        """Выключенный ключ не меняет выбор сканера — регрессия логики подбора."""
        target = _write(tmp_path, "model.safetensors", _clean_pickle_bytes())

        result = Scanner(policy="default").scan(target)

        assert result.results_per_file[target].scanner_name == "safetensors", (
            "Сканер по-прежнему выбирается по расширению; ключ политики влияет "
            "только на отчёт"
        )

    def test_disguised_payload_fixture_is_regression_covered(self) -> None:
        """Фикстура «os.system внутри файла .safetensors» → MLS-FMT-001."""
        path = FIXTURES / "disguise_pickle_as_safetensors.safetensors"
        if not path.exists():
            pytest.skip(f"Фикстура не найдена: {path}")

        result = Scanner(policy="banking").scan(path)
        codes = {i.code for i in result.results_per_file[path].issues}

        assert "MLS-FMT-001" in codes, (
            f"Подмена формата на реальной фикстуре не обнаружена: {codes}"
        )

    def test_reverse_disguise_fixture_is_regression_covered(self) -> None:
        """Фикстура «safetensors внутри файла .pkl» → MLS-FMT-001."""
        path = FIXTURES / "disguise_safetensors_as_pickle.pkl"
        if not path.exists():
            pytest.skip(f"Фикстура не найдена: {path}")

        result = Scanner(policy="banking").scan(path)
        codes = {i.code for i in result.results_per_file[path].issues}

        assert "MLS-FMT-001" in codes, (
            f"Обратная подмена на реальной фикстуре не обнаружена: {codes}"
        )


# ---------------------------------------------------------------------------
# 2. require_safetensors → MLS-FMT-002
# ---------------------------------------------------------------------------


class TestRequireSafetensors:
    """Формат, исполняющий код при загрузке, недопустим в строгом контуре."""

    def test_clean_pickle_flagged_when_enabled(self, tmp_path: Path) -> None:
        """Чистый pickle под banking → MLS-FMT-002 (HIGH)."""
        target = _write(tmp_path, "clean.pkl", _clean_pickle_bytes())

        found = [i for i in _issues(target, "banking") if i.code == "MLS-FMT-002"]

        assert len(found) == 1, f"Ожидался MLS-FMT-002, получено: {_codes(target, 'banking')}"
        assert found[0].severity is Severity.HIGH

    def test_clean_pickle_flagged_not_because_of_malware(self, tmp_path: Path) -> None:
        """MLS-FMT-002 на чистом pickle не сопровождается находками по содержимому."""
        target = _write(tmp_path, "clean.pkl", _clean_pickle_bytes())

        codes = _codes(target, "banking")

        assert codes == ["MLS-FMT-002"], (
            f"Ожидалась ровно одна находка — про формат, а не про вредонос: {codes}"
        )
        assert not [
            c for c in codes if c.startswith(_MALWARE_CODE_PREFIXES)
        ], f"Находка выдана по вредоносу, а не по формату: {codes}"

    def test_remediation_points_to_safetensors(self, tmp_path: Path) -> None:
        """Рекомендация — пересохранить модель в safetensors."""
        target = _write(tmp_path, "clean.pkl", _clean_pickle_bytes())

        issue = next(i for i in _issues(target, "banking") if i.code == "MLS-FMT-002")

        assert issue.remediation is not None
        assert "safetensors" in issue.remediation.lower()
        assert issue.why is not None and "код" in issue.why.lower()

    def test_safetensors_clean_under_both_keys(self, tmp_path: Path) -> None:
        """Валидный safetensors под banking → ни одной находки."""
        target = _write(tmp_path, "model.safetensors", _safetensors_bytes())

        assert _codes(target, "banking") == []

    @pytest.mark.parametrize("policy", STRICT_POLICIES)
    def test_safetensors_clean_under_every_strict_policy(
        self, tmp_path: Path, policy: str
    ) -> None:
        """Безопасный формат не даёт ложных срабатываний ни в одной строгой политике."""
        target = _write(tmp_path, "model.safetensors", _safetensors_bytes())

        assert _codes(target, policy) == []

    @pytest.mark.parametrize("policy", STRICT_POLICIES)
    def test_gguf_is_treated_as_safe_format(self, tmp_path: Path, policy: str) -> None:
        """GGUF не десериализует объекты — MLS-FMT-002 по нему не эмитится."""
        target = _write(tmp_path, "model.gguf", _gguf_bytes())

        assert "MLS-FMT-002" not in _codes(target, policy)

    def test_pickle_not_escalated_when_disabled(self, tmp_path: Path) -> None:
        """default: pickle сам по себе находкой не является."""
        target = _write(tmp_path, "clean.pkl", _clean_pickle_bytes())

        assert _codes(target, "default") == []

    def test_joblib_extension_is_code_bearing(self, tmp_path: Path) -> None:
        """Несжатый joblib — тот же pickle, значит формат code-bearing."""
        target = _write(tmp_path, "model.joblib", _clean_pickle_bytes())

        assert "MLS-FMT-002" in _codes(target, "banking")


# ---------------------------------------------------------------------------
# 3. Комбинация: находка по формату не подавляет находку по содержимому
# ---------------------------------------------------------------------------


class TestFormatAndContentFindingsCoexist:
    """Формат и содержимое — два независимых утверждения, оба доезжают."""

    def test_real_payload_and_format_issue_both_present(self, tmp_path: Path) -> None:
        """pickle с os.system под banking → и MLS-PATTERN-OS-SYSTEM, и MLS-FMT-002."""
        target = _write(tmp_path, "payload.pkl", _os_system_pickle_bytes())

        issues = _issues(target, "banking")
        by_code = {i.code: i for i in issues}

        assert "MLS-PATTERN-OS-SYSTEM" in by_code, (
            f"Находка по содержимому потеряна: {sorted(by_code)}"
        )
        assert "MLS-FMT-002" in by_code, (
            f"Находка по формату потеряна: {sorted(by_code)}"
        )
        assert by_code["MLS-PATTERN-OS-SYSTEM"].severity is Severity.CRITICAL
        assert by_code["MLS-FMT-002"].severity is Severity.HIGH

    def test_format_issue_survives_severity_threshold(self, tmp_path: Path) -> None:
        """HIGH-находка политики формата не уходит под порог внимания.

        MLS-FMT-* не входит в список «неподавляемых» (файл при них проверен
        полностью), но HIGH не понижается ни при каком пороге.
        """
        from poison_check.output.severity_threshold import is_below_threshold

        target = _write(tmp_path, "clean.pkl", _clean_pickle_bytes())
        issue = next(i for i in _issues(target, "banking") if i.code == "MLS-FMT-002")

        assert not is_below_threshold(issue, Severity.CRITICAL)


# ---------------------------------------------------------------------------
# 4. Граница: неопознанный формат — это не подмена формата
# ---------------------------------------------------------------------------


class TestUnknownFormatBoundary:
    """«Формат не определён» и «формат не тот» — разные вещи."""

    def test_unknown_format_is_not_mismatch(self, tmp_path: Path) -> None:
        """Мусор с незнакомым расширением под strict → ни одной MLS-FMT-*."""
        target = _write(tmp_path, "weights.xyz", b"\x01\x02\x03\x04garbage")

        codes = _codes(target, "strict")

        assert not [c for c in codes if c.startswith("MLS-FMT-")], (
            f"Неопознанный формат ошибочно принят за подмену: {codes}"
        )

    def test_unknown_format_still_reported_as_file_error(self, tmp_path: Path) -> None:
        """Файл не теряется: неподдерживаемый формат едет в FileResult.error."""
        target = _write(tmp_path, "weights.xyz", b"\x01\x02\x03\x04garbage")

        result = Scanner(policy="strict").scan(target)

        assert result.results_per_file[target].error is not None

    def test_unparsable_pickle_stays_parse_error(self, tmp_path: Path) -> None:
        """Битый pickle под strict → MLS-PARSE-001, а не MLS-FMT-001.

        Расширение и формат по содержимому совпадают (pickle), сломан сам поток
        опкодов — это зона общего детектора ошибок разбора.
        """
        target = _write(tmp_path, "broken.pkl", b"\x80\x02\xff\xff\xff")

        codes = _codes(target, "strict")

        assert "MLS-PARSE-001" in codes, codes
        assert "MLS-FMT-001" not in codes, codes

    def test_unsupported_file_adds_no_new_noise(self, tmp_path: Path) -> None:
        """README рядом с моделью под strict не даёт ни одной находки."""
        target = _write(tmp_path, "README.md", b"# model card\n")

        assert _codes(target, "strict") == []


# ---------------------------------------------------------------------------
# 5. Регресс: дефолтная политика ведёт себя как раньше
# ---------------------------------------------------------------------------


class TestDefaultPolicyRegression:
    """В default ни один из двух ключей не включён и находок не добавляет."""

    @pytest.mark.parametrize(
        "name,data",
        [
            ("clean.pkl", _clean_pickle_bytes()),
            ("model.safetensors", _safetensors_bytes()),
            ("model.gguf", _gguf_bytes()),
            ("model.joblib", _clean_pickle_bytes()),
        ],
    )
    def test_no_format_issues_under_default(
        self, tmp_path: Path, name: str, data: bytes
    ) -> None:
        """Ни MLS-FMT-001, ни MLS-FMT-002 в дефолтной политике."""
        target = _write(tmp_path, name, data)

        codes = _codes(target, "default")

        assert not [c for c in codes if c.startswith("MLS-FMT-")], codes

    def test_default_policy_has_both_keys_disabled(self) -> None:
        """Реестр и YAML согласованы: в default оба ключа выключены."""
        policy = PolicyLoader.load("default")

        assert policy_strict_format_detection(policy) is False
        assert policy_require_safetensors(policy) is False

    def test_policy_without_extra_rules_keeps_keys_off(self, tmp_path: Path) -> None:
        """Политика вообще без секции extra_rules — оба ключа выключены."""
        custom = _custom_policy(tmp_path, "compliance: []\n")
        policy = PolicyLoader.load(str(custom))

        assert policy_strict_format_detection(policy) is False
        assert policy_require_safetensors(policy) is False


# ---------------------------------------------------------------------------
# 6. Реестр ключей: оба ключа ENFORCED и согласованы с YAML
# ---------------------------------------------------------------------------


class TestPolicyKeyRegistry:
    """Ключи переведены RESERVED → ENFORCED, YAML соответствует реестру."""

    @pytest.mark.parametrize(
        "key", ["strict_format_detection", "require_safetensors"]
    )
    def test_key_is_enforced(self, key: str) -> None:
        """Оба ключа объявлены применяемыми."""
        spec = extra_rule_key_specs()[key]

        assert spec.status is PolicyKeyStatus.ENFORCED, (
            f"{key}: ожидался ENFORCED, получено {spec.status}"
        )
        assert spec.applied_by, f"{key}: не указана точка применения"
        assert spec.proven_by, f"{key}: не указан доказывающий тест"

    @pytest.mark.parametrize("policy_name", STRICT_POLICIES)
    def test_strict_policies_enable_both_keys(self, policy_name: str) -> None:
        """banking / government / strict включают оба ключа."""
        policy = PolicyLoader.load(policy_name)

        assert policy_strict_format_detection(policy) is True, policy_name
        assert policy_require_safetensors(policy) is True, policy_name

    @pytest.mark.parametrize("policy_name", STRICT_POLICIES)
    def test_detector_kwargs_carry_policy_flags(self, policy_name: str) -> None:
        """Флаги едут в детектор существующим механизмом detector_kwargs_for."""
        policy = PolicyLoader.load(policy_name)

        kwargs = detector_kwargs_for(policy, "format_policy")

        assert kwargs == {
            "strict_format_detection": True,
            "require_safetensors": True,
        }

    def test_detector_kwargs_empty_flags_for_default(self) -> None:
        """В default тот же механизм отдаёт выключенные флаги."""
        kwargs = detector_kwargs_for(PolicyLoader.load("default"), "format_policy")

        assert kwargs == {
            "strict_format_detection": False,
            "require_safetensors": False,
        }

    @pytest.mark.parametrize("policy_name", ("default", *STRICT_POLICIES))
    def test_detector_enabled_in_every_builtin_policy(self, policy_name: str) -> None:
        """Детектор включён во всех встроенных политиках — иначе он мёртв."""
        policy = PolicyLoader.load(policy_name)

        assert "format_policy" in (policy.get("enabled_detectors") or [])


# ---------------------------------------------------------------------------
# 7. Детектор в изоляции
# ---------------------------------------------------------------------------


class TestDetectorInIsolation:
    """Детектор молчит, когда его не просят и когда фактов нет."""

    @staticmethod
    def _raw(metadata: dict[str, object] | None) -> RawScanData:
        """Собирает минимальный RawScanData для прямого вызова детектора."""
        return RawScanData(
            file_path=Path("model.pkl"),
            file_hash={},
            file_size=0,
            scanner_name="pickle",
            metadata=metadata,  # type: ignore[arg-type]
        )

    def test_both_keys_off_gives_no_issues(self) -> None:
        """Оба ключа выключены → детектор не читает даже факты."""
        detector = FormatPolicyDetector()
        raw = self._raw({"_format_detected": "pickle", "_format_mismatch": True})

        assert detector.analyze(raw, MLContext("unknown", 0.0, [])) == []

    def test_missing_facts_gives_no_issues(self) -> None:
        """Факты не приложены (сканер вызван в обход пайплайна) → тишина."""
        detector = FormatPolicyDetector(
            strict_format_detection=True, require_safetensors=True
        )

        assert detector.analyze(self._raw(None), MLContext("unknown", 0.0, [])) == []

    def test_detector_defaults_are_off(self) -> None:
        """Конструктор без аргументов = поведение дефолтной политики."""
        detector = FormatPolicyDetector()

        assert detector._strict_format_detection is False
        assert detector._require_safetensors is False

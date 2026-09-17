"""Тесты порога внимания ``severity_threshold``.

Семантика (см. :mod:`poison_check.output.severity_threshold`): порог ВНИМАНИЯ,
не порог отображения. Находка ниже порога остаётся в отчёте во всех четырёх
форматах и лишь помечается; действие (провал гейта) определяется независимым
``fail_on_severity``.

Файл покрывает четыре группы утверждений:

* порог реально влияет на вывод — единообразно в console/JSON/SARIF/SBOM;
* взаимодействие ``severity_threshold`` × ``fail_on_severity`` (матрица);
* ИНВАРИАНТ: CRITICAL/HIGH и факты уровня файла (parse-error, бомба,
  непроверенный файл) не уходят под порог ни при каком его значении;
* регресс: результат без порога форматируется ровно как до появления ключа.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from poison_check.core.result import (
    Confidence,
    FileResult,
    Issue,
    PolicyThresholds,
    ScanResult,
    Severity,
    Summary,
)
from poison_check.i18n.loader import I18n
from poison_check.output.console import ConsoleFormatter
from poison_check.output.json_format import JsonFormatter
from poison_check.output.sarif import SarifFormatter
from poison_check.output.sbom import SbomFormatter
from poison_check.output.severity_threshold import (
    NEVER_DEMOTED_SEVERITY,
    UNSUPPRESSIBLE_CODES,
    count_below_threshold,
    is_below_threshold,
    partition_issues,
    threshold_of,
)
from poison_check.policies import (
    PolicyLoader,
    policy_fail_on_severity,
    policy_severity_threshold,
    policy_thresholds,
)

_FILE = Path("model.pkl")

_ALL_SEVERITIES = (
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
)


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------


def _issue(
    severity: Severity,
    code: str = "MLS-ALW-001",
    message: str | None = None,
) -> Issue:
    """Создаёт Issue заданного уровня с уникальным сообщением."""
    return Issue(
        code=code,
        severity=severity,
        confidence=Confidence.MEDIUM,
        message=message or f"Находка уровня {severity.value} ({code})",
        location=f"{_FILE}#pkg.mod.{severity.value}",
    )


def _scan_result(
    issues: list[Issue],
    severity_threshold: Severity | None = None,
    fail_on_severity: Severity | None = Severity.CRITICAL,
    with_thresholds: bool = True,
    error: str | None = None,
) -> ScanResult:
    """Собирает ScanResult с одним файлом и заданными порогами.

    :param with_thresholds: ``False`` — ``policy_thresholds`` остаётся ``None``
        (результат, собранный вручную; поведение «как до появления порога»).
    """
    file_result = FileResult(
        file_path=_FILE,
        scanner_name="pickle",
        issues=issues,
        duration_ms=1.0,
        error=error,
    )
    return ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(tz=timezone.utc),
        duration_ms=1.0,
        scanned_paths=[_FILE],
        policy="test",
        results_per_file={_FILE: file_result},
        summary=Summary(),
        policy_thresholds=(
            PolicyThresholds(
                severity_threshold=severity_threshold,
                fail_on_severity=fail_on_severity,
            )
            if with_thresholds
            else None
        ),
    )


def _console_text(result: ScanResult) -> str:
    """Рендерит результат консольным форматтером и возвращает текст."""
    console = Console(file=io.StringIO(), width=200, record=True, force_terminal=False)
    formatter = ConsoleFormatter(
        console=console,
        i18n=I18n("ru"),
        severity_threshold=threshold_of(result),
        no_emoji=True,
    )
    for file_result in result.results_per_file.values():
        formatter.format_file_result(file_result)
    formatter.format_summary(result)
    return console.export_text()


def _json_doc(result: ScanResult) -> dict[str, Any]:
    """Рендерит результат JSON-форматтером и возвращает разобранный документ."""
    return json.loads(JsonFormatter().format(result))  # type: ignore[no-any-return]


def _sarif_doc(result: ScanResult) -> dict[str, Any]:
    """Рендерит результат SARIF-форматтером."""
    return json.loads(SarifFormatter().format(result))  # type: ignore[no-any-return]


def _sbom_doc(result: ScanResult) -> dict[str, Any]:
    """Рендерит результат SBOM-форматтером."""
    return json.loads(SbomFormatter().format(result))  # type: ignore[no-any-return]


def _json_issue(doc: dict[str, Any], code: str) -> dict[str, Any]:
    """Находит issue по коду в JSON-отчёте."""
    for file_entry in doc["results"]:
        for issue in file_entry["issues"]:
            if issue["code"] == code:
                return issue  # type: ignore[no-any-return]
    raise AssertionError(f"Issue {code} потеряна в JSON-отчёте")


def _sarif_result(doc: dict[str, Any], code: str) -> dict[str, Any]:
    """Находит SARIF result по ruleId."""
    for res in doc["runs"][0]["results"]:
        if res["ruleId"] == code:
            return res  # type: ignore[no-any-return]
    raise AssertionError(f"Issue {code} потеряна в SARIF-отчёте")


def _sbom_vuln(doc: dict[str, Any], code: str) -> dict[str, Any]:
    """Находит уязвимость SBOM по свойству poison-check:code."""
    for vuln in doc["vulnerabilities"]:
        for prop in vuln["properties"]:
            if prop["name"] == "poison-check:code" and prop["value"] == code:
                return vuln  # type: ignore[no-any-return]
    raise AssertionError(f"Issue {code} потеряна в SBOM")


def _sbom_property(entry: dict[str, Any], name: str) -> str | None:
    """Достаёт значение свойства CycloneDX по имени."""
    for prop in entry.get("properties", []):
        if prop["name"] == name:
            return str(prop["value"])
    return None


# ---------------------------------------------------------------------------
# 1. Семантика: что именно означает порог
# ---------------------------------------------------------------------------


class TestThresholdSemantics:
    """Порог помечает находки, но никогда их не удаляет."""

    def test_below_threshold_is_marked(self) -> None:
        """Находка ниже порога распознаётся как подпороговая."""
        assert is_below_threshold(_issue(Severity.LOW), Severity.MEDIUM) is True

    def test_at_threshold_is_primary(self) -> None:
        """Находка ровно на пороге остаётся основной (порог включительный)."""
        assert is_below_threshold(_issue(Severity.MEDIUM), Severity.MEDIUM) is False

    def test_no_threshold_marks_nothing(self) -> None:
        """Порог не задан → ни одна находка не подпороговая."""
        for severity in _ALL_SEVERITIES:
            assert is_below_threshold(_issue(severity), None) is False

    def test_partition_loses_nothing(self) -> None:
        """Сумма основных и подпороговых равна исходному списку.

        Это инвариант «сигнал не теряется» в исполняемом виде.
        """
        issues = [_issue(sev, code=f"MLS-ALW-00{i}") for i, sev in enumerate(_ALL_SEVERITIES)]
        for threshold in (*_ALL_SEVERITIES, None):
            primary, below = partition_issues(issues, threshold)
            assert len(primary) + len(below) == len(issues)
            assert {id(i) for i in primary + below} == {id(i) for i in issues}

    def test_count_below_threshold_counts_whole_run(self) -> None:
        """Счётчик подпороговых считает находки по всем файлам прогона."""
        result = _scan_result(
            [_issue(Severity.LOW), _issue(Severity.INFO, code="MLS-ALW-002")],
            severity_threshold=Severity.MEDIUM,
        )
        assert count_below_threshold(result, Severity.MEDIUM) == 2


# ---------------------------------------------------------------------------
# 2. Порог влияет на вывод во ВСЕХ форматах
# ---------------------------------------------------------------------------


class TestThresholdInAllFormats:
    """Один и тот же порог даёт согласованную пометку в 4 форматах."""

    _CODE_BELOW = "MLS-ALW-001"
    _CODE_PRIMARY = "MLS-PKL-002"

    def _result(self) -> ScanResult:
        """LOW-находка ниже порога MEDIUM + HIGH-находка над порогом."""
        return _scan_result(
            [
                _issue(Severity.LOW, code=self._CODE_BELOW),
                _issue(Severity.HIGH, code=self._CODE_PRIMARY),
            ],
            severity_threshold=Severity.MEDIUM,
            fail_on_severity=Severity.CRITICAL,
        )

    def test_below_threshold_marked_in_json(self) -> None:
        """JSON: below_threshold=true у подпороговой, false у основной."""
        doc = _json_doc(self._result())
        assert _json_issue(doc, self._CODE_BELOW)["below_threshold"] is True
        assert _json_issue(doc, self._CODE_PRIMARY)["below_threshold"] is False
        assert doc["summary"]["below_threshold"] == 1
        assert doc["tool"]["thresholds"]["severity_threshold"] == "medium"
        assert doc["tool"]["thresholds"]["fail_on_severity"] == "critical"

    def test_below_threshold_marked_in_sarif(self) -> None:
        """SARIF: подпороговый result помечен suppressions, но остаётся в results."""
        doc = _sarif_doc(self._result())
        below = _sarif_result(doc, self._CODE_BELOW)
        primary = _sarif_result(doc, self._CODE_PRIMARY)

        assert below["properties"]["below_threshold"] is True
        assert below["suppressions"][0]["kind"] == "external"
        assert below["suppressions"][0]["justification"]
        assert primary["properties"]["below_threshold"] is False
        assert "suppressions" not in primary
        # level не понижается: подпороговость — свойство подачи, не угрозы
        assert below["level"] == "warning"

    def test_below_threshold_marked_in_sbom(self) -> None:
        """SBOM: подпороговая уязвимость получает analysis.state=in_triage."""
        doc = _sbom_doc(self._result())
        below = _sbom_vuln(doc, self._CODE_BELOW)
        primary = _sbom_vuln(doc, self._CODE_PRIMARY)

        assert _sbom_property(below, "poison-check:below_threshold") == "true"
        assert below["analysis"]["state"] == "in_triage"
        assert below["analysis"]["detail"]
        assert _sbom_property(primary, "poison-check:below_threshold") == "false"
        assert "analysis" not in primary
        # ratings не меняются — severity уязвимости не зависит от подачи отчёта
        assert below["ratings"][0]["severity"] == "low"
        assert (
            _sbom_property(doc["metadata"], "poison-check:severity_threshold")
            == "medium"
        )
        assert (
            _sbom_property(doc["metadata"], "poison-check:below_threshold_count") == "1"
        )

    def test_below_threshold_separated_in_console(self) -> None:
        """Console: подпороговая находка выведена в отдельном блоке."""
        text = _console_text(self._result())
        assert "Ниже порога внимания" in text
        assert self._CODE_BELOW in text, "подпороговая находка исчезла из консоли"
        assert self._CODE_PRIMARY in text
        # Подпороговый блок идёт ПОСЛЕ основных находок
        assert text.index(self._CODE_PRIMARY) < text.index("Ниже порога внимания")

    @pytest.mark.parametrize("fmt", ["console", "json", "sarif", "sbom"])
    def test_no_format_drops_below_threshold_issue(self, fmt: str) -> None:
        """Ни один из четырёх форматов не теряет подпороговую находку."""
        result = self._result()
        rendered = {
            "console": lambda: _console_text(result),
            "json": lambda: JsonFormatter().format(result),
            "sarif": lambda: SarifFormatter().format(result),
            "sbom": lambda: SbomFormatter().format(result),
        }[fmt]()
        assert self._CODE_BELOW in rendered, (
            f"формат {fmt} потерял подпороговую находку — это скрытие сигнала"
        )

    @pytest.mark.parametrize(
        ("threshold", "expected_below"),
        [
            (Severity.INFO, 0),
            (Severity.LOW, 1),
            (Severity.MEDIUM, 2),
            (Severity.HIGH, 3),
            (Severity.CRITICAL, 3),
        ],
    )
    def test_threshold_value_changes_marking(
        self, threshold: Severity, expected_below: int
    ) -> None:
        """Разные значения порога дают разное число подпороговых находок.

        При threshold=critical счётчик не растёт до 4: HIGH под порог не уходит
        (см. NEVER_DEMOTED_SEVERITY).
        """
        issues = [
            _issue(Severity.INFO, code="MLS-ALW-001"),
            _issue(Severity.LOW, code="MLS-ALW-002"),
            _issue(Severity.MEDIUM, code="MLS-NPY-001"),
            _issue(Severity.HIGH, code="MLS-PKL-002"),
        ]
        result = _scan_result(issues, severity_threshold=threshold)
        doc = _json_doc(result)
        assert doc["summary"]["below_threshold"] == expected_below


# ---------------------------------------------------------------------------
# 3. Матрица severity_threshold × fail_on_severity
# ---------------------------------------------------------------------------


class TestThresholdVsFailOnSeverity:
    """Порог внимания и порог действия независимы — и это проверяется."""

    @pytest.mark.parametrize(
        ("severity", "threshold", "fail_on", "expect_below", "expect_exit"),
        [
            # threshold=medium, fail_on=high — типовая связка banking
            (Severity.LOW, Severity.MEDIUM, Severity.HIGH, True, 0),
            (Severity.MEDIUM, Severity.MEDIUM, Severity.HIGH, False, 0),
            (Severity.HIGH, Severity.MEDIUM, Severity.HIGH, False, 1),
            (Severity.CRITICAL, Severity.MEDIUM, Severity.HIGH, False, 1),
            # порог действия НИЖЕ порога внимания: находка помечена подпороговой,
            # но гейт всё равно валит — «помечена» ≠ «проигнорирована»
            (Severity.MEDIUM, Severity.HIGH, Severity.MEDIUM, True, 1),
            (Severity.LOW, Severity.HIGH, Severity.LOW, True, 1),
            # порога внимания нет — ничего не помечается, гейт как раньше
            (Severity.LOW, None, Severity.CRITICAL, False, 0),
        ],
    )
    def test_matrix(
        self,
        severity: Severity,
        threshold: Severity | None,
        fail_on: Severity,
        expect_below: bool,
        expect_exit: int,
    ) -> None:
        """Что видно и что валит гейт для каждой комбинации порогов."""
        from poison_check.cli import _compute_exit_code  # noqa: PLC0415

        issue = _issue(severity)
        result = _scan_result([issue], severity_threshold=threshold, fail_on_severity=fail_on)

        assert is_below_threshold(issue, threshold) is expect_below
        # Находка присутствует в отчёте независимо от пометки
        assert _json_issue(_json_doc(result), issue.code)["below_threshold"] is expect_below

        policy: dict[str, object] = {"fail_on_severity": fail_on.value}
        assert _compute_exit_code(result, policy) == expect_exit

    def test_threshold_does_not_change_exit_code(self) -> None:
        """При фиксированном fail_on_severity любой порог даёт тот же exit code."""
        from poison_check.cli import _compute_exit_code  # noqa: PLC0415

        policy: dict[str, object] = {"fail_on_severity": "medium"}
        codes = set()
        for threshold in (*_ALL_SEVERITIES, None):
            result = _scan_result(
                [_issue(Severity.MEDIUM)],
                severity_threshold=threshold,
                fail_on_severity=Severity.MEDIUM,
            )
            codes.add(_compute_exit_code(result, policy))
        assert codes == {1}, "severity_threshold не должен влиять на exit code"

    def test_legacy_exit_on_severity_still_gates(self) -> None:
        """Устаревший ключ exit_on_severity продолжает определять exit code."""
        from poison_check.cli import _compute_exit_code  # noqa: PLC0415

        result = _scan_result([_issue(Severity.HIGH)])
        assert policy_fail_on_severity({"exit_on_severity": "high"}) is Severity.HIGH
        assert _compute_exit_code(result, {"exit_on_severity": "high"}) == 1
        assert _compute_exit_code(result, {"exit_on_severity": "critical"}) == 0

    def test_fail_on_severity_wins_over_legacy_alias(self) -> None:
        """При обоих ключах приоритет у нового fail_on_severity."""
        policy = {"fail_on_severity": "critical", "exit_on_severity": "low"}
        assert policy_fail_on_severity(policy) is Severity.CRITICAL

    @pytest.mark.parametrize(
        ("name", "threshold", "fail_on"),
        [
            ("default", Severity.MEDIUM, Severity.CRITICAL),
            ("banking", Severity.LOW, Severity.HIGH),
            ("government", Severity.LOW, Severity.HIGH),
            ("strict", Severity.INFO, Severity.MEDIUM),
        ],
    )
    def test_builtin_policies_expose_both_thresholds(
        self, name: str, threshold: Severity, fail_on: Severity
    ) -> None:
        """Встроенные политики отдают оба порога через аксессоры."""
        policy = PolicyLoader.load(name)
        assert policy_severity_threshold(policy) is threshold
        assert policy_fail_on_severity(policy) is fail_on
        assert policy_thresholds(policy) == PolicyThresholds(threshold, fail_on)

    @pytest.mark.parametrize("value", ["", "  ", "нет", "42", None, 3, True])
    def test_invalid_threshold_value_is_ignored(self, value: Any) -> None:
        """Некорректное значение порога = «порога нет», а не чужой уровень."""
        assert policy_severity_threshold({"severity_threshold": value}) is None

    def test_threshold_value_is_case_insensitive(self) -> None:
        """`MEDIUM` и `medium` — один и тот же порог (YAML пишут люди)."""
        assert policy_severity_threshold({"severity_threshold": "MEDIUM"}) is (
            Severity.MEDIUM
        )


# ---------------------------------------------------------------------------
# 4. ИНВАРИАНТ: что порог не подавляет никогда
# ---------------------------------------------------------------------------


class TestNeverSuppressed:
    """CRITICAL/HIGH и факты уровня файла не уходят под порог ни при каком значении."""

    @pytest.mark.parametrize("severity", [Severity.CRITICAL, Severity.HIGH])
    @pytest.mark.parametrize("threshold", _ALL_SEVERITIES)
    def test_critical_and_high_never_below(
        self, severity: Severity, threshold: Severity
    ) -> None:
        """Даже threshold=critical не делает HIGH подпороговой."""
        assert is_below_threshold(_issue(severity), threshold) is False

    def test_never_demoted_boundary_is_high(self) -> None:
        """Граница «не понижаем» зафиксирована на HIGH."""
        assert NEVER_DEMOTED_SEVERITY is Severity.HIGH

    @pytest.mark.parametrize("code", sorted(UNSUPPRESSIBLE_CODES))
    @pytest.mark.parametrize("threshold", _ALL_SEVERITIES)
    def test_file_level_fact_codes_never_below(
        self, code: str, threshold: Severity
    ) -> None:
        """Факт уровня файла остаётся основным при любом пороге.

        Сюда входят parse-error (MLS-PARSE-001), бомбы (MLS-BOMB-001,
        MLS-CMP-*), непроверенный файл (MLS-KERAS-003, MLS-JOBLIB-001/002) и
        обрыв разбора (MLS-PKL-006). Их severity низкий, но они сообщают, что
        проверка НЕ состоялась — скрывать такое нельзя.
        """
        assert is_below_threshold(_issue(Severity.INFO, code=code), threshold) is False

    @pytest.mark.parametrize(
        "code",
        ["MLS-PARSE-001", "MLS-BOMB-001", "MLS-KERAS-003", "MLS-JOBLIB-001"],
    )
    def test_file_level_facts_stay_primary_in_all_formats(self, code: str) -> None:
        """Факты уровня файла не помечаются подпороговыми ни в одном формате."""
        result = _scan_result(
            [_issue(Severity.INFO, code=code)],
            severity_threshold=Severity.CRITICAL,
        )
        assert _json_issue(_json_doc(result), code)["below_threshold"] is False
        assert "suppressions" not in _sarif_result(_sarif_doc(result), code)
        assert "analysis" not in _sbom_vuln(_sbom_doc(result), code)
        text = _console_text(result)
        assert code in text
        assert "Ниже порога внимания" not in text

    def test_file_result_error_survives_maximum_threshold(self) -> None:
        """FileResult.error доезжает до вывода при самом высоком пороге.

        Ошибка разбора идёт отдельным каналом файловых фактов и порогу не
        подчиняется вовсе (см. output/file_level_facts.py).
        """
        result = _scan_result(
            [_issue(Severity.LOW)],
            severity_threshold=Severity.CRITICAL,
            error="genops: truncated opcode stream",
        )
        notifications = _sarif_doc(result)["runs"][0]["invocations"][0][
            "toolExecutionNotifications"
        ]
        assert any(
            "truncated opcode stream" in n["message"]["text"] for n in notifications
        )
        assert "truncated opcode stream" in _json_doc(result)["results"][0]["error"]
        assert "truncated opcode stream" in _console_text(result)

    def test_severity_counters_are_not_reduced_by_threshold(self) -> None:
        """Порог не вычитает находки из счётчиков severity в summary."""
        result = _scan_result(
            [_issue(Severity.LOW), _issue(Severity.INFO, code="MLS-ALW-002")],
            severity_threshold=Severity.CRITICAL,
        )
        summary = _json_doc(result)["summary"]
        assert summary["low"] == 1
        assert summary["info"] == 1
        assert summary["below_threshold"] == 2

    def test_sarif_result_count_is_independent_of_threshold(self) -> None:
        """Число SARIF result'ов не зависит от порога — метрики не искажаются."""
        issues = [_issue(sev, code=f"MLS-ALW-00{i}") for i, sev in enumerate(_ALL_SEVERITIES)]
        counts = {
            len(_sarif_doc(_scan_result(issues, severity_threshold=t))["runs"][0]["results"])
            for t in (*_ALL_SEVERITIES, None)
        }
        assert counts == {len(issues)}


# ---------------------------------------------------------------------------
# 5. Регресс: без порога всё как раньше
# ---------------------------------------------------------------------------


class TestNoThresholdRegression:
    """Результат без порогов форматируется ровно как до появления ключа."""

    def _result(self) -> ScanResult:
        return _scan_result(
            [_issue(Severity.LOW), _issue(Severity.CRITICAL, code="MLS-PKL-001")],
            with_thresholds=False,
        )

    def test_threshold_of_result_without_policy_thresholds_is_none(self) -> None:
        """ScanResult без policy_thresholds не имеет порога."""
        assert threshold_of(self._result()) is None

    def test_policy_without_key_has_no_threshold(self) -> None:
        """Политика без ключа severity_threshold порога не задаёт."""
        assert policy_severity_threshold({}) is None
        assert policy_thresholds({}) == PolicyThresholds(None, Severity.CRITICAL)

    def test_json_marks_nothing(self) -> None:
        """JSON: below_threshold=false у всех, счётчик нулевой, пороги null."""
        doc = _json_doc(self._result())
        assert all(
            issue["below_threshold"] is False
            for entry in doc["results"]
            for issue in entry["issues"]
        )
        assert doc["summary"]["below_threshold"] == 0
        assert doc["tool"]["thresholds"] == {
            "severity_threshold": None,
            "fail_on_severity": None,
        }

    def test_sarif_has_no_suppressions(self) -> None:
        """SARIF: ни одного подавленного результата, ключа порога нет."""
        doc = _sarif_doc(self._result())
        for res in doc["runs"][0]["results"]:
            assert "suppressions" not in res
            assert res["properties"]["below_threshold"] is False
            assert "severity_threshold" not in res["properties"]

    def test_sbom_has_no_analysis(self) -> None:
        """SBOM: ни одной уязвимости в состоянии in_triage, порога в metadata нет."""
        doc = _sbom_doc(self._result())
        for vuln in doc["vulnerabilities"]:
            assert "analysis" not in vuln
            assert _sbom_property(vuln, "poison-check:below_threshold") == "false"
        assert (
            _sbom_property(doc["metadata"], "poison-check:severity_threshold") is None
        )

    def test_console_has_no_threshold_section(self) -> None:
        """Console: блок «Ниже порога внимания» не печатается, находки на месте."""
        text = _console_text(self._result())
        assert "Ниже порога внимания" not in text
        assert "MLS-ALW-001" in text
        assert "MLS-PKL-001" in text


# ---------------------------------------------------------------------------
# 6. Сквозной путь: CLI и Python API применяют порог одинаково
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Порог доезжает до отчёта и через CLI, и через Python API."""

    def _make_pickle(self, tmp_path: Path) -> Path:
        """Создаёт безобидный pickle-файл (opcode-конструирование, без dumps)."""
        target = tmp_path / "clean.pkl"
        # PROTO 2, EMPTY_LIST, MARK, ..., STOP — валидный поток без глобалов.
        target.write_bytes(b"\x80\x02]q\x00(K\x01K\x02e.")
        return target

    def test_python_api_result_carries_policy_thresholds(self, tmp_path: Path) -> None:
        """Scanner кладёт пороги политики в ScanResult."""
        from poison_check.scanner import Scanner  # noqa: PLC0415

        result = Scanner(policy="strict").scan(self._make_pickle(tmp_path))
        assert result.policy_thresholds is not None
        assert result.policy_thresholds.severity_threshold is Severity.INFO
        assert result.policy_thresholds.fail_on_severity is Severity.MEDIUM

    def test_cli_json_report_contains_thresholds(self, tmp_path: Path) -> None:
        """CLI --format json печатает пороги активной политики."""
        from typer.testing import CliRunner  # noqa: PLC0415

        from poison_check.cli import app  # noqa: PLC0415

        target = self._make_pickle(tmp_path)
        out = tmp_path / "report.json"
        CliRunner().invoke(
            app,
            ["scan", str(target), "--policy", "banking", "--format", "json",
             "--output", str(out)],
        )
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["tool"]["thresholds"] == {
            "severity_threshold": "low",
            "fail_on_severity": "high",
        }

    def test_cli_and_api_agree_on_marking(self, tmp_path: Path) -> None:
        """Одна и та же политика даёт одинаковую пометку в CLI и в API."""
        from poison_check.cli import _load_policy  # noqa: PLC0415
        from poison_check.scanner import Scanner  # noqa: PLC0415

        api_result = Scanner(policy="strict").scan(self._make_pickle(tmp_path))
        cli_policy = _load_policy("strict")

        assert api_result.policy_thresholds == policy_thresholds(cli_policy)

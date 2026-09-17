"""Forensic-краевой случай: «payload + оборванный хвост».

Вредоносный pickle, у которого поток опкодов СНАЧАЛА содержит опасный глобал
(os.system + REDUCE → КРИТ), а ПОТОМ намеренно обрывается (genops падает на
битом хвосте). Комбинация «валидный payload + испорченный хвост» — более сильный
forensic-сигнал, чем просто payload.

MLS-PARSE-001 (LOW) при наличии globals ПОДАВЛЯЕТСЯ (верно — не шуметь дублями
поверх КРИТ). Но факт обрыва не должен исчезнуть: SARIF-форматтер строит вывод
ТОЛЬКО из Issues и игнорирует FileResult.error, поэтому факт обрыва протаскивается
в ``details`` уже эмитируемой находки (``parse_truncated`` / ``parse_error``) —
см. ``annotate_suppressed_parse_error``. Там он виден во всех форматах и
скоррелирован с самим payload.

Фикстура собрана ВРУЧНУЮ из байт (opcode-конструирование), без pickle.dumps.
Временные файлы — через tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Severity,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.detectors.parse_error_detector import (
    ParseErrorDetector,
    annotate_suppressed_parse_error,
)
from poison_check.output.json_format import JsonFormatter
from poison_check.output.sarif import SarifFormatter
from poison_check.scanner import Scanner

_CODE_PARSE = "MLS-PARSE-001"
_CTX = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])


def _payload_then_truncated() -> bytes:
    """Опкоды: GLOBAL os system + REDUCE (КРИТ), затем оборванный опкод в хвосте.

    Ручная сборка байт (без pickle.dumps живого объекта):
      PROTO 4
      GLOBAL 'os' 'system'                      -> глобал ('os','system') => КРИТ
      SHORT_BINUNICODE 'id' / TUPLE1 / REDUCE   -> reduce os.system('id')
      SHORT_BINUNICODE len=255, БЕЗ данных      -> genops бросает на хвосте
      (STOP отсутствует)
    """
    return (
        b"\x80\x04"          # PROTO 4
        b"cos\nsystem\n"     # GLOBAL os system
        b"\x8c\x02id"        # SHORT_BINUNICODE 'id'
        b"\x85"              # TUPLE1
        b"R"                 # REDUCE
        b"\x8c\xff"          # SHORT_BINUNICODE length=255, данных нет -> обрыв
    )


def _clean_truncated_no_payload() -> bytes:
    """Валидный старт (can_handle=True), но обрыв БЕЗ извлечённого payload.

    PROTO 4 + FRAME-опкод (0x95) без 8-байтной длины — genops падает сразу,
    globals не извлекаются.
    """
    return b"\x80\x04\x95\xff\xff"


def _codes(issues: list[Issue]) -> list[str]:
    return [i.code for i in issues]


def _threat_issue(code: str = "MLS-PATTERN-OS-SYSTEM") -> Issue:
    """Минимальная threat-Issue для unit-тестов annotate_*."""
    return Issue(
        code=code,
        severity=Severity.CRITICAL,
        confidence=Confidence.CERTAIN,
        message="test",
        location="x.pkl:offset 18",
        details={"matched_globals": "os.system"},
    )


def _raw(**kwargs: object) -> RawScanData:
    base: dict[str, object] = {
        "file_path": Path("x.pkl"),
        "file_hash": {},
        "file_size": 1,
        "scanner_name": "pickle",
    }
    base.update(kwargs)
    return RawScanData(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end: payload + оборванный хвост через полный пайплайн
# ---------------------------------------------------------------------------


class TestPayloadThenTruncatedPipeline:
    """Полный Scanner на фикстуре «payload + обрыв»."""

    def test_crit_os_system_present(self, tmp_path: Path) -> None:
        """КРИТ по os.system присутствует — payload не потерян из-за обрыва."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        fr = Scanner().scan(f).results_per_file[f]

        crit = [i for i in fr.issues if i.severity is Severity.CRITICAL]
        assert crit, f"КРИТ по os.system потерян. Codes: {_codes(fr.issues)}"
        assert any("os.system" in str(i.details.get("matched_globals", "")) for i in crit)

    def test_no_separate_parse_issue(self, tmp_path: Path) -> None:
        """MLS-PARSE-001 как ОТДЕЛЬНЫЙ Issue отсутствует (подавление работает)."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        fr = Scanner().scan(f).results_per_file[f]
        assert _CODE_PARSE not in _codes(fr.issues), _codes(fr.issues)

    def test_truncation_fact_visible_in_issue_details(self, tmp_path: Path) -> None:
        """Факт обрыва ВИДЕН в details находки — конкретные поля parse_truncated/parse_error."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        fr = Scanner().scan(f).results_per_file[f]

        crit = next(i for i in fr.issues if i.severity is Severity.CRITICAL)
        assert crit.details.get("parse_truncated") is True
        # Сырой текст ошибки genops сохранён (не хардкод в Output).
        assert "parse_error" in crit.details
        assert "remain" in str(crit.details["parse_error"]).lower() or "byte" in str(
            crit.details["parse_error"]
        ).lower()

    def test_truncation_fact_visible_in_sarif(self, tmp_path: Path) -> None:
        """SARIF (CI/SIEM-канал) теперь несёт факт обрыва — раньше он там терялся."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        res = Scanner().scan(f)

        sarif = SarifFormatter().format(res)
        doc = json.loads(sarif)
        result0 = doc["runs"][0]["results"][0]
        details = result0["properties"]["details"]
        assert details.get("parse_truncated") is True
        assert "parse_error" in details

    def test_forensic_raw_global_in_output(self, tmp_path: Path) -> None:
        """Форензик: сырое имя глобала os.system присутствует в выводе (не нормализовано)."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        res = Scanner().scan(f)

        js = JsonFormatter().format(res)
        sarif = SarifFormatter().format(res)
        # Сырое имя глобала видно в обоих машинных форматах.
        assert "os.system" in js
        assert "os.system" in sarif
        # И сырой текст ошибки genops (обрыв) — тоже виден в отчёте.
        assert "expected 255 bytes" in js
        assert "expected 255 bytes" in sarif

    def test_file_not_lost_record_and_issue_present(self, tmp_path: Path) -> None:
        """Инвариант «файл не теряется»: запись по файлу есть и в ней ≥1 Issue."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        res = Scanner().scan(f)

        assert f in res.results_per_file
        fr = res.results_per_file[f]
        assert fr.issues, "по упавшему файлу нет ни одной Issue — файл потерян"
        # error по файлу тоже сохранён (exit 2 отработает штатно).
        assert fr.error is not None


# ---------------------------------------------------------------------------
# Регресс: чистый непарсящийся файл БЕЗ payload → ровно один MLS-PARSE-001
# ---------------------------------------------------------------------------


class TestCleanUnparseableRegression:
    """Там, где специфичных фактов нет, подавление НЕ срабатывает."""

    def test_clean_truncated_yields_single_parse_issue(self, tmp_path: Path) -> None:
        """Обрыв без payload → ровно один MLS-PARSE-001 (LOW), без parse_truncated-enrichment."""
        f = tmp_path / "clean_truncated.pkl"
        f.write_bytes(_clean_truncated_no_payload())
        fr = Scanner().scan(f).results_per_file[f]

        assert _codes(fr.issues) == [_CODE_PARSE], _codes(fr.issues)
        issue = fr.issues[0]
        assert issue.severity is Severity.LOW
        # Отдельная Issue сама несёт текст ошибки; отдельного parse_truncated-флага
        # (enrichment для payload-случая) тут нет.
        assert "parse_truncated" not in issue.details


# ---------------------------------------------------------------------------
# Unit: annotate_suppressed_parse_error — узкое срабатывание
# ---------------------------------------------------------------------------


class TestAnnotateSuppressedParseError:
    """Прямое тестирование enrichment-функции (без пайплайна)."""

    def test_annotates_when_payload_and_error(self) -> None:
        """globals + error + threat-Issue → details обогащены parse_truncated/parse_error."""
        issues = [_threat_issue()]
        raw = _raw(globals={("os", "system")}, error="broken tail at position 18")
        annotate_suppressed_parse_error(issues, raw)
        assert issues[0].details["parse_truncated"] is True
        assert issues[0].details["parse_error"] == "broken tail at position 18"

    def test_no_annotation_without_error(self) -> None:
        """Без error — ничего не добавляется."""
        issues = [_threat_issue()]
        raw = _raw(globals={("os", "system")}, error=None)
        annotate_suppressed_parse_error(issues, raw)
        assert "parse_truncated" not in issues[0].details

    def test_no_annotation_without_extracted_payload(self) -> None:
        """error есть, но globals/reduce нет → это чистый сбой (свой MLS-PARSE-001)."""
        issues = [_threat_issue()]
        raw = _raw(error="broken", globals=None, reduce_calls=None)
        annotate_suppressed_parse_error(issues, raw)
        assert "parse_truncated" not in issues[0].details

    def test_no_annotation_for_bomb(self) -> None:
        """Bomb (свой MLS-BOMB-001 с деталями) не помечается parse_truncated."""
        issues = [_threat_issue(code="MLS-BOMB-001")]
        raw = _raw(
            scanner_name="joblib",
            globals={("os", "system")},
            error="Decompression bomb (zlib): ...",
            metadata={"joblib_decompression_bomb": "true"},
        )
        annotate_suppressed_parse_error(issues, raw)
        assert "parse_truncated" not in issues[0].details

    def test_no_annotation_when_parse_issue_present(self) -> None:
        """Если MLS-PARSE-001 уже в списке — факт в своей Issue, не дублируем в details."""
        parse_issue = Issue(
            code=_CODE_PARSE,
            severity=Severity.LOW,
            confidence=Confidence.LOW,
            message="parse fail",
            location="x.pkl",
            details={"error": "broken"},
        )
        threat = _threat_issue()
        issues = [parse_issue, threat]
        raw = _raw(globals={("os", "system")}, error="broken")
        annotate_suppressed_parse_error(issues, raw)
        assert "parse_truncated" not in threat.details

    def test_does_not_overwrite_existing_detail(self) -> None:
        """setdefault не затирает parse_error, уже положенный детектором."""
        issue = _threat_issue()
        issue.details["parse_error"] = "detector-provided"
        raw = _raw(globals={("os", "system")}, error="aggregator-error")
        annotate_suppressed_parse_error([issue], raw)
        assert issue.details["parse_error"] == "detector-provided"


# ---------------------------------------------------------------------------
# Регресс: подавление MLS-PARSE-001 при наличии globals (без enrichment-побочек)
# ---------------------------------------------------------------------------


def test_parse_detector_still_suppressed_on_globals() -> None:
    """ParseErrorDetector сам по-прежнему молчит при наличии globals."""
    raw = _raw(globals={("os", "system")}, error="broken tail")
    assert ParseErrorDetector().analyze(raw, _CTX) == []

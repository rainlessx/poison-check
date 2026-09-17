"""Факты УРОВНЯ ФАЙЛА доезжают до SARIF независимо от числа issues.

Закрывается КЛАСС, а не случай. Ранее ``SarifFormatter`` строил вывод только из
``file_result.issues`` и терял ``FileResult.error`` целиком, если по файлу не
было ни одной Issue (Console/JSON/SBOM его несли — системная асимметрия
форматов). Теперь файловые факты идут штатным каналом SARIF 2.1.0 —
``runs[].invocations[].toolExecutionNotifications[]`` (§3.58 notification,
§3.20 invocation), а не через синтетические result'ы (это исказило бы метрики
находок). Канал общий: факты собираются единой точкой
:mod:`poison_check.output.file_level_facts`, поэтому новый файловый факт доедет
до SARIF без правки форматтера.

Фикстуры:
* ``_bad_magic_gguf`` / ``_payload_then_truncated`` — ручная сборка байт (без
  ``pickle.dumps`` живого объекта), прогон через полный ``Scanner``.
* Состояние «error И ноль issues» штатно НЕ достижимо через пайплайн: сетка
  безопасности ``ParseErrorDetector`` (MLS-PARSE-001) гарантирует, что упавший
  файл всегда даёт ≥1 Issue. Поэтому это состояние конструируется на уровне
  МОДЕЛИ (``FileResult`` с ``error`` и ``issues=[]``) — задокументированное
  ограничение фикстуры. Именно этот случай раньше терялся в SARIF.

Ограничение валидации: каноническая JSON-схема SARIF 2.1.0 офлайн в репозитории
не поставляется, поэтому валидность проверяется СТРУКТУРНО вручную
(:func:`_assert_valid_sarif`): обязательные поля (version, $schema,
runs[].tool.driver), корректность ``level``-энумов и структуры notifications.
Это санкционированный запасной путь (ТЗ: «если jsonschema/схема недоступна —
проверить обязательные поля вручную и явно указать ограничение»).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from poison_check.core.result import (
    FileResult,
    Issue,
    ScanResult,
    Summary,
)
from poison_check.output.console import ConsoleFormatter
from poison_check.output.file_level_facts import SCAN_ERROR_DESCRIPTOR_ID
from poison_check.output.json_format import JsonFormatter
from poison_check.output.sarif import SarifFormatter
from poison_check.output.sbom import SbomFormatter
from poison_check.scanner import Scanner

_SARIF_LEVELS = {"note", "warning", "error"}


# ---------------------------------------------------------------------------
# Фикстуры (ручная сборка байт)
# ---------------------------------------------------------------------------


def _bad_magic_gguf() -> bytes:
    """GGUF с неверным magic: даёт FileResult.error И >=1 issue (MLS-GGUF-006).

    Ручная сборка байт: 4 байта неверного magic + версия + паддинг.
    """
    return b"XXXX" + b"\x03\x00\x00\x00" + b"\x00" * 32


def _payload_then_truncated() -> bytes:
    """os.system + REDUCE (КРИТ), затем оборванный опкод: error + issue + details.

    Тот же forensic-краевой случай, что в test_parse_suppression_forensics:
    payload извлечён, поток оборван → annotate_suppressed_parse_error кладёт
    parse_truncated/parse_error в details находки. Ручная сборка байт.
    """
    return (
        b"\x80\x04"          # PROTO 4
        b"cos\nsystem\n"     # GLOBAL os system
        b"\x8c\x02id"        # SHORT_BINUNICODE 'id'
        b"\x85"              # TUPLE1
        b"R"                 # REDUCE
        b"\x8c\xff"          # SHORT_BINUNICODE length=255, данных нет -> обрыв
    )


def _model_file_result(
    path: Path,
    *,
    error: str | None,
    issues: list[Issue] | None = None,
) -> FileResult:
    """FileResult, собранный на уровне модели (для недостижимых через пайплайн состояний)."""
    return FileResult(
        file_path=path,
        scanner_name="gguf",
        issues=issues or [],
        error=error,
    )


def _scan_result(results_per_file: dict[Path, FileResult]) -> ScanResult:
    """Оборачивает набор FileResult в ScanResult для форматтеров."""
    return ScanResult(
        tool_version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        duration_ms=1.0,
        results_per_file=results_per_file,
        summary=Summary(),
    )


# ---------------------------------------------------------------------------
# Структурная валидация SARIF 2.1.0 (ручная — схема офлайн недоступна)
# ---------------------------------------------------------------------------


def _assert_valid_sarif(doc: dict) -> None:
    """Проверяет обязательные поля и структуру SARIF 2.1.0 вручную.

    Ограничение: без канонической JSON-схемы (офлайн недоступна) — проверяем
    инварианты, которые нужны потребителям (GitHub Code Scanning / SIEM).
    """
    assert doc["version"] == "2.1.0"
    assert "$schema" in doc and "sarif" in doc["$schema"].lower()
    assert isinstance(doc["runs"], list) and doc["runs"]

    for run in doc["runs"]:
        driver = run["tool"]["driver"]
        assert isinstance(driver["name"], str) and driver["name"]

        # results: каждая находка структурно корректна
        for res in run.get("results", []):
            assert isinstance(res["ruleId"], str) and res["ruleId"]
            assert res["level"] in _SARIF_LEVELS
            assert isinstance(res["message"]["text"], str)
            assert isinstance(res["locations"], list)

        # invocations: обязательный executionSuccessful + корректные notifications
        for inv in run.get("invocations", []):
            assert isinstance(inv["executionSuccessful"], bool)
            for note in inv.get("toolExecutionNotifications", []):
                assert isinstance(note["message"]["text"], str)
                assert note["level"] in _SARIF_LEVELS
                assert isinstance(note["descriptor"]["id"], str) and note["descriptor"]["id"]
                assert isinstance(note["locations"], list) and note["locations"]
                phys = note["locations"][0]["physicalLocation"]
                assert isinstance(phys["artifactLocation"]["uri"], str)

        # driver.notifications: дескрипторы объявлены
        for desc in driver.get("notifications", []):
            assert isinstance(desc["id"], str) and desc["id"]


def _notifications(doc: dict) -> list[dict]:
    """Все toolExecutionNotifications из первого run."""
    notes: list[dict] = []
    for inv in doc["runs"][0].get("invocations", []):
        notes.extend(inv.get("toolExecutionNotifications", []))
    return notes


# ---------------------------------------------------------------------------
# (1) Файл с error и НОЛЬ issues → присутствует в SARIF, сырой текст ошибки
# ---------------------------------------------------------------------------


class TestErrorNoIssues:
    """Раньше терялся целиком; теперь виден через notifications."""

    _ERR = "Неверный magic bytes: b'XXXX' (ожидался b'GGUF')"

    def _doc(self) -> dict:
        p = Path("tests/fixtures/corrupt.gguf")
        res = _scan_result({p: _model_file_result(p, error=self._ERR)})
        return json.loads(SarifFormatter().format(res))

    def test_no_phantom_result(self) -> None:
        """Файловая ошибка НЕ добавляет фантомный result (метрики чисты)."""
        assert self._doc()["runs"][0]["results"] == []

    def test_present_via_notification(self) -> None:
        """Ноль следов запрещён: файл виден через toolExecutionNotifications."""
        notes = _notifications(self._doc())
        assert len(notes) == 1
        assert notes[0]["descriptor"]["id"] == SCAN_ERROR_DESCRIPTOR_ID

    def test_raw_error_text_present(self) -> None:
        """Сырой текст ошибки (форензик — без нормализации) в message notification."""
        note = _notifications(self._doc())[0]
        assert note["message"]["text"] == self._ERR
        assert note["level"] == "error"

    def test_notification_bound_to_artifact(self) -> None:
        """Notification привязан к конкретному файлу через locations."""
        note = _notifications(self._doc())[0]
        uri = note["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert "corrupt.gguf" in uri

    def test_valid_sarif(self) -> None:
        """Сгенерированный SARIF структурно валиден."""
        _assert_valid_sarif(self._doc())


# ---------------------------------------------------------------------------
# (2) Файл с error и >=1 issue → оба канала; details-корреляция не потеряна
# ---------------------------------------------------------------------------


class TestErrorWithIssue:
    """bad_magic GGUF и payload+обрыв — result и notification сосуществуют."""

    def test_bad_magic_both_channels(self, tmp_path: Path) -> None:
        """bad_magic GGUF: и result (MLS-GGUF-006), и notification присутствуют."""
        f = tmp_path / "bad_magic.gguf"
        f.write_bytes(_bad_magic_gguf())
        res = Scanner().scan(f)
        doc = json.loads(SarifFormatter().format(res))

        results = doc["runs"][0]["results"]
        assert [r["ruleId"] for r in results] == ["MLS-GGUF-006"]
        notes = _notifications(doc)
        assert len(notes) == 1
        assert notes[0]["descriptor"]["id"] == SCAN_ERROR_DESCRIPTOR_ID
        _assert_valid_sarif(doc)

    def test_payload_truncated_correlation_preserved(self, tmp_path: Path) -> None:
        """payload+обрыв: notification есть И details-корреляция в result цела."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        res = Scanner().scan(f)
        doc = json.loads(SarifFormatter().format(res))

        # Независимый канал: файловый факт как notification.
        notes = _notifications(doc)
        assert len(notes) == 1
        assert notes[0]["descriptor"]["id"] == SCAN_ERROR_DESCRIPTOR_ID

        # Точечный обход НЕ сломан: корреляция «payload найден И поток оборван»
        # осталась в details находки (annotate_suppressed_parse_error).
        result0 = doc["runs"][0]["results"][0]
        details = result0["properties"]["details"]
        assert details.get("parse_truncated") is True
        assert "parse_error" in details
        _assert_valid_sarif(doc)

    def test_raw_global_and_error_in_output(self, tmp_path: Path) -> None:
        """Форензик: сырое имя глобала и сырой текст обрыва — оба видны в SARIF."""
        f = tmp_path / "payload_then_truncated.pkl"
        f.write_bytes(_payload_then_truncated())
        sarif = SarifFormatter().format(Scanner().scan(f))
        assert "os.system" in sarif
        assert "expected 255 bytes" in sarif


# ---------------------------------------------------------------------------
# (3) Метрики не искажены: число result'ов == числу реальных issues
# ---------------------------------------------------------------------------


def test_metrics_not_distorted(tmp_path: Path) -> None:
    """Файловая ошибка идёт в notifications, не добавляя фантомный result."""
    f = tmp_path / "bad_magic.gguf"
    f.write_bytes(_bad_magic_gguf())
    res = Scanner().scan(f)
    doc = json.loads(SarifFormatter().format(res))

    real_issues = sum(len(fr.issues) for fr in res.results_per_file.values())
    assert len(doc["runs"][0]["results"]) == real_issues


# ---------------------------------------------------------------------------
# (4) Регресс: Console/JSON/SBOM несут error своим каналом (не изменены)
# ---------------------------------------------------------------------------


def test_other_formats_still_carry_error() -> None:
    """JSON/SBOM/Console по-прежнему несут FileResult.error своим штатным каналом."""
    err = "Неверный magic bytes: b'XXXX' (ожидался b'GGUF')"
    p = Path("tests/fixtures/corrupt.gguf")
    res = _scan_result({p: _model_file_result(p, error=err)})

    # JSON: поле error
    js = json.loads(JsonFormatter().format(res))
    assert js["results"][0]["error"] == err

    # SBOM: property poison-check:scan_error
    sbom = json.loads(SbomFormatter().format(res))
    props = sbom["components"][0]["properties"]
    assert any(
        pr["name"] == "poison-check:scan_error" and pr["value"] == err for pr in props
    )

    # Console: строка ошибки печатается (проверяем через capture)
    import io

    from rich.console import Console

    buf = io.StringIO()
    formatter = ConsoleFormatter(console=Console(file=buf, force_terminal=False, width=200))
    formatter.format_file_result(res.results_per_file[p])
    assert "XXXX" in buf.getvalue()


# ---------------------------------------------------------------------------
# (5) Инвариант КЛАССА: N файлов, часть с error → каждый error виден в SARIF
# ---------------------------------------------------------------------------


def test_class_invariant_every_errored_file_present() -> None:
    """Набор файлов: каждый файл с error представлен в SARIF-канале фактов."""
    errored = {
        Path("a.gguf"): "error A: битый контейнер",
        Path("b.pkl"): "error B: обрыв опкода",
        Path("c.joblib"): "error C: bad decompress",
    }
    clean = {Path("clean1.safetensors"), Path("clean2.npy")}

    rpf: dict[Path, FileResult] = {
        p: _model_file_result(p, error=e) for p, e in errored.items()
    }
    for p in clean:
        rpf[p] = _model_file_result(p, error=None)

    doc = json.loads(SarifFormatter().format(_scan_result(rpf)))
    _assert_valid_sarif(doc)

    notes = _notifications(doc)
    # Каждый errored-файл представлен ровно одним notification с сырым текстом.
    assert len(notes) == len(errored)
    covered = {
        note["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]: note[
            "message"
        ]["text"]
        for note in notes
    }
    for p, e in errored.items():
        uri = SarifFormatter._path_to_uri(p)
        assert uri in covered
        assert covered[uri] == e

    # Чистые файлы фантомных notifications не порождают.
    assert not (set(covered) & {SarifFormatter._path_to_uri(p) for p in clean})
    # И метрики находок пусты (файловые ошибки ≠ уязвимости).
    assert doc["runs"][0]["results"] == []

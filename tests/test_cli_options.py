"""Тесты CLI-опций и МЕТА-ТЕСТ целостности CLI.

Закрываемый класс дефекта — рассинхрон «объявлено ↔ показано ↔ применяется».
До этой задачи в CLI одновременно жили три его формы:

* ``--recursive`` вообще не существовал: ``typer.Option("-r/--no-recursive")``
  объявляет положительной формой ТОЛЬКО ``-r``, и команда с ``--recursive``
  падала с «No such option»;
* ``--format sbom`` не существовал при готовом ``SbomFormatter``, а опечатка в
  ``--format`` обнаруживалась ПОСЛЕ полного сканирования каталога;
* ``--output`` при ``--format console`` молча игнорировался — файл не
  создавался и предупреждения не было;
* ``--client``/``--auditor`` доезжали только до PDF и молча терялись в
  console/JSON/SARIF/SBOM.

Поэтому файл проверяет не только поведение каждой опции, но и сам инвариант
целостности (класс TestCliIntegrity) — включая ЛОВУШКУ: во временное
приложение добавляется скрытая опция, и проверяется, что гейт на ней краснеет.
Иначе мета-тест был бы такой же декорацией, как опции, от которых он защищает.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Annotated, Any

import pytest
import typer
from typer.testing import CliRunner

from poison_check.cli import OutputFormat, app
from poison_check.cli_options import (
    CLI_OPTIONS,
    TOP_LEVEL,
    CliOptionSpec,
    options_missing_from_help,
    stale_option_specs,
    unregistered_options,
)
from poison_check.i18n.loader import I18n
from poison_check.output import console as console_module

runner = CliRunner()

_FIXTURES = Path(__file__).parent / "fixtures"
_SAFE = _FIXTURES / "safe"
_MALICIOUS = _FIXTURES / "malicious"

#: Широкий терминал: rich переносит длинные строки help по ширине и мог бы
#: разорвать саму форму опции, из-за чего проверка видимости стала бы флаки.
_WIDE_ENV = {"COLUMNS": "200", "TERM": "dumb"}


@pytest.fixture(autouse=True)
def _reset_i18n() -> None:
    """Сбрасывает singleton I18n перед каждым тестом."""
    I18n.reset()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _help_of(command: str) -> str:
    """Возвращает текст ``--help`` указанной команды.

    Пустое имя (:data:`~poison_check.cli_options.TOP_LEVEL`) означает help
    самого приложения, а не подкоманды.
    """
    args = ["--help"] if command == TOP_LEVEL else [command, "--help"]
    result = runner.invoke(app, args, env=_WIDE_ENV)
    assert result.exit_code == 0, f"`{command} --help` завершился с ошибкой"
    return result.output


def _scan_json(*args: str) -> dict[str, Any]:
    """Запускает ``scan ... --format json`` и разбирает отчёт."""
    result = runner.invoke(app, ["scan", *args, "--format", "json"])
    try:
        parsed: dict[str, Any] = json.loads(result.output)
    except json.JSONDecodeError as exc:  # pragma: no cover — диагностика
        pytest.fail(f"Вывод не JSON: {exc}\n{result.output}")
    return parsed


def _write_pickle(path: Path) -> Path:
    """Создаёт минимальный безобидный pickle (protocol 4, значение None)."""
    path.write_bytes(b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00N.")
    return path


def _write_url_pickle(path: Path) -> Path:
    """Создаёт pickle со строкой-URL — находка MLS-NET-002.

    Собран вручную опкодами (BINUNICODE + STOP), а не через ``pickle.dumps``:
    требование CLAUDE.md к вредоносным/сигнальным фикстурам.
    """
    payload = b"https://attacker.example/exfil"
    path.write_bytes(
        b"\x80\x04" + b"X" + len(payload).to_bytes(4, "little") + payload + b"."
    )
    return path


def _resolve(ref: str) -> object:
    """Резолвит ссылку ``"модуль:qualname"`` в объект.

    Именно это делает точку применения машинно-проверяемой: строка в реестре
    либо указывает на существующий объект, либо тест падает.
    """
    module_name, sep, qualname = ref.partition(":")
    assert sep and qualname, f"Ссылка {ref!r} должна иметь формат 'модуль:qualname'"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover — сообщение важнее ветки
        raise AssertionError(
            f"Ссылка {ref!r}: модуль {module_name!r} не импортируется ({exc})"
        ) from exc

    obj: Any = module
    for part in qualname.split("."):
        assert hasattr(obj, part), (
            f"Ссылка {ref!r} не найдена: у {obj!r} нет атрибута {part!r}. "
            f"Опция объявлена применяемой и покрытой, но код/тест отсутствует."
        )
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# 1. Видимость: --help верхнего уровня и scan --help
# ---------------------------------------------------------------------------


class TestHelpVisibility:
    """``--help`` показывает всё, что принимает парсер."""

    def test_top_level_help_lists_all_commands(self) -> None:
        """Верхний ``--help`` перечисляет все три команды."""
        output = runner.invoke(app, ["--help"], env=_WIDE_ENV).output
        for command in ("scan", "list-scanners", "doctor"):
            assert command in output, f"Команда {command} не показана в --help:\n{output}"

    def test_scan_help_lists_all_published_options(self) -> None:
        """``scan --help`` содержит каждую опубликованную опцию реестра."""
        output = " ".join(_help_of("scan").split())
        for spec in CLI_OPTIONS:
            if spec.command != "scan" or not spec.option.startswith("--"):
                continue
            assert spec.option in output, (
                f"Опция {spec.option} принимается парсером, но не показана в "
                f"`scan --help`:\n{output}"
            )

    def test_recursive_long_form_exists(self) -> None:
        """Регрессия: ``--recursive`` существует, а не только ``-r``.

        До фикса ``typer.Option("-r/--no-recursive")`` объявлял положительной
        формой только ``-r``, и ``--recursive`` падал с «No such option».
        """
        output = " ".join(_help_of("scan").split())
        assert "--recursive" in output
        assert "--no-recursive" in output

    def test_format_choices_shown_in_help(self) -> None:
        """``scan --help`` перечисляет допустимые значения ``--format``."""
        output = " ".join(_help_of("scan").split())
        for value in (fmt.value for fmt in OutputFormat):
            assert value in output, f"Значение --format {value} не показано в help"

    def test_policy_help_lists_builtin_names(self) -> None:
        """``scan --help`` перечисляет встроенные политики."""
        output = " ".join(_help_of("scan").split())
        for name in ("default", "banking", "government", "strict"):
            assert name in output, f"Встроенная политика {name} не упомянута в help"


# ---------------------------------------------------------------------------
# 2. --format
# ---------------------------------------------------------------------------


class TestFormatOption:
    """``--format`` действительно меняет формат вывода."""

    @pytest.mark.parametrize(
        ("fmt", "marker"),
        [
            ("json", "schema_version"),
            ("sarif", "$schema"),
            ("sbom", "bomFormat"),
        ],
    )
    def test_each_format_produces_its_own_shape(
        self, tmp_path: Path, fmt: str, marker: str
    ) -> None:
        """Каждое значение ``--format`` даёт документ своей схемы."""
        model = _write_pickle(tmp_path / "m.pkl")
        result = runner.invoke(app, ["scan", str(model), "--format", fmt])
        doc = json.loads(result.output)
        assert marker in doc, f"В выводе --format {fmt} нет маркера {marker}: {list(doc)}"

    def test_console_format_is_human_readable(self, tmp_path: Path) -> None:
        """``--format console`` даёт текстовый отчёт, а не JSON."""
        model = _write_pickle(tmp_path / "m.pkl")
        result = runner.invoke(app, ["scan", str(model)])
        assert "Начинаю сканирование" in result.output
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.output)

    def test_html_format_reachable_from_cli(self, tmp_path: Path) -> None:
        """``--format html`` рендерит HTML тем же шаблоном, что и PDF.

        README обещал этот формат, а CLI его не знал: HTML умел только
        ``PdfReportFormatter.format_html`` через Python API.
        """
        pytest.importorskip("jinja2", reason="HTML-отчёт требует jinja2")
        model = _write_pickle(tmp_path / "m.pkl")
        result = runner.invoke(app, ["scan", str(model), "--format", "html"])
        assert "<html" in result.output.lower(), f"Не HTML:\n{result.output[:400]}"

    def test_sbom_format_reachable_from_cli(self, tmp_path: Path) -> None:
        """Регрессия: ``--format sbom`` существует (SbomFormatter был недостижим)."""
        model = _write_pickle(tmp_path / "m.pkl")
        result = runner.invoke(app, ["scan", str(model), "--format", "sbom"])
        doc = json.loads(result.output)
        assert doc["bomFormat"] == "CycloneDX"
        assert doc["specVersion"] == "1.4"

    def test_invalid_format_is_clean_error(self, tmp_path: Path) -> None:
        """Невалидное значение → внятная ошибка со списком значений, без трейсбека."""
        model = _write_pickle(tmp_path / "m.pkl")
        result = runner.invoke(app, ["scan", str(model), "--format", "yaml"])

        assert result.exit_code == 2, f"Ожидался exit code 2, получен {result.exit_code}"
        combined = " ".join((result.output + result.stderr).split())
        assert "Traceback" not in combined, f"Трейсбек в выводе:\n{combined}"
        assert "yaml" in combined
        assert "console" in combined and "sarif" in combined, (
            f"Ошибка не перечисляет допустимые значения:\n{combined}"
        )

    def test_invalid_format_rejected_before_scanning(self, tmp_path: Path) -> None:
        """Опечатка отвергается ДО сканирования: файлы не читаются."""
        model = _write_pickle(tmp_path / "m.pkl")
        result = runner.invoke(app, ["scan", str(model), "--format", "yaml"])
        combined = result.output + result.stderr
        assert "Сканируется файл" not in combined, (
            "Разбор аргументов обязан завершиться до начала сканирования"
        )


# ---------------------------------------------------------------------------
# 3. --policy
# ---------------------------------------------------------------------------


class TestPolicyOption:
    """``--policy`` резолвится в существующий механизм политик."""

    @pytest.mark.parametrize("name", ["default", "banking", "government", "strict"])
    def test_each_builtin_policy_applies(self, tmp_path: Path, name: str) -> None:
        """Каждая встроенная политика загружается и попадает в отчёт."""
        model = _write_pickle(tmp_path / "m.pkl")
        parsed = _scan_json(str(model), "--policy", name)
        assert parsed["tool"]["policy"] == name
        assert parsed["tool"]["thresholds"]["fail_on_severity"] is not None

    def test_banking_escalates_external_url(self, tmp_path: Path) -> None:
        """banking: внешний URL в модели эскалируется до HIGH (MLS-NET-002)."""
        model = _write_url_pickle(tmp_path / "c2.pkl")
        parsed = _scan_json(str(model), "--policy", "banking")
        severities = [
            i["severity"]
            for r in parsed["results"]
            for i in r["issues"]
            if i["code"] == "MLS-NET-002"
        ]
        assert severities == ["high"], f"Ожидался HIGH под banking, получено: {severities}"

    def test_default_does_not_escalate(self, tmp_path: Path) -> None:
        """default: та же находка остаётся MEDIUM — политики дают разный результат."""
        model = _write_url_pickle(tmp_path / "c2.pkl")
        parsed = _scan_json(str(model), "--policy", "default")
        severities = [
            i["severity"]
            for r in parsed["results"]
            for i in r["issues"]
            if i["code"] == "MLS-NET-002"
        ]
        assert severities == ["medium"], f"Ожидался MEDIUM под default, получено: {severities}"

    def test_custom_yaml_path_accepted(self, tmp_path: Path) -> None:
        """``--policy /path/to.yaml`` загружает пользовательскую политику."""
        policy = tmp_path / "custom.yaml"
        policy.write_text(
            "name: custom_test\n"
            "description: Тестовая политика\n"
            "fail_on_severity: high\n"
            "compliance: []\n",
            encoding="utf-8",
        )
        model = _write_pickle(tmp_path / "m.pkl")
        parsed = _scan_json(str(model), "--policy", str(policy))
        assert parsed["tool"]["thresholds"]["fail_on_severity"] == "high"


# ---------------------------------------------------------------------------
# 4. --output
# ---------------------------------------------------------------------------


class TestOutputOption:
    """``--output`` пишет отчёт в файл во всех форматах."""

    @pytest.mark.parametrize("fmt", ["json", "sarif", "sbom"])
    def test_output_writes_file(self, tmp_path: Path, fmt: str) -> None:
        """Текстовый формат сохраняется в указанный файл как валидный JSON."""
        model = _write_pickle(tmp_path / "m.pkl")
        out = tmp_path / f"report.{fmt}"
        runner.invoke(app, ["scan", str(model), "--format", fmt, "--output", str(out)])

        assert out.is_file(), f"--output не создал файл для формата {fmt}"
        assert json.loads(out.read_text(encoding="utf-8"))

    def test_console_format_writes_file(self, tmp_path: Path) -> None:
        """Регрессия: ``--format console --output`` пишет файл, а не игнорирует флаг."""
        model = _write_pickle(tmp_path / "m.pkl")
        out = tmp_path / "report.txt"
        result = runner.invoke(app, ["scan", str(model), "--output", str(out)])

        assert out.is_file(), (
            "--output при --format console молча игнорировался — файл не создан"
        )
        text = out.read_text(encoding="utf-8")
        assert "Начинаю сканирование" in text
        assert "\x1b[" not in text, "В файле не должно быть ANSI-разметки"
        assert "Отчёт сохранён" in result.output

    def test_unwritable_path_reports_error(self, tmp_path: Path) -> None:
        """Недоступный путь → внятная ошибка и exit code 2, без трейсбека."""
        model = _write_pickle(tmp_path / "m.pkl")
        unwritable = tmp_path / "nonexistent_dir" / "report.json"
        result = runner.invoke(
            app, ["scan", str(model), "--format", "json", "--output", str(unwritable)]
        )

        assert result.exit_code == 2, f"Ожидался exit code 2, получен {result.exit_code}"
        assert "Traceback" not in result.output, f"Трейсбек в выводе:\n{result.output}"
        assert "Ошибка записи" in result.output


# ---------------------------------------------------------------------------
# 5. --recursive / --no-recursive
# ---------------------------------------------------------------------------


class TestRecursiveOption:
    """Семантика ``--recursive``: по умолчанию обхода вглубь НЕТ."""

    @staticmethod
    def _tree(tmp_path: Path) -> Path:
        """Каталог с файлом на верхнем уровне и файлом во вложенном каталоге."""
        root = tmp_path / "models"
        (root / "nested").mkdir(parents=True)
        _write_pickle(root / "top.pkl")
        _write_pickle(root / "nested" / "deep.pkl")
        return root

    def _scanned(self, root: Path, *flags: str) -> list[str]:
        parsed = _scan_json(str(root), *flags)
        return sorted(Path(r["file"]["path"]).name for r in parsed["results"])

    def test_default_is_not_recursive(self, tmp_path: Path) -> None:
        """Без флага сканируется только верхний уровень каталога."""
        root = self._tree(tmp_path)
        assert self._scanned(root) == ["top.pkl"]

    def test_recursive_finds_nested_file(self, tmp_path: Path) -> None:
        """``--recursive`` добавляет файлы вложенных каталогов."""
        root = self._tree(tmp_path)
        assert self._scanned(root, "--recursive") == ["deep.pkl", "top.pkl"]

    def test_short_form_equals_long_form(self, tmp_path: Path) -> None:
        """``-r`` и ``--recursive`` дают одинаковый результат."""
        root = self._tree(tmp_path)
        assert self._scanned(root, "-r") == self._scanned(root, "--recursive")

    def test_no_recursive_is_explicit_default(self, tmp_path: Path) -> None:
        """``--no-recursive`` — явная форма поведения по умолчанию."""
        root = self._tree(tmp_path)
        assert self._scanned(root, "--no-recursive") == self._scanned(root)

    def test_recursive_skips_hidden_paths(self, tmp_path: Path) -> None:
        """Задокументированная семантика: при рекурсии скрытые пути пропускаются."""
        root = self._tree(tmp_path)
        (root / ".cache").mkdir()
        _write_pickle(root / ".cache" / "hidden.pkl")
        assert "hidden.pkl" not in self._scanned(root, "--recursive")


# ---------------------------------------------------------------------------
# 6. --client / --auditor (метаданные шапки отчёта)
# ---------------------------------------------------------------------------


class TestReportMetadataOptions:
    """``--client``/``--auditor`` доезжают до ВСЕХ форматов единообразно."""

    _CLIENT = "ООО Ромашка"
    _AUDITOR = "Иванов И.И."

    def _outputs(self, model: Path, *extra: str) -> dict[str, str]:
        """Возвращает ``формат → текст отчёта`` для всех текстовых форматов."""
        formats = ["console", "json", "sarif", "sbom"]
        if importlib.util.find_spec("jinja2") is not None:
            formats.append("html")

        outputs: dict[str, str] = {}
        for fmt in formats:
            result = runner.invoke(app, ["scan", str(model), "--format", fmt, *extra])
            outputs[fmt] = result.output
        return outputs

    def test_client_present_in_all_formats(self, tmp_path: Path) -> None:
        """Заказчик виден в console, JSON, SARIF и SBOM."""
        model = _write_pickle(tmp_path / "m.pkl")
        outputs = self._outputs(model, "--client", self._CLIENT)
        for fmt, text in outputs.items():
            assert self._CLIENT in text, f"--client потерян в формате {fmt}:\n{text}"

    def test_auditor_present_in_all_formats(self, tmp_path: Path) -> None:
        """Аудитор виден в console, JSON, SARIF и SBOM."""
        model = _write_pickle(tmp_path / "m.pkl")
        outputs = self._outputs(model, "--auditor", self._AUDITOR)
        for fmt, text in outputs.items():
            assert self._AUDITOR in text, f"--auditor потерян в формате {fmt}:\n{text}"

    def test_metadata_lands_in_documented_places(self, tmp_path: Path) -> None:
        """Реквизиты лежат в предсказуемых местах структурированных форматов."""
        model = _write_pickle(tmp_path / "m.pkl")
        args = ["--client", self._CLIENT, "--auditor", self._AUDITOR]

        parsed = _scan_json(str(model), *args)
        assert parsed["report_metadata"] == {
            "client": self._CLIENT,
            "auditor": self._AUDITOR,
        }

        sarif = json.loads(
            runner.invoke(app, ["scan", str(model), "--format", "sarif", *args]).output
        )
        assert sarif["runs"][0]["properties"]["client"] == self._CLIENT
        assert sarif["runs"][0]["properties"]["auditor"] == self._AUDITOR

        sbom = json.loads(
            runner.invoke(app, ["scan", str(model), "--format", "sbom", *args]).output
        )
        props = {p["name"]: p["value"] for p in sbom["metadata"]["properties"]}
        assert props["poison-check:client"] == self._CLIENT
        assert props["poison-check:auditor"] == self._AUDITOR

    def test_empty_metadata_omitted(self, tmp_path: Path) -> None:
        """Без флагов поля ОПУСКАЮТСЯ, а не пишутся пустыми."""
        model = _write_pickle(tmp_path / "m.pkl")

        parsed = _scan_json(str(model))
        assert "report_metadata" not in parsed, (
            "Незаданные реквизиты не должны создавать секцию отчёта"
        )

        sarif = json.loads(
            runner.invoke(app, ["scan", str(model), "--format", "sarif"]).output
        )
        assert "properties" not in sarif["runs"][0]

        sbom = json.loads(
            runner.invoke(app, ["scan", str(model), "--format", "sbom"]).output
        )
        names = {p["name"] for p in sbom["metadata"]["properties"]}
        assert "poison-check:client" not in names
        assert "poison-check:auditor" not in names

    def test_partial_metadata_omits_only_missing_field(self, tmp_path: Path) -> None:
        """Задан только заказчик → в отчёте только он, без пустого аудитора."""
        model = _write_pickle(tmp_path / "m.pkl")
        parsed = _scan_json(str(model), "--client", self._CLIENT)
        assert parsed["report_metadata"] == {"client": self._CLIENT}

    def test_whitespace_only_treated_as_empty(self, tmp_path: Path) -> None:
        """``--client "   "`` — это незаданное значение, а не пустая строка."""
        model = _write_pickle(tmp_path / "m.pkl")
        parsed = _scan_json(str(model), "--client", "   ")
        assert "report_metadata" not in parsed


# ---------------------------------------------------------------------------
# 7. Прочие опции: --locale, --no-emoji, --verbose
# ---------------------------------------------------------------------------


class TestLocaleOption:
    """``--locale`` переключает язык сообщений."""

    def test_locale_switches_language(self, tmp_path: Path) -> None:
        """``--locale en`` даёт английские сообщения вместо русских."""
        model = _write_pickle(tmp_path / "m.pkl")
        ru = runner.invoke(app, ["scan", str(model), "--locale", "ru"]).output
        I18n.reset()
        en = runner.invoke(app, ["scan", str(model), "--locale", "en"]).output

        assert "Начинаю сканирование" in ru
        assert "Starting scan" in en
        assert "Начинаю сканирование" not in en


class TestNoEmojiOption:
    """``--no-emoji`` переключает иконки на ASCII."""

    def test_no_emoji_uses_ascii_icons(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """С флагом иконки severity становятся ASCII, без флага — эмодзи.

        Авто-детект в тестах всегда выключает эмодзи (stdout не TTY), поэтому
        он принудительно включён — иначе разницы между «с флагом» и «без
        флага» не было бы видно и тест ничего бы не доказывал.
        """
        model = _MALICIOUS / "os_system.pkl"
        if not model.exists():
            pytest.skip(f"Фикстура не найдена: {model}")

        monkeypatch.setattr(console_module, "_should_use_emoji", lambda: True)

        without_flag = runner.invoke(app, ["scan", str(model)]).output
        I18n.reset()
        with_flag = runner.invoke(app, ["scan", str(model), "--no-emoji"]).output

        assert "[!]" not in without_flag, (
            f"Без флага ожидались эмодзи-иконки:\n{without_flag}"
        )
        assert "[!]" in with_flag, f"ASCII-иконка не найдена:\n{with_flag}"


class TestVerboseOption:
    """``--verbose`` печатает breakdown глобалов в stderr."""

    def test_verbose_lists_globals(self) -> None:
        """С флагом в stderr появляется разметка глобалов файла."""
        model = _MALICIOUS / "os_system.pkl"
        if not model.exists():
            pytest.skip(f"Фикстура не найдена: {model}")

        result = runner.invoke(app, ["scan", str(model), "--verbose"])
        assert "verbose" in result.stderr
        assert "os.system" in result.stderr

    def test_without_verbose_no_globals_dump(self) -> None:
        """Без флага breakdown не печатается."""
        model = _MALICIOUS / "os_system.pkl"
        if not model.exists():
            pytest.skip(f"Фикстура не найдена: {model}")

        result = runner.invoke(app, ["scan", str(model)])
        assert "globals (" not in result.stderr


# ---------------------------------------------------------------------------
# 8. МЕТА-ТЕСТ целостности CLI + ловушки
# ---------------------------------------------------------------------------


def _trap_scan_hidden(
    secret: Annotated[
        str, typer.Option("--secret-flag", hidden=True, help="Фиктивная опция")
    ] = "",
) -> None:
    """Фиктивная команда со СКРЫТОЙ опцией — приманка для гейта видимости."""


def _trap_scan_visible(
    secret: Annotated[
        str, typer.Option("--secret-flag", hidden=False, help="Фиктивная опция")
    ] = "",
) -> None:
    """Та же команда с ВИДИМОЙ опцией — контроль, что гейт не срабатывает зря."""


def _trap_top_level_callback(
    secret: Annotated[
        str, typer.Option("--secret-global", hidden=True, help="Фиктивная опция")
    ] = "",
) -> None:
    """Скрытая опция ВЕРХНЕГО УРОВНЯ — приманка для проверки TOP_LEVEL."""


def _trap_app(*, hidden: bool) -> typer.Typer:
    """Строит временное приложение с одной опцией, скрытой или видимой.

    Используется ловушкой: гейт обязан краснеть на скрытой опции и молчать на
    той же опции без ``hidden=True``.
    """
    trap = typer.Typer(add_completion=False)
    trap.command("scan")(_trap_scan_hidden if hidden else _trap_scan_visible)
    return trap


def _trap_app_with_hidden_top_level() -> typer.Typer:
    """Приложение со скрытой опцией НЕ в подкоманде, а в самом приложении."""
    trap = typer.Typer(add_completion=False)
    trap.callback()(_trap_top_level_callback)
    trap.command("scan")(_trap_scan_visible)
    return trap


class TestCliIntegrity:
    """«Объявленное = показанное = применяемое» для каждой опции CLI."""

    def test_every_parser_option_is_registered(self) -> None:
        """Каждый параметр парсера имеет запись в реестре CLI_OPTIONS."""
        unknown = unregistered_options(app)
        assert unknown == [], (
            f"Опции принимаются парсером, но не объявлены в CLI_OPTIONS: {unknown}. "
            f"Незаявленная опция может молча ничего не делать — добавьте запись "
            f"с applied_by и proven_by."
        )

    def test_no_stale_option_specs(self) -> None:
        """В реестре нет записей об опциях, которых больше нет в парсере."""
        stale = stale_option_specs(app)
        assert stale == [], f"Реестр описывает несуществующие опции: {stale}"

    def test_every_option_visible_in_help(self) -> None:
        """Ни одной скрытой формы: всё, что принимает парсер, показано в help."""
        missing = options_missing_from_help(app, _help_of)
        assert missing == [], (
            f"Опции принимаются парсером, но не показаны в --help: {missing}"
        )

    @pytest.mark.parametrize("spec", CLI_OPTIONS, ids=lambda s: f"{s.command}{s.option}")
    def test_spec_declares_application_and_proof(self, spec: CliOptionSpec) -> None:
        """У каждой опции заявлены и точка применения, и доказывающий тест."""
        assert spec.applied_by, (
            f"{spec.command} {spec.option}: не указана точка применения — "
            f"опция принимается, но неизвестно, влияет ли на поведение"
        )
        assert spec.proven_by, (
            f"{spec.command} {spec.option}: не указан тест, доказывающий влияние"
        )
        assert spec.summary, f"{spec.command} {spec.option}: пустое описание"

    @pytest.mark.parametrize("spec", CLI_OPTIONS, ids=lambda s: f"{s.command}{s.option}")
    def test_applied_by_references_resolve(self, spec: CliOptionSpec) -> None:
        """Каждая ссылка ``applied_by`` указывает на существующий объект."""
        for ref in spec.applied_by:
            _resolve(ref)

    @pytest.mark.parametrize("spec", CLI_OPTIONS, ids=lambda s: f"{s.command}{s.option}")
    def test_proven_by_references_resolve(self, spec: CliOptionSpec) -> None:
        """Каждая ссылка ``proven_by`` указывает на существующий тест."""
        for ref in spec.proven_by:
            _resolve(ref)


class TestCliIntegrityTraps:
    """Ловушки: гейт обязан КРАСНЕТЬ на подложенном дефекте.

    Без этих тестов мета-тест был бы декоративным ровно так же, как опции, от
    которых он защищает.
    """

    def test_hidden_option_is_detected(self) -> None:
        """Скрытая опция во временном приложении ловится проверкой видимости."""
        trap = _trap_app(hidden=True)

        def help_of(command: str) -> str:
            args = ["--help"] if command == TOP_LEVEL else [command, "--help"]
            return runner.invoke(trap, args, env=_WIDE_ENV).output

        missing = options_missing_from_help(trap, help_of)
        assert "scan --secret-flag" in missing, (
            "Ловушка не сработала: скрытая опция не обнаружена, значит гейт "
            "видимости не защищает и настоящее приложение"
        )

    def test_same_option_visible_passes(self) -> None:
        """Та же опция БЕЗ hidden=True гейт не роняет — проверка не тривиальна."""
        trap = _trap_app(hidden=False)

        def help_of(command: str) -> str:
            args = ["--help"] if command == TOP_LEVEL else [command, "--help"]
            return runner.invoke(trap, args, env=_WIDE_ENV).output

        assert options_missing_from_help(trap, help_of) == []

    def test_hidden_top_level_option_is_detected(self) -> None:
        """Скрытая опция САМОГО приложения ловится так же, как в подкоманде.

        Опция, добавленная не в подкоманду, а в callback приложения, обошла бы
        гейт стороной, если бы он смотрел только на подкоманды.
        """
        trap = _trap_app_with_hidden_top_level()

        def help_of(command: str) -> str:
            args = ["--help"] if command == TOP_LEVEL else [command, "--help"]
            return runner.invoke(trap, args, env=_WIDE_ENV).output

        assert "--secret-global" in options_missing_from_help(trap, help_of), (
            "Ловушка не сработала: скрытая опция верхнего уровня не обнаружена"
        )

    def test_missing_registry_entry_is_detected(self) -> None:
        """Опция без записи в реестре ловится проверкой полноты реестра."""
        truncated = tuple(spec for spec in CLI_OPTIONS if spec.option != "--policy")
        unknown = unregistered_options(app, truncated)
        assert "scan --policy" in unknown, (
            "Ловушка не сработала: опция без записи в реестре не обнаружена"
        )

    def test_stale_registry_entry_is_detected(self) -> None:
        """Запись о несуществующей опции ловится проверкой актуальности."""
        ghost = CliOptionSpec(
            option="--ghost",
            command="scan",
            summary="Опция, которой нет в парсере",
            applied_by=("poison_check.cli:scan",),
            proven_by=("tests.test_cli_options:TestCliIntegrityTraps",),
        )
        stale = stale_option_specs(app, (*CLI_OPTIONS, ghost))
        assert "scan --ghost" in stale, (
            "Ловушка не сработала: устаревшая запись реестра не обнаружена"
        )

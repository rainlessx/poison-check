"""РЕЕСТР CLI-опций: опция → команда → точка применения → тест.

Класс дефекта, который закрывает этот модуль: рассинхрон «объявлено ↔ показано
↔ применяется». Опция регистрируется в парсере, но не видна в ``--help``; или
видна, принимается — и нигде не читается. Пользователь уверен, что настроил
поведение, а оно молчит. В проекте это уже случалось с ключами политик
(см. :mod:`poison_check.policies`, реестр ``POLICY_KEYS``) и повторилось в CLI:
``--client``/``--auditor`` доезжали только до PDF, ``--output`` молча
игнорировался при ``--format console``, ``--format sbom`` не существовал при
наличии ``SbomFormatter``, а ``--recursive`` вообще не существовал как длинная
форма — парсер знал только ``-r``.

Обычные тесты класс не ловят: «опция есть в сигнатуре и парсится» ≠ «опция
показана пользователю» ≠ «опция влияет на поведение». Поэтому каждая опция
обязана иметь запись в :data:`CLI_OPTIONS` с машинно-проверяемой привязкой:

* :attr:`CliOptionSpec.applied_by` — где опция читается и влияет на поведение;
* :attr:`CliOptionSpec.proven_by`  — тест, который это влияние доказывает.

Ссылки записаны как ``"модуль:qualname"`` и резолвятся импортом. Мета-тест
``tests/test_cli_options.py`` проверяет три инварианта разом:

1. каждый параметр парсера имеет запись в реестре (нет необъявленных опций);
2. каждая опция реестра присутствует в ``--help`` своей команды (нет скрытых);
3. каждая ссылка ``applied_by``/``proven_by`` резолвится (нет опций, которые
   принимаются, но ничего не делают, и нет недоказанных заявлений).

Оба списка обязательны и непусты по конструкции: объявить опцию «просто так»,
без точки применения и без доказывающего теста, реестр не позволяет.

Слой: CLI. Модуль намеренно НЕ импортирует :mod:`poison_check.cli` — иначе
получился бы цикл; проверки принимают приложение параметром.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click
import typer

#: Имя «команды» для параметров самого приложения (не подкоманды). Пустая
#: строка, потому что в командной строке такие опции пишутся без имени команды:
#: ``poison-check --version``.
TOP_LEVEL: str = ""


@dataclass(frozen=True)
class CliOptionSpec:
    """Декларация одной CLI-опции (или аргумента) и её точки применения.

    :ivar option: Основная форма, как её видит пользователь: ``"--policy"``
        для опции, имя параметра (``"path"``) для позиционного аргумента.
    :ivar command: Имя команды CLI (``"scan"``, ``"doctor"``, ...).
    :ivar summary: Что опция делает — одна строка для документации.
    :ivar applied_by: Точки в коде, где значение опции читается и влияет на
        поведение или на вывод.
    :ivar proven_by: Тесты, доказывающие это влияние.
    :ivar aliases: Дополнительные формы (короткая, отрицательная), которые
        обязаны присутствовать в ``--help`` наравне с основной.
    """

    option: str
    command: str
    summary: str
    applied_by: tuple[str, ...]
    proven_by: tuple[str, ...]
    aliases: tuple[str, ...] = ()


#: Все опции и аргументы CLI. Добавляете параметр в команду — добавьте строку
#: сюда, иначе мета-тест упадёт (и правильно сделает).
CLI_OPTIONS: tuple[CliOptionSpec, ...] = (
    CliOptionSpec(
        option="path",
        command="scan",
        summary="Файл или каталог для сканирования (позиционный аргумент).",
        applied_by=("poison_check.cli:_collect_paths",),
        proven_by=(
            "tests.test_cli:test_scan_safe_pkl_exit_code_zero",
            "tests.test_cli:test_scan_nonexistent_file_exit_code_two",
        ),
    ),
    CliOptionSpec(
        option="--policy",
        command="scan",
        summary=(
            "Политика сканирования: встроенное имя или путь к YAML. "
            "Резолвится существующим механизмом PolicyLoader."
        ),
        applied_by=(
            "poison_check.cli:_load_policy",
            "poison_check.policies:PolicyLoader.load",
        ),
        proven_by=(
            "tests.test_cli_options:TestPolicyOption.test_banking_escalates_external_url",
            "tests.test_cli_options:TestPolicyOption.test_default_does_not_escalate",
            "tests.test_cli:test_scan_banking_policy_escalates_external_url",
        ),
    ),
    CliOptionSpec(
        option="--format",
        command="scan",
        summary="Формат отчёта: console / json / sarif / sbom / html / pdf.",
        applied_by=("poison_check.cli:OutputFormat", "poison_check.cli:scan"),
        proven_by=(
            "tests.test_cli_options:TestFormatOption.test_each_format_produces_its_own_shape",
            "tests.test_cli_options:TestFormatOption.test_invalid_format_is_clean_error",
        ),
    ),
    CliOptionSpec(
        option="--output",
        command="scan",
        summary="Путь для сохранения отчёта; работает для всех форматов.",
        applied_by=("poison_check.cli:_TEXT_FORMATS", "poison_check.cli:scan"),
        proven_by=(
            "tests.test_cli_options:TestOutputOption.test_output_writes_file",
            "tests.test_cli_options:TestOutputOption.test_console_format_writes_file",
            "tests.test_cli_options:TestOutputOption.test_unwritable_path_reports_error",
        ),
    ),
    CliOptionSpec(
        option="--recursive",
        command="scan",
        summary=(
            "Обход вложенных каталогов. По умолчанию выключен: без флага "
            "сканируется только верхний уровень."
        ),
        applied_by=("poison_check.cli:_collect_paths",),
        proven_by=(
            "tests.test_cli_options:TestRecursiveOption.test_recursive_finds_nested_file",
            "tests.test_cli_options:TestRecursiveOption.test_default_is_not_recursive",
            "tests.test_cli_options:TestRecursiveOption.test_no_recursive_is_explicit_default",
        ),
        aliases=("-r", "--no-recursive", "-R"),
    ),
    CliOptionSpec(
        option="--bundle-check",
        command="scan",
        summary=(
            "Проверка комплекта модели (код рядом с весами, MLS-BUNDLE-*). "
            "По умолчанию включена; --no-bundle-check полностью отключает "
            "(ни отчёта, ни гейта), имея приоритет над enabled_detectors политики."
        ),
        applied_by=("poison_check.cli:scan", "poison_check.scanner:Scanner.scan"),
        proven_by=(
            "tests.test_bundle_cli_flag:TestBundleCheckFlag."
            "test_with_flag_disabled_no_findings_and_exit_zero",
            "tests.test_bundle_cli_flag:TestBundleCheckFlag.test_default_is_enabled",
        ),
        aliases=("--no-bundle-check",),
    ),
    CliOptionSpec(
        option="--locale",
        command="scan",
        summary="Язык сообщений вывода: ru или en.",
        applied_by=("poison_check.i18n.loader:I18n.set_locale",),
        proven_by=("tests.test_cli_options:TestLocaleOption.test_locale_switches_language",),
    ),
    CliOptionSpec(
        option="--client",
        command="scan",
        summary="Заказчик аудита в шапке отчёта; выводится всеми форматами.",
        applied_by=(
            "poison_check.output.report_metadata:build_report_metadata",
            "poison_check.output.report_metadata:REPORT_METADATA_FIELDS",
        ),
        proven_by=(
            "tests.test_cli_options:TestReportMetadataOptions.test_client_present_in_all_formats",
            "tests.test_cli_options:TestReportMetadataOptions.test_empty_metadata_omitted",
        ),
    ),
    CliOptionSpec(
        option="--auditor",
        command="scan",
        summary="ФИО аудитора в шапке отчёта; выводится всеми форматами.",
        applied_by=(
            "poison_check.output.report_metadata:build_report_metadata",
            "poison_check.output.report_metadata:REPORT_METADATA_FIELDS",
        ),
        proven_by=(
            "tests.test_cli_options:TestReportMetadataOptions.test_auditor_present_in_all_formats",
            "tests.test_cli_options:TestReportMetadataOptions.test_empty_metadata_omitted",
        ),
    ),
    CliOptionSpec(
        option="--max-file-size",
        command="scan",
        summary="Лимит размера файла в ГБ; сильнее extra_rules.max_file_size_gb.",
        applied_by=("poison_check.cli:scan", "poison_check.cli:_scan_single_file"),
        proven_by=("tests.test_cli:test_scan_cli_flag_overrides_policy_limit",),
    ),
    CliOptionSpec(
        option="--no-emoji",
        command="scan",
        summary="ASCII-иконки вместо эмодзи в консольном выводе.",
        applied_by=("poison_check.output.console:ConsoleFormatter.__init__",),
        proven_by=("tests.test_cli_options:TestNoEmojiOption.test_no_emoji_uses_ascii_icons",),
    ),
    CliOptionSpec(
        option="--verbose",
        command="scan",
        summary="Показывает глобалы каждого файла в stderr (отладка FP).",
        applied_by=("poison_check.cli:_print_verbose_globals",),
        proven_by=("tests.test_cli_options:TestVerboseOption.test_verbose_lists_globals",),
        aliases=("-v",),
    ),
    CliOptionSpec(
        option="--audit-log",
        command="scan",
        summary="Путь к JSON Lines audit-логу прогонов.",
        applied_by=("poison_check.audit:write_audit_record",),
        proven_by=("tests.test_audit_log:test_cli_audit_log_flag_writes_record",),
    ),
)


def _key(command: str, option: str) -> str:
    """Строит человекочитаемый ключ ``"команда опция"``.

    Для параметров верхнего уровня (:data:`TOP_LEVEL`) имя команды опускается,
    чтобы ключ читался как реальная командная строка.
    """
    return f"{command} {option}".strip()


def _primary_option(param: click.Parameter) -> str:
    """Возвращает основную форму параметра для сопоставления с реестром.

    Для опции это первая длинная форма (``--policy``), для позиционного
    аргумента — его имя (``path``).
    """
    long_opts = [opt for opt in param.opts if opt.startswith("--")]
    if long_opts:
        return long_opts[0]
    return param.opts[0] if param.opts else param.name or ""


def iter_app_params(app: typer.Typer) -> list[tuple[str, click.Parameter]]:
    """Перечисляет ВСЕ параметры всех команд приложения.

    Источник истины — сам click-парсер, а не сигнатуры функций: проверять надо
    ровно то, что реально принимает командная строка.

    Параметры САМОЙ группы (например, гипотетический ``--version`` верхнего
    уровня) тоже попадают в выборку — под пустым именем команды
    :data:`TOP_LEVEL`. Иначе опция, добавленная не в подкоманду, а в
    приложение, обошла бы весь гейт стороной.

    :param app: Приложение typer.
    :return: Список пар ``(имя команды, параметр)``; служебный ``--help``
        пропускается — он добавляется click автоматически.
    """
    command = typer.main.get_command(app)
    pairs: list[tuple[str, click.Parameter]] = []

    commands: dict[str, click.Command]
    if isinstance(command, click.Group):
        commands = {TOP_LEVEL: command, **dict(command.commands)}
    else:  # pragma: no cover — приложение проекта всегда группа
        commands = {command.name or TOP_LEVEL: command}

    for name, sub in sorted(commands.items()):
        for param in sub.params:
            if param.name == "help":
                continue
            pairs.append((name, param))
    return pairs


def registered_option_keys(specs: tuple[CliOptionSpec, ...] = CLI_OPTIONS) -> set[str]:
    """Множество ключей ``"команда опция"`` из реестра.

    :param specs: Реестр опций (по умолчанию :data:`CLI_OPTIONS`).
    :return: Множество ключей вида ``"scan --policy"``.
    """
    return {_key(spec.command, spec.option) for spec in specs}


def unregistered_options(
    app: typer.Typer,
    specs: tuple[CliOptionSpec, ...] = CLI_OPTIONS,
) -> list[str]:
    """Параметры парсера, у которых НЕТ записи в реестре.

    Непустой список означает опцию, о применении и покрытии которой ничего не
    заявлено — ровно тот случай, когда она может молча ничего не делать.

    :param app: Приложение typer.
    :param specs: Реестр опций.
    :return: Отсортированный список ключей ``"команда опция"``.
    """
    known = registered_option_keys(specs)
    return sorted(
        key
        for key in (
            _key(cmd, _primary_option(param)) for cmd, param in iter_app_params(app)
        )
        if key not in known
    )


def stale_option_specs(
    app: typer.Typer,
    specs: tuple[CliOptionSpec, ...] = CLI_OPTIONS,
) -> list[str]:
    """Записи реестра, которым больше не соответствует ни один параметр.

    Защищает от обратного рассинхрона: опцию удалили или переименовали, а
    запись в реестре осталась и продолжает «доказывать» несуществующее.

    :param app: Приложение typer.
    :param specs: Реестр опций.
    :return: Отсортированный список ключей ``"команда опция"``.
    """
    actual = {_key(cmd, _primary_option(param)) for cmd, param in iter_app_params(app)}
    return sorted(key for key in registered_option_keys(specs) if key not in actual)


def options_missing_from_help(
    app: typer.Typer,
    help_of: Callable[[str], str],
) -> list[str]:
    """Формы опций, которые парсер принимает, но ``--help`` не показывает.

    Проверяются ВСЕ формы каждого параметра — и основная, и короткая, и
    отрицательная: показанная наполовину опция (как ``-r`` без ``--recursive``)
    остаётся ловушкой для пользователя. Скрытая опция (``hidden=True``) в help
    не попадает и потому обнаруживается тем же способом.

    :param app: Приложение typer.
    :param help_of: Функция, возвращающая текст ``--help`` по имени команды.
    :return: Отсортированный список ``"команда форма"``, которых нет в help.
    """
    missing: list[str] = []
    help_cache: dict[str, str] = {}

    for cmd, param in iter_app_params(app):
        if cmd not in help_cache:
            help_cache[cmd] = _normalize_help(help_of(cmd))
        text = help_cache[cmd]

        forms = list(param.opts) + list(param.secondary_opts)
        for form in forms:
            needle = form if form.startswith("-") else form.upper()
            if needle not in text:
                missing.append(_key(cmd, form))

    return sorted(missing)


def _normalize_help(text: str) -> str:
    """Убирает переносы, которыми rich разрывает длинные строки help.

    Rich переносит текст по ширине терминала и может разорвать саму форму
    опции; без нормализации проверка вхождения давала бы ложные срабатывания.
    """
    return " ".join(text.split())

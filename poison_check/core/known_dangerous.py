"""Общий список известных опасных globals.

Используется BlocklistDetector и AllowlistDetector для дедупликации.
Хранит только пары (module, name) без severity — severity определяется
в соответствующем детекторе на основе CVE-правил или hardcoded правил.
"""

from __future__ import annotations

KNOWN_DANGEROUS_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        # --- Выполнение системных команд (POSIX / Windows) ---
        ("os", "system"),
        ("os", "popen"),
        ("posix", "system"),  # Linux os.system через модуль posix
        ("posix", "popen"),  # Linux os.popen через модуль posix
        ("nt", "system"),  # Windows os.system через модуль nt
        # --- Py2-модуль commands: прямой аналог os.system ---
        # Модульный алиас commands→subprocess неприменим (см. комментарий
        # к _MODULE_ONLY_ALIASES ниже) — пары заносим в blocklist явно.
        ("commands", "getoutput"),
        ("commands", "getstatusoutput"),
        # --- subprocess: выполнение команды через shell ---
        ("subprocess", "getoutput"),
        ("subprocess", "getstatusoutput"),
        # --- Замена процесса (exec-семейство) ---
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "execvpe"),
        ("os", "execl"),
        ("os", "execle"),
        ("os", "execlp"),
        # --- Spawn-семейство ---
        ("os", "spawnl"),
        ("os", "spawnlp"),
        ("os", "spawnv"),
        ("os", "spawnve"),
        ("os", "spawnvp"),
        ("os", "spawnvpe"),
        ("os", "posix_spawn"),
        ("os", "posix_spawnp"),
        # --- Псевдотерминал ---
        ("pty", "spawn"),  # RCE через псевдотерминал (Linux)
        # --- subprocess ---
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "run"),
        ("subprocess", "check_output"),
        ("subprocess", "check_call"),
        # --- Python builtins с произвольным выполнением кода ---
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("builtins", "breakpoint"),  # вызывает sys.breakpointhook (PYTHONBREAKPOINT)
        # --- Интерактивный интерпретатор ---
        ("code", "interact"),  # REPL внутри процесса — произвольное выполнение кода
        # --- Динамический импорт ---
        ("importlib", "import_module"),
        # --- Косвенный импорт и рефлексия (indirect RCE chain) ---
        ("builtins", "__import__"),   # динамический импорт модулей
        ("builtins", "getattr"),      # получение атрибута объекта — ключевой элемент indirect-chain
        # --- Вторичная десериализация (pickle-in-pickle) ---
        ("marshal", "loads"),  # RCE через marshal внутри pickle
        ("pickle", "loads"),  # pickle внутри pickle
        ("dill", "loads"),  # dill внутри pickle
        # --- Сетевые подключения ---
        ("socket", "socket"),  # прямые TCP/UDP-соединения
    }
)

DANGEROUS_FUNCTION_NAMES: frozenset[str] = frozenset(
    {
        # Имена функций, опасные вне зависимости от модуля.
        # Используется AllowlistDetector для override trusted_prefix→HIGH.
        "system", "popen",
        "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp",
        "spawnl", "spawnlp", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "posix_spawn", "posix_spawnp",
        "Popen", "call", "run", "check_output", "check_call",
        "getoutput", "getstatusoutput",
        "eval", "exec", "compile", "breakpoint",
        "interact",
        "__import__", "getattr", "setattr", "delattr",
    }
)


# ---------------------------------------------------------------------------
# Нормализация Python 2 / legacy pickle-имён
# ---------------------------------------------------------------------------
# Pickle-поток может сериализовать функцию под именем модуля Python 2,
# либо потому что pickle сохраняет ``__module__`` объекта, либо потому что
# malware намеренно использует legacy-имена в надежде обойти сканеры,
# которые ловят только Py3 форму. Держим таблицу ``module → module`` —
# в текущих алиасах имена внутри совпадают. Функция ``normalize_global``
# принимает пару (module, name) и возвращает пару, чтобы будущие алиасы
# с расхождением по имени не требовали переписывать её сигнатуру.
_MODULE_ONLY_ALIASES: dict[str, str] = {
    "__builtin__": "builtins",   # Py2 builtin-модуль
    "copy_reg":    "copyreg",    # переименован в Py3
    "cStringIO":   "io",         # ускоренная реализация Py2, в Py3 нет
    "StringIO":    "io",         # Py2 модуль верхнего уровня → io.StringIO
    "Queue":       "queue",      # переименован в Py3
    "ConfigParser": "configparser",  # переименован в Py3
    "cPickle":     "pickle",     # ускоренная реализация Py2
    "thread":      "_thread",    # переименован в Py3, имена функций совпадают
}
# ВАЖНО: таблица module→module корректна только если ВСЕ имена внутри модуля
# совпадают с каноническими. Py2-модуль ``commands`` этому не удовлетворяет:
# getoutput/getstatusoutput перешли в ``subprocess``, но getstatus/mkarg/mk2arg
# аналога не имеют — алиас ``commands``→``subprocess`` порождал бы
# несуществующие канонические пары. Поэтому опасные пары ``commands.*``
# занесены в KNOWN_DANGEROUS_GLOBALS напрямую.


def normalize_global(module: str, name: str) -> tuple[str, str]:
    """Возвращает канонический (module, name) для сравнения с blocklist/allowlist.

    Не меняет входящую пару, если алиаса нет. Вызывающий код должен
    сохранять СЫРУЮ пару в location/message для forensics — нормализация
    применяется только на этапе поиска в правилах.
    """
    canonical_module = _MODULE_ONLY_ALIASES.get(module, module)
    return canonical_module, name


def is_py2_alias(module: str) -> bool:
    """True, если модуль — Python 2 / legacy алиас (для UI-подсказки)."""
    return module in _MODULE_ONLY_ALIASES

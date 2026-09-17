"""Резолвинг путей к data-директориям (``rules/`` и ``policies/``).

Пакет ищет data-директории в двух местах, в порядке приоритета:

1. **Внутри установленного пакета** — ``poison_check/_data/{rules,policies}``.
   Так они попадают на сервер при ``pip install`` (см. секцию
   ``[tool.hatch.build.targets.wheel.force-include]`` в ``pyproject.toml``).

2. **Рядом с корнем репозитория** — ``<repo>/rules`` и ``<repo>/policies``.
   Так они лежат при разработке из исходников без установки.

Разработке из редактируемой установки (``pip install -e .``) подходит второй
вариант, потому что force-include в этом режиме не выполняется.
"""

from __future__ import annotations

from pathlib import Path

_PKG_ROOT: Path = Path(__file__).parent
_REPO_ROOT: Path = _PKG_ROOT.parent


def _resolve(name: str) -> Path:
    """Возвращает первый существующий путь к data-директории с заданным именем.

    :param name: короткое имя (``"rules"`` или ``"policies"``).
    :return: путь к директории. Если ни один из кандидатов не существует,
        возвращает "installed"-путь — он попадёт в сообщение об ошибке
        как более информативный (пользователь увидит, куда именно
        pip install должен был это положить).
    """
    installed = _PKG_ROOT / "_data" / name
    if installed.is_dir():
        return installed
    dev = _REPO_ROOT / name
    if dev.is_dir():
        return dev
    return installed


RULES_DIR: Path = _resolve("rules")
POLICIES_DIR: Path = _resolve("policies")

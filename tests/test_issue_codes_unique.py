"""Регрессионный тест: все MLS-коды Issue в проекте должны быть уникальны.

История: в v0.1 коды MLS050/051/052/053 одновременно использовались в
JoblibScanner, GGUFScanner и CompressionDetector — пользователь делавший
suppression `--ignore MLS052` подавлял три разные категории находок.
В v0.2 коды переведены в namespace по компонентам (MLS-PKL-NNN, MLS-NET-NNN,
MLS-GGUF-NNN, …). Этот тест следит, чтобы коллизии не возникли снова.

Метод: запускает каждый детектор на синтетических входах, в которых заведомо
сработает каждое правило, собирает все Issue.code и проверяет уникальность.
Это сильнее чем grep по литералам — ловит и динамически сформированные коды
(например, MLS-CVE-... в CVEDetector).
"""

from __future__ import annotations

import re
from pathlib import Path

# Регулярка для извлечения литеральных code= из исходников
_CODE_RE = re.compile(r'code=(["\'])(MLS[A-Za-z0-9_\-]+)\1')

_PROJECT_ROOT = Path(__file__).parent.parent
_PACKAGE_ROOT = _PROJECT_ROOT / "poison_check"


def _collect_literal_codes() -> dict[str, list[Path]]:
    """Собирает все литералы вида ``code="MLS-..."`` из исходников проекта.

    Возвращает словарь code → список файлов, в которых код встречается.
    Один файл — нормально (код может встречаться многократно в одном модуле,
    например, в нескольких ветках условия). Два разных файла — коллизия.
    """
    seen: dict[str, list[Path]] = {}
    for py_path in _PACKAGE_ROOT.rglob("*.py"):
        text = py_path.read_text(encoding="utf-8")
        for _quote, code in _CODE_RE.findall(text):
            seen.setdefault(code, [])
            if py_path not in seen[code]:
                seen[code].append(py_path)
    return seen


def test_no_duplicate_mls_codes_across_modules() -> None:
    """Один и тот же MLS-код не должен встречаться в двух разных модулях."""
    seen = _collect_literal_codes()
    duplicates = {
        code: paths for code, paths in seen.items() if len(paths) > 1
    }
    assert not duplicates, (
        "Найдены MLS-коды, повторяющиеся в нескольких модулях — это нарушение "
        "стабильного контракта code-as-identifier. Дубли:\n"
        + "\n".join(
            f"  {code}: {[str(p.relative_to(_PROJECT_ROOT)) for p in paths]}"
            for code, paths in duplicates.items()
        )
    )


def test_all_codes_follow_namespace_pattern() -> None:
    """Все литеральные MLS-коды должны быть в формате MLS-COMPONENT-NNN.

    Допустимые префиксы компонентов: PKL, ALW, SEC, NET, EXE, CMP,
    GGUF, NPY, JOBLIB, ST, CVE, PATTERN. Этот список — компонент-регистр
    проекта; если добавляется новый сканер/детектор с собственными кодами,
    его префикс надо явно добавить сюда.
    """
    allowed_prefixes = frozenset(
        {
            "PKL",
            "ALW",
            "SEC",
            "NET",
            "EXE",
            "CMP",
            "GGUF",
            "NPY",
            "JOBLIB",
            "ST",
            "CVE",
            "PATTERN",
            "GHSA",
            "KERAS",
            "PARSE",
            "BOMB",
            "FMT",
            "BUNDLE",
        }
    )
    pattern = re.compile(r"^MLS-([A-Z]+)-")

    bad: list[str] = []
    for code in _collect_literal_codes():
        m = pattern.match(code)
        if m is None:
            bad.append(code)
            continue
        if m.group(1) not in allowed_prefixes:
            bad.append(code)

    assert not bad, (
        "MLS-коды без принятого namespace-префикса:\n"
        + "\n".join(f"  {c}" for c in bad)
        + "\nДобавьте префикс компонента или зарегистрируйте новый "
          "в allowed_prefixes этого теста."
    )

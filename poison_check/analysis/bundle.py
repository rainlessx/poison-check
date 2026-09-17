"""Анализ комплекта модели на уровне директории (код рядом с весами).

Единая точка, из которой и CLI (``cli.scan``), и Python API (``Scanner.scan``)
запускают directory-level проверку комплекта — благодаря этому оба пути ведут
себя одинаково (тот же инвариант, что у ``instantiate_detectors``).

Поток: :class:`BundleScanner` извлекает факты из директории → переданные
детекторы интерпретируют их → :class:`FileResult`, привязанный к КОРНЮ
директории (а не к отдельному файлу). Файлы, которыми владеет bundle-анализ
(``config.json`` и ``*.py``), исключаются из пофайлового цикла вызывающей
стороной через :func:`bundle_companion_files`: иначе каждый ``config.json`` и
каждый ``modeling.py`` давал бы «Неподдерживаемый формат» → ошибку разбора →
exit 2, маскируя результат самого bundle-анализа.
"""

from __future__ import annotations

from pathlib import Path

from poison_check.core.detector_base import BaseDetector
from poison_check.core.result import FileResult, MLContext, dedupe_issues
from poison_check.scanners.bundle_scanner import BundleScanner

#: Имена/суффиксы файлов, которыми владеет bundle-анализ. Исключаются из
#: пофайлового цикла при сканировании ДИРЕКТОРИИ (не отдельного файла), чтобы
#: companion-файлы модели не репортились как ошибки «Неподдерживаемый формат».
_COMPANION_NAMES: frozenset[str] = frozenset({"config.json"})
_COMPANION_SUFFIXES: frozenset[str] = frozenset({".py"})


def bundle_companion_files(files: list[Path]) -> set[Path]:
    """Возвращает companion-файлы комплекта (config.json, *.py) из списка файлов.

    Вызывающая сторона исключает их из пофайлового сканирования: их содержимое
    разбирает bundle-анализ, и отдельная ошибка «неподдерживаемый формат» по ним
    была бы ложной (это штатные части комплекта модели, а не цели сканера весов).
    """
    return {
        f
        for f in files
        if f.name in _COMPANION_NAMES or f.suffix in _COMPANION_SUFFIXES
    }


def analyze_bundle(
    root: Path, detectors: list[BaseDetector]
) -> FileResult | None:
    """Прогоняет bundle-анализ директории и возвращает FileResult или None.

    :param root: Корень директории-комплекта.
    :param detectors: Готовые экземпляры детекторов (те же, что в пофайловом
        цикле; реагирует только :class:`BundleCodeDetector`).
    :return: FileResult, привязанный к ``root``, если найдена хотя бы одна Issue;
        иначе None (директория не является комплектом либо кода/ссылок нет —
        обычная модель, шуметь не нужно).
    """
    raw_data = BundleScanner().scan(root)
    ctx = MLContext(framework="unknown", confidence=0.0, detected_patterns=[])

    all_issues = []
    for detector in detectors:
        all_issues.extend(detector.analyze(raw_data, ctx))
    all_issues = dedupe_issues(all_issues)

    if not all_issues:
        return None

    return FileResult(
        file_path=root,
        scanner_name=BundleScanner.name,
        issues=all_issues,
        error=None,
    )

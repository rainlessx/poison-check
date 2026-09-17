"""Загрузчик локализаций.

Локализация — first-class feature архитектуры. Сообщения загружаются из
YAML-файлов в каталоге poison_check/i18n/, доступ к ним — по dot-notation
ключам ("cli.scan_start").

Класс I18n реализован как singleton: в большинстве случаев в процессе
работы CLI используется одна локаль, и нет нужды передавать инстанс
повсюду. Для тестов и нестандартных сценариев singleton можно сбросить
методом ``reset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import yaml  # type: ignore[import-untyped]

_LOCALES_DIR = Path(__file__).parent
_FALLBACK_LOCALE = "en"


class I18n:
    """Загрузчик и доступ к локализованным сообщениям.

    Загружает основной {locale}.yaml. Если запрошенный ключ
    отсутствует — пытается найти его в en.yaml (fallback). Если ключа
    нет ни там, ни там — возвращает сам ключ как fallback-строку
    (никогда не падает с исключением).
    """

    _instance: ClassVar[I18n | None] = None

    def __init__(self, locale: str = "ru") -> None:
        """Инициализация загрузчика.

        :param locale: Код локали ("ru", "en"). Соответствующий файл
            poison_check/i18n/{locale}.yaml должен существовать. Если не
            существует — используется только fallback-словарь.
        """
        self.locale = locale
        self._messages: dict[str, Any] = self._load_yaml(locale)
        # Fallback-словарь грузим только если основная локаль — не fallback,
        # чтобы не дублировать одно и то же.
        if locale != _FALLBACK_LOCALE:
            self._fallback: dict[str, Any] = self._load_yaml(_FALLBACK_LOCALE)
        else:
            self._fallback = {}

    @staticmethod
    def _load_yaml(locale: str) -> dict[str, Any]:
        """Загружает YAML-файл локали. Возвращает пустой dict при ошибке.

        Обработка ошибок строгая: отсутствие файла или ошибка парсинга
        не приводит к падению — мы возвращаем пустой словарь и далее
        пользователь получит сами ключи в качестве сообщений.
        """
        path = _LOCALES_DIR / f"{locale}.yaml"
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    @staticmethod
    def _lookup(messages: dict[str, Any], key: str) -> str | None:
        """Ищет ключ в словаре по dot-notation. None, если не нашёл."""
        parts = key.split(".")
        node: Any = messages
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        if isinstance(node, str):
            return node
        return None

    def t(self, key: str, **kwargs: Any) -> str:
        """Возвращает локализованное сообщение по dot-notation ключу.

        :param key: Ключ вида "cli.scan_start".
        :param kwargs: Параметры для подстановки через str.format().
        :return: Локализованную строку. Если ключ не найден ни в основной
            локали, ни в fallback — возвращает сам ключ.
        """
        template = self._lookup(self._messages, key)
        if template is None:
            template = self._lookup(self._fallback, key)
        if template is None:
            return key
        if not kwargs:
            return template
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            # Если в шаблоне есть placeholder, для которого не передан
            # аргумент — возвращаем сырой шаблон, не падаем.
            return template

    @classmethod
    def get(cls) -> I18n:
        """Возвращает singleton-инстанс. Создаёт с локалью "ru" при первом вызове."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_locale(cls, locale: str) -> I18n:
        """Меняет локаль singleton'а. Возвращает новый инстанс."""
        cls._instance = cls(locale)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Сбрасывает singleton (полезно для тестов)."""
        cls._instance = None


def t(key: str, **kwargs: Any) -> str:
    """Сокращённый доступ к I18n.get().t()."""
    return I18n.get().t(key, **kwargs)

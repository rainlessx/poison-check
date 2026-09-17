"""Распаковка ZIP и TAR архивов в память без извлечения на диск."""

from __future__ import annotations

import logging
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import ClassVar

logger = logging.getLogger(__name__)


def _is_unsafe_archive_member_name(name: str) -> bool:
    """Возвращает True если имя члена архива выглядит как path-traversal атака.

    Проверяет следующие классы атак (расширенный набор по сравнению с v0.1):

    1. Абсолютные POSIX-пути: ``/etc/passwd``.
    2. Абсолютные Windows-пути: ``C:\\Windows\\System32`` или ``\\\\server\\share``.
    3. Любая компонента пути, равная ``..`` (классический traversal) — после
       нормализации через PurePosixPath и проверки каждой части. Это ловит
       варианты типа ``foo/../../../etc/passwd``, которые v0.1 ловил
       только наивной проверкой ``".." in name`` (давала ложные срабатывания
       на легитимных именах вида ``my..backup.bin``).
    4. Backslash где-то посередине пути — на Windows ZIP тоже использует
       прямые слэши, любой backslash подозрителен.
    5. NUL-байт — попытка обмануть downstream-парсер.

    Не ловит (намеренно): UTF-8 normalization атаки (``\\u2024``) — это требует
    полноценного Unicode-нормализатора, оставлено как known limitation.
    """
    if not name or "\x00" in name:
        return True

    # Backslash в любом месте — отказываем (ZIP-стандарт требует /).
    if "\\" in name:
        return True

    # Абсолютный POSIX-путь.
    if name.startswith("/"):
        return True

    # Абсолютный Windows-путь: 'C:foo' или 'C:/foo' или 'C:\\foo'.
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        return True

    # Нормализуем через PurePosixPath и проверяем каждую часть.
    # Это ловит 'a/../../etc/passwd' независимо от позиции '..'.
    pure = PurePosixPath(name)
    if pure.is_absolute():
        return True
    return any(part == ".." for part in pure.parts)


class ContainerError(Exception):
    """Ошибка при работе с контейнером: повреждённый архив или превышение лимита zip-bomb."""


class ContainerExtractor:
    """Извлекает содержимое ZIP и TAR архивов исключительно в память.

    Никогда не записывает файлы на диск.
    Защита от zip-bomb: суммарный объём данных ограничен MAX_EXTRACT_SIZE.
    """

    MAX_EXTRACT_SIZE: ClassVar[int] = 2 * 1024 * 1024 * 1024  # 2 ГБ суммарно
    MAX_MEMBER_SIZE: ClassVar[int] = 500 * 1024 * 1024  # 500 МБ на один член

    _ZIP_MAGIC: ClassVar[bytes] = b"\x50\x4b\x03\x04"
    # Сигнатура 7z-контейнера (6 байт). Используется nullifAI-обходом
    # (ReversingLabs, 2025): 7z-сжатый/битый pickle для обхода сканеров.
    _7Z_MAGIC: ClassVar[bytes] = b"7z\xbc\xaf\x27\x1c"

    @classmethod
    def is_zip(cls, path: Path) -> bool:
        """Проверяет, является ли файл ZIP-архивом по magic bytes."""
        try:
            with path.open("rb") as f:
                return f.read(4) == cls._ZIP_MAGIC
        except OSError:
            return False

    @classmethod
    def is_tar(cls, path: Path) -> bool:
        """Проверяет, является ли файл TAR-архивом (включая .tar.gz, .tar.bz2, .tar.xz)."""
        try:
            return tarfile.is_tarfile(str(path))
        except OSError:
            return False

    @classmethod
    def extract_zip_members(cls, path: Path) -> Iterator[tuple[str, bytes]]:
        """Итератор по членам ZIP-архива: (имя_файла, содержимое).

        Содержимое читается в память; на диск ничего не записывается.
        Директории пропускаются.

        Raises:
            ContainerError: при превышении MAX_EXTRACT_SIZE (zip-bomb) или повреждённом архиве.
        """
        total_bytes = 0
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue

                    # Защита от path traversal — расширенная проверка через
                    # _is_unsafe_archive_member_name (Windows-абсолютные пути,
                    # backslash, NUL-байт, любая компонента-двоеточие).
                    name = info.filename
                    if _is_unsafe_archive_member_name(name):
                        logger.warning("ZIP: пропуск подозрительного пути %r", name)
                        continue

                    # Баг: info.file_size берётся из central directory и может быть
                    # намеренно занижен атакующим («поддельный заголовок»).
                    # Поэтому мы НЕ добавляем info.file_size в total_bytes до чтения —
                    # реальный размер станет известен только после member_file.read().
                    with zf.open(info) as member_file:
                        data = member_file.read(cls.MAX_MEMBER_SIZE + 1)

                    if len(data) > cls.MAX_MEMBER_SIZE:
                        raise ContainerError(
                            f"Член архива {info.filename!r} превысил лимит "
                            f"{cls.MAX_MEMBER_SIZE // (1024 * 1024)} МБ"
                        )

                    total_bytes += len(data)
                    if total_bytes > cls.MAX_EXTRACT_SIZE:
                        raise ContainerError(
                            f"Превышен лимит извлечения {cls.MAX_EXTRACT_SIZE} байт "
                            "(защита от zip-bomb)"
                        )

                    yield info.filename, data
        except zipfile.BadZipFile as exc:
            raise ContainerError(f"Повреждённый ZIP-архив: {path}") from exc

    @classmethod
    def extract_tar_members(cls, path: Path) -> Iterator[tuple[str, bytes]]:
        """Итератор по членам TAR-архива: (имя_файла, содержимое).

        Содержимое читается в память; на диск ничего не записывается.
        Пропускает директории, символические ссылки и жёсткие ссылки.
        Пропускает члены с путями, содержащими '..' или начинающимися с '/' (path traversal).

        Защита (аудит #6): для каждого члена проверяются и ``name``, и
        ``linkname`` через ``_is_unsafe_archive_member_name``. Это закрывает
        вектор «ссылка вне архива» (CVE-2007-4559 family). Также при чтении
        используется ``filter="data"`` (Python ≥3.12) — стандартная защитная
        политика, отвергающая абсолютные пути, traversal и небезопасные типы.

        Размер каждого члена сравнивается с MAX_MEMBER_SIZE. Декомпрессированный
        размер контролируется по ``len(data)`` после чтения, а не по
        ``member.size`` из заголовка (атакующий может занизить заявленный размер).

        Raises:
            ContainerError: при превышении лимитов или повреждённом архиве.
        """
        total_bytes = 0
        try:
            with tarfile.open(str(path), "r:*") as tf:
                # Python 3.12+: устанавливаем безопасный фильтр распаковки
                # на уровне TarFile. На Python <3.12 атрибут отсутствует,
                # и мы полагаемся только на _is_unsafe_archive_member_name.
                if hasattr(tarfile, "data_filter"):
                    tf.extraction_filter = tarfile.data_filter

                for member in tf.getmembers():
                    if not member.isfile():
                        # Symlink/hardlink/dir/dev игнорируем целиком —
                        # tarfile.extractfile() для них всё равно вернул бы None
                        # или вредоносное содержимое.
                        continue

                    # Проверяем и name, и linkname — атакующий может положить
                    # обычный файл с linkname='/etc/passwd' (для hardlink-цели).
                    name = member.name
                    if _is_unsafe_archive_member_name(name):
                        logger.warning(
                            "TAR: пропуск подозрительного пути %r", name
                        )
                        continue
                    link_target = getattr(member, "linkname", "") or ""
                    if link_target and _is_unsafe_archive_member_name(link_target):
                        logger.warning(
                            "TAR: пропуск %r — подозрительный linkname %r",
                            name, link_target,
                        )
                        continue

                    # Заявленный размер — НЕ доверяем для аккумуляции, сначала
                    # проверим что он не превышает лимит на один член.
                    declared_size = max(0, int(member.size))
                    if declared_size > cls.MAX_MEMBER_SIZE:
                        raise ContainerError(
                            f"Член TAR-архива {name!r} превышает лимит "
                            f"{cls.MAX_MEMBER_SIZE // (1024 * 1024)} МБ "
                            f"(заявленный размер {declared_size} байт)"
                        )

                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    # Читаем не больше MAX_MEMBER_SIZE+1, чтобы поймать
                    # ситуацию занижения size в заголовке.
                    data = extracted.read(cls.MAX_MEMBER_SIZE + 1)
                    if len(data) > cls.MAX_MEMBER_SIZE:
                        raise ContainerError(
                            f"Член TAR-архива {name!r} превысил лимит "
                            f"{cls.MAX_MEMBER_SIZE // (1024 * 1024)} МБ "
                            f"(фактический размер > заявленного)"
                        )

                    total_bytes += len(data)
                    if total_bytes > cls.MAX_EXTRACT_SIZE:
                        raise ContainerError(
                            f"Превышен лимит извлечения {cls.MAX_EXTRACT_SIZE} байт "
                            "(защита от zip-bomb)"
                        )

                    yield name, data
        except tarfile.TarError as exc:
            raise ContainerError(f"Повреждённый TAR-архив: {path}") from exc

    @classmethod
    def list_zip_members(cls, path: Path) -> list[str]:
        """Возвращает список имён файлов в ZIP без извлечения содержимого.

        Директории в список не включаются.

        Raises:
            ContainerError: при повреждённом архиве.
        """
        try:
            with zipfile.ZipFile(path, "r") as zf:
                return [info.filename for info in zf.infolist() if not info.is_dir()]
        except zipfile.BadZipFile as exc:
            raise ContainerError(f"Повреждённый ZIP-архив: {path}") from exc

    # ------------------------------------------------------------------
    # 7z (опциональная зависимость py7zr) — nullifAI-обход (ReversingLabs 2025)
    # ------------------------------------------------------------------

    @classmethod
    def is_7z(cls, path: Path) -> bool:
        """Проверяет, является ли файл 7z-архивом по magic bytes."""
        try:
            with path.open("rb") as f:
                return f.read(len(cls._7Z_MAGIC)) == cls._7Z_MAGIC
        except OSError:
            return False

    @classmethod
    def extract_7z_members(cls, path: Path) -> Iterator[tuple[str, bytes]]:
        """Итератор по членам 7z-архива: (имя_файла, содержимое).

        7z поддержан как ОПЦИОНАЛЬНАЯ зависимость (py7zr): часть вредоносных
        моделей (nullifAI, ReversingLabs 2025) прячут pickle в 7z-контейнере,
        чтобы обойти сканеры, знающие только zip/tar. При отсутствии py7zr —
        graceful degradation: бросаем :class:`ContainerError` (не ImportError),
        чтобы вызывающий сканер мог зафиксировать факт и эмитить Issue INFO о
        зависимости (как joblib при отсутствии lz4/zstd).

        Применяются те же лимиты, что для zip/tar: MAX_MEMBER_SIZE на член,
        MAX_EXTRACT_SIZE суммарно, и проверка path-traversal через
        ``_is_unsafe_archive_member_name``. Директории пропускаются.

        Raises:
            ContainerError: при отсутствии py7zr, повреждённом архиве или
                превышении лимитов (защита от decompression bomb).
        """
        try:
            import py7zr
        except ImportError as exc:
            raise ContainerError(
                "Файл сжат 7z, но библиотека py7zr не установлена. "
                "Установите: pip install py7zr (или poison-check[7z])."
            ) from exc

        total_bytes = 0
        try:
            with py7zr.SevenZipFile(path, "r") as archive:
                # Предпроверка заявленных размеров до чтения в RAM (защита от bomb).
                for info in archive.list():
                    if getattr(info, "is_directory", False):
                        continue
                    declared = int(getattr(info, "uncompressed", 0) or 0)
                    if declared > cls.MAX_MEMBER_SIZE:
                        raise ContainerError(
                            f"Член 7z-архива {getattr(info, 'filename', '?')!r} "
                            f"превышает лимит {cls.MAX_MEMBER_SIZE // (1024 * 1024)} МБ "
                            f"(заявленный размер {declared} байт)"
                        )
                # py7zr требует reset() перед повторным чтением после list().
                archive.reset()
                extracted = archive.readall()

            for name, stream in extracted.items():
                if _is_unsafe_archive_member_name(name):
                    logger.warning("7z: пропуск подозрительного пути %r", name)
                    continue
                data = stream.read()
                if len(data) > cls.MAX_MEMBER_SIZE:
                    raise ContainerError(
                        f"Член 7z-архива {name!r} превысил лимит "
                        f"{cls.MAX_MEMBER_SIZE // (1024 * 1024)} МБ"
                    )
                total_bytes += len(data)
                if total_bytes > cls.MAX_EXTRACT_SIZE:
                    raise ContainerError(
                        f"Превышен лимит извлечения {cls.MAX_EXTRACT_SIZE} байт "
                        "(защита от zip-bomb)"
                    )
                yield name, data
        except ContainerError:
            raise
        except Exception as exc:  # noqa: BLE001 — py7zr бросает разнотипные ошибки
            raise ContainerError(f"Повреждённый 7z-архив {path}: {exc}") from exc

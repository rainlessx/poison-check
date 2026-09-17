"""Тесты для poison_check/core/container.py."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from poison_check.core.container import ContainerError, ContainerExtractor


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _make_zip(tmp_path: Path, files: dict[str, bytes], *, add_dir: bool = False) -> Path:
    """Создаёт ZIP-архив из словаря {имя: содержимое} в tmp_path."""
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        if add_dir:
            dir_info = zipfile.ZipInfo("subdir/")
            zf.writestr(dir_info, "")
        for name, content in files.items():
            zf.writestr(name, content)
    return zip_path


def _make_tar(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """Создаёт TAR-архив из словаря {имя: содержимое} в tmp_path."""
    tar_path = tmp_path / "archive.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return tar_path


# ---------------------------------------------------------------------------
# is_zip
# ---------------------------------------------------------------------------


def test_is_zip_true_for_zip_file(tmp_path: Path) -> None:
    """is_zip возвращает True для корректного ZIP-файла."""
    zip_path = _make_zip(tmp_path, {"a.txt": b"hello"})
    assert ContainerExtractor.is_zip(zip_path) is True


def test_is_zip_false_for_non_zip(tmp_path: Path) -> None:
    """is_zip возвращает False для не-ZIP файла."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00.")
    assert ContainerExtractor.is_zip(f) is False


def test_is_zip_false_for_missing_file(tmp_path: Path) -> None:
    """is_zip возвращает False, если файл не существует."""
    assert ContainerExtractor.is_zip(tmp_path / "nonexistent.zip") is False


# ---------------------------------------------------------------------------
# is_tar
# ---------------------------------------------------------------------------


def test_is_tar_true_for_tar_file(tmp_path: Path) -> None:
    """is_tar возвращает True для корректного TAR-файла."""
    tar_path = _make_tar(tmp_path, {"data.pkl": b"\x80\x04."})
    assert ContainerExtractor.is_tar(tar_path) is True


def test_is_tar_false_for_zip_file(tmp_path: Path) -> None:
    """is_tar возвращает False для ZIP-файла."""
    zip_path = _make_zip(tmp_path, {"a.txt": b"hello"})
    assert ContainerExtractor.is_tar(zip_path) is False


def test_is_tar_false_for_missing_file(tmp_path: Path) -> None:
    """is_tar возвращает False, если файл не существует."""
    assert ContainerExtractor.is_tar(tmp_path / "nonexistent.tar") is False


# ---------------------------------------------------------------------------
# extract_zip_members
# ---------------------------------------------------------------------------


def test_extract_zip_members_yields_files(tmp_path: Path) -> None:
    """extract_zip_members возвращает правильные имена и содержимое."""
    content = b"\x80\x04\x95\x00."
    zip_path = _make_zip(tmp_path, {"model/data.pkl": content, "config.json": b"{}"})
    members = dict(ContainerExtractor.extract_zip_members(zip_path))
    assert members["model/data.pkl"] == content
    assert members["config.json"] == b"{}"


def test_extract_zip_members_skips_directories(tmp_path: Path) -> None:
    """extract_zip_members не включает директории в результат."""
    zip_path = _make_zip(tmp_path, {"file.txt": b"x"}, add_dir=True)
    members = dict(ContainerExtractor.extract_zip_members(zip_path))
    assert "subdir/" not in members
    assert "file.txt" in members


def test_extract_zip_members_zip_bomb_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """extract_zip_members поднимает ContainerError при превышении MAX_EXTRACT_SIZE."""
    monkeypatch.setattr(ContainerExtractor, "MAX_EXTRACT_SIZE", 10)
    zip_path = _make_zip(tmp_path, {"big.bin": b"x" * 100})
    with pytest.raises(ContainerError, match="zip-bomb"):
        list(ContainerExtractor.extract_zip_members(zip_path))


def test_extract_zip_members_corrupt_raises(tmp_path: Path) -> None:
    """extract_zip_members поднимает ContainerError для повреждённого архива."""
    corrupt = tmp_path / "bad.zip"
    corrupt.write_bytes(b"not a zip at all")
    with pytest.raises(ContainerError, match="Повреждённый ZIP-архив"):
        list(ContainerExtractor.extract_zip_members(corrupt))


# ---------------------------------------------------------------------------
# extract_tar_members
# ---------------------------------------------------------------------------


def test_extract_tar_members_yields_files(tmp_path: Path) -> None:
    """extract_tar_members возвращает правильные имена и содержимое."""
    content = b"model weights"
    tar_path = _make_tar(tmp_path, {"model.pkl": content, "cfg.json": b"{}"})
    members = dict(ContainerExtractor.extract_tar_members(tar_path))
    assert members["model.pkl"] == content
    assert members["cfg.json"] == b"{}"


def test_extract_tar_members_zip_bomb_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """extract_tar_members поднимает ContainerError при превышении MAX_EXTRACT_SIZE."""
    monkeypatch.setattr(ContainerExtractor, "MAX_EXTRACT_SIZE", 10)
    tar_path = _make_tar(tmp_path, {"big.bin": b"x" * 100})
    with pytest.raises(ContainerError, match="zip-bomb"):
        list(ContainerExtractor.extract_tar_members(tar_path))


def test_extract_tar_members_skips_path_traversal(tmp_path: Path) -> None:
    """extract_tar_members пропускает члены с path traversal в имени."""
    tar_path = tmp_path / "traversal.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        safe_info = tarfile.TarInfo(name="safe.txt")
        safe_data = b"safe"
        safe_info.size = len(safe_data)
        tf.addfile(safe_info, io.BytesIO(safe_data))

        evil_info = tarfile.TarInfo(name="../etc/passwd")
        evil_data = b"evil"
        evil_info.size = len(evil_data)
        tf.addfile(evil_info, io.BytesIO(evil_data))

    members = dict(ContainerExtractor.extract_tar_members(tar_path))
    assert "safe.txt" in members
    assert "../etc/passwd" not in members


def test_extract_tar_members_skips_unsafe_linkname(tmp_path: Path) -> None:
    """Регрессия аудита #6: член с подозрительным linkname отбрасывается.

    Атакующий может положить обычный файл (member.isfile() == True), но с
    linkname='/etc/passwd' для hardlink-цели. Раньше мы проверяли только name,
    теперь — и linkname.
    """
    tar_path = tmp_path / "evil_link.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        # Безопасный файл
        ok_info = tarfile.TarInfo(name="safe.txt")
        ok_info.size = 4
        tf.addfile(ok_info, io.BytesIO(b"safe"))

        # «Файл» с подозрительным linkname (hardlink-конструкция)
        evil_info = tarfile.TarInfo(name="link_to_passwd")
        evil_info.type = tarfile.REGTYPE
        evil_info.linkname = "/etc/passwd"
        evil_info.size = 4
        tf.addfile(evil_info, io.BytesIO(b"evil"))

    members = dict(ContainerExtractor.extract_tar_members(tar_path))
    assert "safe.txt" in members
    assert "link_to_passwd" not in members, (
        "Член с абсолютным linkname должен быть отброшен"
    )


def test_extract_tar_members_oversize_member_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Член, превышающий MAX_MEMBER_SIZE, отвергается до полного чтения."""
    monkeypatch.setattr(ContainerExtractor, "MAX_MEMBER_SIZE", 16)
    monkeypatch.setattr(ContainerExtractor, "MAX_EXTRACT_SIZE", 10_000)

    tar_path = tmp_path / "big_member.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        info = tarfile.TarInfo(name="big.bin")
        payload = b"x" * 1000
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises(ContainerError, match="MAX_MEMBER_SIZE|превышает лимит"):
        list(ContainerExtractor.extract_tar_members(tar_path))


# ---------------------------------------------------------------------------
# list_zip_members
# ---------------------------------------------------------------------------


def test_list_zip_members_returns_filenames(tmp_path: Path) -> None:
    """list_zip_members возвращает список имён файлов без директорий."""
    zip_path = _make_zip(
        tmp_path,
        {"data/model.pkl": b"\x80\x04.", "meta.json": b"{}"},
        add_dir=True,
    )
    members = ContainerExtractor.list_zip_members(zip_path)
    assert "data/model.pkl" in members
    assert "meta.json" in members
    assert "subdir/" not in members
    assert len(members) == 2


def test_list_zip_members_corrupt_raises(tmp_path: Path) -> None:
    """list_zip_members поднимает ContainerError для повреждённого архива."""
    corrupt = tmp_path / "bad.zip"
    corrupt.write_bytes(b"garbage data")
    with pytest.raises(ContainerError, match="Повреждённый ZIP-архив"):
        ContainerExtractor.list_zip_members(corrupt)


# ---------------------------------------------------------------------------
# Защита от path traversal в ZIP
# ---------------------------------------------------------------------------


def test_zip_path_traversal_rejected(tmp_path: Path) -> None:
    """extract_zip_members пропускает членов ZIP с path traversal в имени.

    ZIP central directory может содержать произвольные имена — в отличие от TAR,
    стандартная библиотека не нормализует их. Атакующий создаёт архив с членом
    "../../evil.py", который при наивной распаковке записал бы файл за пределами
    целевой директории.
    """
    zip_path = tmp_path / "traversal.zip"

    # Создаём ZIP вручную через ZipInfo, чтобы задать произвольное имя члена
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Безопасный файл — должен пройти
        zf.writestr("safe/model.pkl", b"\x80\x04.")
        # Path traversal через ".." — должен быть отброшен
        info_dotdot = zipfile.ZipInfo("../../evil.py")
        zf.writestr(info_dotdot, b"malicious content")
        # Абсолютный путь — должен быть отброшен
        info_abs = zipfile.ZipInfo("/etc/passwd")
        zf.writestr(info_abs, b"root:x:0:0")
        # Backslash traversal — должен быть отброшен
        info_bs = zipfile.ZipInfo("..\\windows\\evil.dll")
        zf.writestr(info_bs, b"pe payload")

    members = dict(ContainerExtractor.extract_zip_members(zip_path))

    assert "safe/model.pkl" in members
    assert "../../evil.py" not in members
    assert "/etc/passwd" not in members
    assert "..\\windows\\evil.dll" not in members


def test_zip_traversal_extended_attack_variants(tmp_path: Path) -> None:
    """Расширенный набор path-traversal атак: Windows-абсолютные, traversal в середине, NUL.

    Регрессия v0.2: v0.1 использовал только проверку ``".." in name``,
    что давало false negative на Windows-абсолютных путях ('C:/foo')
    и false positive на легитимных именах ('my..backup.bin').
    """
    zip_path = tmp_path / "extended_traversal.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Легитимный файл — должен пройти, несмотря на '..' в имени
        zf.writestr("my..backup.bin", b"\x80\x04.")
        zf.writestr("safe/model.pkl", b"\x80\x04.")
        # Windows-абсолютный путь
        zf.writestr(zipfile.ZipInfo("C:/Windows/System32/cmd.exe"), b"pe")
        # UNC путь
        zf.writestr(zipfile.ZipInfo("\\\\server\\share\\evil"), b"x")
        # Traversal в середине пути
        zf.writestr(zipfile.ZipInfo("foo/bar/../../../../etc/shadow"), b"x")
        # NUL-байт — попытка обмануть downstream-парсер
        zf.writestr(zipfile.ZipInfo("safe\x00.pkl"), b"x")

    members = dict(ContainerExtractor.extract_zip_members(zip_path))

    # Легитимные — должны быть
    assert "my..backup.bin" in members, (
        "'..' как часть имени файла без separator-ов не должен блокироваться: "
        "это легитимное имя файла."
    )
    assert "safe/model.pkl" in members

    # Атаки — все отброшены
    assert "C:/Windows/System32/cmd.exe" not in members
    assert "\\\\server\\share\\evil" not in members
    assert not any("etc/shadow" in m for m in members)
    assert not any("\x00" in m for m in members)


# ---------------------------------------------------------------------------
# ZIP bomb с поддельным заголовком (занижен file_size в central directory)
# ---------------------------------------------------------------------------


def test_zip_fake_header_bomb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """extract_zip_members поднимает ContainerError когда реальный размер члена
    превышает MAX_MEMBER_SIZE, даже если info.file_size в заголовке занижен.

    Суть атаки: ZIP central directory позволяет указать произвольный file_size.
    Атакующий может создать архив с file_size=100 в заголовке, но реальный
    поток данных (особенно при высоком deflate-ratio) может быть в гигабайты.
    Прежний код доверял info.file_size для проверки zip-bomb; новый код читает
    реальные байты через member_file.read(MAX_MEMBER_SIZE + 1) и проверяет
    len(data) — тем самым не полагаясь на заголовок.

    Тест снижает MAX_MEMBER_SIZE до 10 байт и помещает в архив 600 байт данных,
    имитируя пропорцию 10 МБ vs 600 МБ в боевом сценарии.

    Примечание о конструкции теста: Python's zipfile при read(n) с n < реального
    размера члена честно возвращает только n байт (не CRC-ошибку), поскольку
    неполное чтение допустимо. Поэтому мы не патчим infolist (это нарушает CRC),
    а просто выставляем лимит ниже размера данных — read(11) из 600-байтного
    члена вернёт 11 байт, чего достаточно чтобы сработал check len(data) > 10.
    """
    monkeypatch.setattr(ContainerExtractor, "MAX_MEMBER_SIZE", 10)

    # Реальные данные (600 байт) > лимит (10 байт)
    # Используем ZIP_STORED чтобы данные в архиве не сжимались —
    # ratio 1:1, но проверяемый код-путь тот же, что при deflate 1:1000.
    real_data = b"X" * 600
    zip_path = tmp_path / "fake_header.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("payload.bin", real_data)

    # read(MAX_MEMBER_SIZE + 1) = read(11) вернёт 11 байт → len(11) > 10 → ContainerError
    with pytest.raises(ContainerError, match="превысил лимит"):
        list(ContainerExtractor.extract_zip_members(zip_path))


# ---------------------------------------------------------------------------
# ContainerError
# ---------------------------------------------------------------------------


def test_container_error_is_exception() -> None:
    """ContainerError является подклассом Exception."""
    err = ContainerError("тест")
    assert isinstance(err, Exception)
    assert str(err) == "тест"

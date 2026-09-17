"""Детектор встроенных исполняемых файлов в ML-файлах.

Ищет сигнатуры PE/ELF/Mach-O в:
- raw_data.embedded_bytes — заранее найденные сканером сигнатуры (основной путь)
- raw_data.raw_content_sample — первые N байт файла (fallback для форматов
  без embedded_bytes)

Наличие исполняемого файла внутри ML-формата является крайне подозрительным
и, как правило, указывает на вредоносный payload.

Примечание: поиск по raw_content_sample также применяет структурную валидацию
PE/ELF (см. _validate_pe / _validate_elf), чтобы исключить FP из тензорных данных.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from poison_check.core.detector_base import BaseDetector
from poison_check.core.executable_signatures import _validate_elf, _validate_pe
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Severity,
)
from poison_check.core.scanner_base import RawScanData

logger = logging.getLogger(__name__)

# Сигнатуры исполняемых форматов: magic bytes → (тип, описание)
_SIGNATURES: dict[bytes, tuple[str, str]] = {
    b"MZ": ("PE", "Windows PE-исполняемый файл"),
    b"\x7fELF": ("ELF", "Linux ELF-исполняемый файл"),
    b"\xfe\xed\xfa\xce": ("Mach-O", "macOS Mach-O 32-bit"),
    b"\xfe\xed\xfa\xcf": ("Mach-O", "macOS Mach-O 64-bit"),
    b"\xca\xfe\xba\xbe": ("Mach-O", "macOS Mach-O Universal Binary"),
    b"\xce\xfa\xed\xfe": ("Mach-O", "macOS Mach-O 32-bit (little-endian)"),
    b"\xcf\xfa\xed\xfe": ("Mach-O", "macOS Mach-O 64-bit (little-endian)"),
}

# Минимальный размер сигнатуры (для ограничения поиска в sample)
_MIN_SIG_LEN = min(len(sig) for sig in _SIGNATURES)
# Максимальный размер сигнатуры (для проверки в sample)
_MAX_SIG_LEN = max(len(sig) for sig in _SIGNATURES)


@DetectorRegistry.register
class ExecutableDetector(BaseDetector):
    """Детектор встроенных исполняемых файлов (PE/ELF/Mach-O) в ML-файлах.

    Проверяет два источника:
    1. raw_data.embedded_bytes — список EmbeddedSignature, найденных сканером.
       Каждая запись уже содержит тип и смещение.
    2. raw_data.raw_content_sample — первые 4096 байт файла (сырой образец).
       Ищет сигнатуры напрямую для форматов, где сканер не извлекает embedded_bytes.

    Находка PE/ELF/Mach-O внутри ML-файла → Issue CRITICAL.
    """

    name: ClassVar[str] = "executable"
    description: ClassVar[str] = "Детектор встроенных исполняемых файлов в ML-файлах"
    severity_range: ClassVar[tuple[Severity, Severity]] = (Severity.CRITICAL, Severity.CRITICAL)

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Ищет сигнатуры исполняемых файлов в embedded_bytes и raw_content_sample.

        Алгоритм:
        1. Проверяет raw_data.embedded_bytes (уже распознанные сканером сигнатуры).
        2. Проверяет raw_data.raw_content_sample на наличие сигнатур в произвольных позициях.
        3. Дедупликация: не создаёт duplicate Issue для одной и той же позиции.
        """
        if raw_data.error is not None:
            return []

        issues: list[Issue] = []
        # Смещения, по которым уже создан Issue (чтобы не дублировать)
        reported_offsets: set[int] = set()

        # --- Источник 1: embedded_bytes (от сканера) ---
        if raw_data.embedded_bytes:
            for emb in raw_data.embedded_bytes:
                if emb.offset in reported_offsets:
                    continue
                reported_offsets.add(emb.offset)
                # Ищем описание по типу из EmbeddedSignature
                description = _get_description_for_type(emb.signature_type)
                issues.append(
                    _make_executable_issue(
                        raw_data=raw_data,
                        exec_type=emb.signature_type,
                        description=description,
                        offset=emb.offset,
                        source="embedded_bytes",
                    )
                )

        # --- Источник 2: raw_content_sample ---
        if raw_data.raw_content_sample:
            sample = raw_data.raw_content_sample
            for offset, (exec_type, description) in _scan_bytes_for_signatures(sample):
                if offset in reported_offsets:
                    continue
                reported_offsets.add(offset)
                issues.append(
                    _make_executable_issue(
                        raw_data=raw_data,
                        exec_type=exec_type,
                        description=description,
                        offset=offset,
                        source="raw_content_sample",
                    )
                )

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _scan_bytes_for_signatures(
    data: bytes,
) -> list[tuple[int, tuple[str, str]]]:
    """Ищет сигнатуры исполняемых файлов в байтовой строке.

    Возвращает список (offset, (тип, описание)) для каждого совпадения.
    Поиск перебирает все позиции — не только начало буфера.

    Применяет структурную валидацию для PE (MZ → e_lfanew → PE\\x00\\x00)
    и ELF (\\x7fELF → ei_class/ei_data/ei_version), чтобы исключить FP
    на случайных бинарных данных.
    """
    results: list[tuple[int, tuple[str, str]]] = []
    data_len = len(data)

    for sig, info in _SIGNATURES.items():
        sig_type = info[0]
        sig_len = len(sig)
        start = 0
        while start <= data_len - sig_len:
            pos = data.find(sig, start)
            if pos == -1:
                break
            # Структурная валидация для коротких magic bytes
            if sig_type == "PE" and not _validate_pe(data, pos):
                start = pos + 1
                continue
            if sig_type == "ELF" and not _validate_elf(data, pos):
                start = pos + 1
                continue
            results.append((pos, info))
            start = pos + 1  # не пропускаем перекрывающиеся совпадения

    # Сортируем по offset для детерминированного порядка
    results.sort(key=lambda x: x[0])
    return results


def _get_description_for_type(exec_type: str) -> str:
    """Возвращает описание для известного типа исполняемого файла.

    Используется когда сканер уже определил тип через embedded_bytes.
    """
    # Ищем первое совпадение по типу
    for _sig, (t, desc) in _SIGNATURES.items():
        if t == exec_type:
            return desc
    return f"Исполняемый файл типа {exec_type}"


def _make_executable_issue(
    raw_data: RawScanData,
    exec_type: str,
    description: str,
    offset: int,
    source: str,
) -> Issue:
    """Создаёт Issue CRITICAL для найденного исполняемого файла (MLS040)."""
    return Issue(
        code="MLS-EXE-001",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        message=f"Обнаружен встроенный исполняемый файл: {description}",
        location=f"{raw_data.file_path} (offset {offset})",
        details={
            "executable_type": exec_type,
            "description": description,
            "offset": offset,
            "detection_source": source,
        },
        why=(
            f"Наличие {exec_type}-исполняемого файла внутри ML-модели является "
            "крайне подозрительным. Это может быть вредоносный payload, "
            "который выполнится при загрузке или обработке файла."
        ),
        remediation=(
            "Немедленно прекратите использование этой модели. "
            "Проверьте источник файла. Если модель получена из недоверенного источника — "
            "удалите её и сообщите о находке в службу безопасности."
        ),
        compliance_tags=[
            "owasp-ml:ml03",
            "owasp-ml:ml10",
            "fstec:ubi-067",
            "fstec:ubi-068",
            "fstec:ubi-162",
            "gost:56939-2024:5.3",
            "gost:56939-2024:5.4",
            "gost:56939-2024:6.1",
            "gost:56939-2024:6.3",
        ],
    )

"""Детектор политики формата: MLS-FMT-001 и MLS-FMT-002.

Переводит факты о формате, собранные слоем сканеров
(:mod:`poison_check.scanners.format_facts`), в находки — но только если этого
требует политика. Оба правила выключены по умолчанию и включаются ключами
``extra_rules`` через тот же механизм, что и ``no_external_urls`` /
``no_unverified_files``: :func:`poison_check.policies.detector_kwargs_for`
передаёт флаги в конструктор.

**MLS-FMT-001 — ``strict_format_detection``.** Расширению файла доверять
нельзя. Если содержимое говорит одно, а имя обещает другое — это находка.
Вектор атаки прямой: файл называется ``model.safetensors``, потребитель видит
«безопасный формат» и загружает его через небезопасный путь, а внутри pickle.
Severity ВЫСОКИЙ: расхождение имени и содержимого не бывает случайным у
инструментов сериализации — все они пишут своё расширение сами. Это ещё не
доказанный payload (поэтому не CRITICAL), но достаточное основание не пускать
файл дальше: в banking/government (``fail_on_severity: high``) находка валит
гейт, что и есть обещание строгой политики. Confidence ВЫСОКАЯ — факт
несовпадения получен из прочитанных байт, а не из эвристики.

**MLS-FMT-002 — ``require_safetensors``.** В строгом контуре допустимы только
форматы, не исполняющие код при загрузке. Находка эмитится по САМОМУ ФАКТУ
формата, даже если в содержимом ничего вредоносного не нашли: чистый сегодня
pickle остаётся исполняемой программой, и его безопасность зависит от того,
всё ли увидел анализатор. Severity ВЫСОКИЙ, и это осознанно не CRITICAL:
CRITICAL в проекте означает найденный механизм RCE (os.system + REDUCE), здесь
же речь про недопустимый формат, а не про обнаруженный эксплойт. MEDIUM был бы
хуже, чем ничего: banking и government валят гейт с HIGH, и на MEDIUM правило
«запрещаем небезопасные форматы» снова стало бы декларацией. Confidence
ВЫСОКАЯ (не CERTAIN): формат определён достоверно, но утверждение «этот файл
опасен» здесь категориальное, а не доказанное содержимым.

Обе находки — обычные находки политики, а не факты уровня файла: они проходят
``severity_threshold`` как все прочие (и при HIGH никогда не уходят под порог,
см. :data:`poison_check.output.severity_threshold.NEVER_DEMOTED_SEVERITY`), но
в списке «неподавляемых» им делать нечего — файл при них проверен полностью.

Взаимоподавления с находками по содержимому нет: MLS-FMT-002 на pickle с
``os.system`` эмитится ВМЕСТЕ с MLS-PATTERN-OS-SYSTEM. Это разные утверждения —
«формат недопустим» и «в файле найден вызов команды ОС», и терять любое из них
нельзя.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from poison_check.core.detector_base import BaseDetector
from poison_check.core.registry import DetectorRegistry
from poison_check.core.result import (
    Confidence,
    Issue,
    MLContext,
    Severity,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.format_facts import (
    FormatFacts,
    FormatSafety,
    format_facts_of,
)

logger = logging.getLogger(__name__)

#: Namespace FMT закреплён за политикой формата (см. test_issue_codes_unique).
_ISSUE_CODE_MISMATCH: str = "MLS-FMT-001"
_ISSUE_CODE_CODE_BEARING: str = "MLS-FMT-002"


@DetectorRegistry.register
class FormatPolicyDetector(BaseDetector):
    """Эмитит находки по политике формата (MLS-FMT-001 / MLS-FMT-002).

    Оба правила выключены по умолчанию: в default-политике поведение остаётся
    прежним, ни одной новой находки не появляется.

    :param strict_format_detection: Включает MLS-FMT-001 — расхождение
        «расширение ↔ фактический формат».
    :param require_safetensors: Включает MLS-FMT-002 — недопустимость
        форматов, исполняющих код при загрузке.
    """

    name: ClassVar[str] = "format_policy"
    description: ClassVar[str] = (
        "Детектор политики формата: подмена формата и code-bearing форматы"
    )
    severity_range: ClassVar[tuple[Severity, Severity]] = (
        Severity.HIGH,
        Severity.HIGH,
    )

    def __init__(
        self,
        *,
        strict_format_detection: bool = False,
        require_safetensors: bool = False,
    ) -> None:
        """Сохраняет флаги политики.

        :param strict_format_detection: Значение ``extra_rules.strict_format_detection``.
        :param require_safetensors: Значение ``extra_rules.require_safetensors``.
        """
        self._strict_format_detection = strict_format_detection
        self._require_safetensors = require_safetensors

    def analyze(self, raw_data: RawScanData, context: MLContext) -> list[Issue]:
        """Возвращает находки политики формата.

        :param raw_data: Результат сканирования с фактами формата в metadata.
        :param context: ML-контекст (не используется — формат файла не зависит
            от определённого фреймворка).
        :return: Список находок; пустой, если оба ключа выключены либо факты
            формата не приложены.
        """
        if not (self._strict_format_detection or self._require_safetensors):
            return []

        facts = format_facts_of(raw_data)
        if facts is None:
            return []

        location = str(raw_data.file_path)
        issues: list[Issue] = []

        if self._strict_format_detection and facts.mismatch:
            issues.append(_make_mismatch_issue(facts, location))

        if self._require_safetensors and facts.safety is FormatSafety.CODE_BEARING:
            issues.append(_make_code_bearing_issue(facts, location))

        return issues


# ---------------------------------------------------------------------------
# Вспомогательные функции (модульного уровня — тестируются независимо)
# ---------------------------------------------------------------------------


def _make_mismatch_issue(facts: FormatFacts, location: str) -> Issue:
    """Строит MLS-FMT-001 — содержимое не совпало с обещанием расширения.

    В message и details пишутся СЫРЫЕ значения: заявленное расширение, формат
    по содержимому и признак, по которому он определён (magic-байты и смещение).
    Ничего не нормализуется — пользователь должен видеть исходные факты.
    """
    expected = ", ".join(sorted(fmt.value for fmt in facts.expected_formats))
    basis = facts.detection_basis or "структура заголовка"
    return Issue(
        code=_ISSUE_CODE_MISMATCH,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=(
            f"Формат файла не совпадает с расширением: расширение "
            f"{facts.declared_ext!r} обещает {expected}, а по содержимому это "
            f"{facts.detected_format.value} (определено по признаку: {basis})."
        ),
        location=location,
        details={
            "declared_ext": facts.declared_ext,
            "detected_format": facts.detected_format.value,
            "expected_formats": expected,
            "detection_basis": basis,
            "scanner": facts.scanner_name,
        },
        why=(
            "Инструменты сериализации всегда пишут файл со своим расширением, "
            "поэтому расхождение имени и содержимого не бывает случайным. "
            "Типовой вектор: вредоносный pickle называют model.safetensors — "
            "потребитель видит «безопасный формат» и загружает файл небезопасным "
            "путём. Доверять расширению при выборе загрузчика нельзя."
        ),
        remediation=(
            "Не загружайте файл по расширению. Установите фактический формат "
            "и происхождение файла; если источник доверенный — попросите "
            "пересохранить модель в safetensors с корректным именем."
        ),
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )


def _make_code_bearing_issue(facts: FormatFacts, location: str) -> Issue:
    """Строит MLS-FMT-002 — формат исполняет код, а политика этого не допускает.

    Находка эмитится по факту формата, независимо от того, нашли ли в
    содержимом что-то вредоносное. Обоснование класса берётся из реестра
    :data:`poison_check.scanners.format_facts.FORMAT_SAFETY` — пользователь
    видит причину, а не вердикт.
    """
    return Issue(
        code=_ISSUE_CODE_CODE_BEARING,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        message=(
            f"Формат {facts.safety_title} исполняет код при загрузке и не "
            f"допускается политикой (требуется safetensors). Файл с расширением "
            f"{facts.declared_ext!r}, формат по содержимому: "
            f"{facts.detected_format.value}, сканер: {facts.scanner_name}."
        ),
        location=location,
        details={
            "declared_ext": facts.declared_ext,
            "detected_format": facts.detected_format.value,
            "format_title": facts.safety_title,
            "scanner": facts.scanner_name,
            "policy_rule": "extra_rules.require_safetensors",
        },
        why=(
            f"{facts.safety_rationale} Политика требует форматов, безопасных по "
            "построению: находка означает недопустимый формат, а не найденный "
            "эксплойт. Отсутствие находок по содержимому здесь ничего не "
            "гарантирует — статический анализ видит не всё, а формат оставляет "
            "возможность исполнения кода при каждой загрузке."
        ),
        remediation=(
            "Пересохраните модель в формате safetensors "
            "(safetensors.torch.save_file / save_model) и распространяйте "
            "именно его. Если пересохранение невозможно — согласуйте "
            "исключение и загружайте файл только в изолированном окружении."
        ),
        compliance_tags=[
            "owasp-ml:ml03",
            "fstec:ubi-067",
            "gost:56939-2024:5.3",
        ],
    )

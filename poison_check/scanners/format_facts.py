"""Факты о формате файла: «что обещает расширение» и «что лежит внутри».

Слой: Scanners. Модуль знает ФОРМАТ и только формат — он не эмитит ни одной
Issue. Перевод фактов в находки — зона Detector
(:mod:`poison_check.detectors.format_policy_detector`, MLS-FMT-001/002).

**Зачем модуль существует.** Два ключа строгих политик —
``extra_rules.strict_format_detection`` и ``extra_rules.require_safetensors`` —
задают один и тот же вопрос к файлу: «какой это формат на самом деле?».
Первому нужен ответ, чтобы сверить его с расширением; второму — чтобы понять,
исполняет ли формат код при загрузке. Ответ считается здесь ОДИН раз и кладётся
в ``RawScanData.metadata``, откуда его читает детектор.

**Почему не в каждом сканере.** Определение формата по содержимому уже
существует единой точкой — :meth:`FormatDetector.detect_content` (magic-байты
GGUF/NumPy/ZIP/HDF5, опкод PROTO у pickle, структура заголовка safetensors).
Дублировать её в семи сканерах было бы ровно тем, чего требует не делать
постановка задачи. Поэтому факты пришивает одна функция
:func:`attach_format_facts`, которую агрегатор (CLI и Scanner-фасад) вызывает
сразу после ``scanner.scan()`` — симметрично ``annotate_suppressed_parse_error``
из слоя детекторов.

**Два факта, два вопроса.**

1. *Расхождение* (``_format_mismatch``). Формат определён по содержимому и НЕ
   входит в множество форматов, допустимых для заявленного расширения. Файл,
   формат которого определить не удалось (``UNKNOWN``), расхождением НЕ
   считается: это зона MLS-PARSE-001 / «неподдерживаемый формат», а не
   политики формата. Расширения, которые ничего не обещают (``.bin``), в
   таблицу ожиданий намеренно не входят — обещания нет, нарушать нечего.

2. *Класс безопасности* (``_format_safety``). Формат либо не исполняет код при
   загрузке (safetensors, GGUF, обычный .npy), либо исполняет
   (pickle/joblib/pytorch/keras), либо не определён. Классификация
   декларативная: :data:`FORMAT_SAFETY` покрывает КАЖДОЕ значение
   :class:`FileFormat` с обоснованием, а мета-тест не даёт добавить новый
   формат, забыв его классифицировать.

Класс считается по двум источникам сразу — формату по содержимому и формату
сканера, который файл обработал, — и берётся строгий из них. Так закрываются
оба краевых случая: pickle, замаскированный под ``.safetensors`` (сканер
считает файл безопасным, содержимое — нет) и сжатый joblib (содержимое —
неопознанный поток кодека, а сканер знает, что внутри pickle).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from poison_check.core.format_detector import (
    FileFormat,
    FormatDetector,
    FormatEvidence,
)
from poison_check.core.scanner_base import RawScanData
from poison_check.scanners.numpy_scanner import (
    META_OBJECT_DTYPE,
    META_PICKLE_PAYLOAD,
)

# ---------------------------------------------------------------------------
# Ключи metadata
# ---------------------------------------------------------------------------
#
# Префикс «_» — принятая в проекте пометка системного ключа, добавленного не из
# самого файла, а сканером/пайплайном: GGUFMetadataDetector такие ключи
# пропускает и не пытается искать в них URL. SecretsDetector нестроковые
# значения тоже пропускает.

#: Расширение файла как оно записано у пользователя (сырое, в нижнем регистре).
META_DECLARED_EXT: str = "_format_declared_ext"

#: Формат, определённый ТОЛЬКО по содержимому (значение :class:`FileFormat`).
META_DETECTED_FORMAT: str = "_format_detected"

#: Сырой признак, по которому формат определён (magic-байты + смещение).
META_DETECTION_BASIS: str = "_format_detection_basis"

#: Форматы, допустимые для заявленного расширения (через запятую).
META_EXPECTED_FORMATS: str = "_format_expected"

#: ``True``, если содержимое не совпало с обещанием расширения.
META_FORMAT_MISMATCH: str = "_format_mismatch"

#: Класс безопасности формата (значение :class:`FormatSafety`).
META_FORMAT_SAFETY: str = "_format_safety"

#: Обоснование класса безопасности — почему формат (не) исполняет код.
META_SAFETY_RATIONALE: str = "_format_safety_rationale"

#: Человекочитаемое имя формата, определившего класс безопасности.
META_SAFETY_FORMAT_TITLE: str = "_format_safety_title"

#: Имя сканера, обработавшего файл (или ``unknown``).
META_FORMAT_SCANNER: str = "_format_scanner"

#: ``True``, если ни один сканер не взялся за файл (формат не поддержан).
META_FORMAT_UNSUPPORTED: str = "_format_unsupported"


# ---------------------------------------------------------------------------
# Реестр безопасности форматов
# ---------------------------------------------------------------------------


class FormatSafety(Enum):
    """Класс формата по признаку «исполняется ли код при загрузке».

    * ``SAFE``         — загрузка не десериализует произвольный код.
    * ``CODE_BEARING`` — формат допускает исполнение кода при загрузке.
    * ``UNDETERMINED`` — формат не определён либо является контейнером общего
      назначения; класс задаёт вложенное содержимое, а не сам контейнер.
    """

    SAFE = "safe"
    CODE_BEARING = "code_bearing"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class FormatSafetySpec:
    """Классификация одного формата с обоснованием.

    :ivar safety: Класс безопасности.
    :ivar title: Человекочитаемое имя формата для сообщений.
    :ivar rationale: Почему формат отнесён к этому классу. Попадает в
        ``why`` находки MLS-FMT-002 — пользователь видит не вердикт, а причину.
    """

    safety: FormatSafety
    title: str
    rationale: str


#: Классификация КАЖДОГО формата из :class:`FileFormat`. Реестр декларативный:
#: добавление формата без записи здесь роняет мета-тест
#: ``tests/test_format_facts.py::TestSafetyRegistryIsComplete``, поэтому
#: «завтрашний формат» не может остаться неклассифицированным по недосмотру.
FORMAT_SAFETY: dict[FileFormat, FormatSafetySpec] = {
    FileFormat.SAFETENSORS: FormatSafetySpec(
        safety=FormatSafety.SAFE,
        title="safetensors",
        rationale=(
            "Формат состоит из JSON-заголовка с описанием тензоров и сырых "
            "байтов весов. В нём нет конструкции, которая при загрузке вызывала "
            "бы произвольный код: загрузчик читает числа, а не инструкции."
        ),
    ),
    FileFormat.GGUF: FormatSafetySpec(
        safety=FormatSafety.SAFE,
        title="GGUF",
        rationale=(
            "GGUF хранит типизированные пары ключ-значение и тензоры. Парсер "
            "(llama.cpp / Ollama) не десериализует объекты и не исполняет код; "
            "риск ограничен содержимым метаданных, которое проверяется отдельно "
            "(MLS-GGUF-001..006)."
        ),
    ),
    FileFormat.NUMPY_NPY: FormatSafetySpec(
        safety=FormatSafety.SAFE,
        title="NumPy .npy",
        rationale=(
            "Массив фиксированного dtype читается как сырые байты "
            "(numpy.load(allow_pickle=False)). Исключение — dtype=object: такой "
            "массив сериализуется через pickle, и файл перестаёт быть безопасным "
            "(факт object-dtype фиксирует NumpyScanner, см. MLS-NPY-001)."
        ),
    ),
    FileFormat.NUMPY_NPZ: FormatSafetySpec(
        safety=FormatSafety.SAFE,
        title="NumPy .npz",
        rationale=(
            "ZIP-архив из .npy-элементов; загрузка читает те же сырые байты. "
            "То же исключение по dtype=object, что и у .npy."
        ),
    ),
    FileFormat.PICKLE: FormatSafetySpec(
        safety=FormatSafety.CODE_BEARING,
        title="pickle",
        rationale=(
            "Pickle — не формат данных, а программа для стековой машины: опкоды "
            "GLOBAL/STACK_GLOBAL и REDUCE вызывают произвольную функцию при "
            "загрузке. Это штатная возможность формата, а не дефект, поэтому "
            "безопасным его не делает никакая проверка содержимого."
        ),
    ),
    FileFormat.JOBLIB: FormatSafetySpec(
        safety=FormatSafety.CODE_BEARING,
        title="joblib",
        rationale=(
            "joblib.load разворачивает компрессию (zlib/lz4/zstd) и отдаёт поток "
            "тому же pickle. Дополнительно компрессия скрывает payload от "
            "поверхностного просмотра файла."
        ),
    ),
    FileFormat.PYTORCH: FormatSafetySpec(
        safety=FormatSafety.CODE_BEARING,
        title="PyTorch (.pt/.pth/.ckpt)",
        rationale=(
            "torch.load распаковывает ZIP-контейнер и десериализует data.pkl "
            "обычным pickle (CVE-2025-32434 — обход weights_only). Формат "
            "унаследовал все возможности pickle по исполнению кода."
        ),
    ),
    FileFormat.KERAS_H5: FormatSafetySpec(
        safety=FormatSafety.CODE_BEARING,
        title="Keras (.keras/.h5)",
        rationale=(
            "Конфигурация модели допускает слои Lambda с сериализованным "
            "python-байткодом, который исполняется при загрузке "
            "(CVE-2025-1550). Формат несёт код по определению."
        ),
    ),
    FileFormat.ONNX: FormatSafetySpec(
        safety=FormatSafety.CODE_BEARING,
        title="ONNX",
        rationale=(
            "Сам protobuf-граф не десериализует Python-объекты, но допускает "
            "ссылки на кастомные операторы, подгружаемые рантаймом как нативные "
            "библиотеки. Гарантии «загрузка не исполняет код» формат не даёт, "
            "поэтому в строгом контуре он не считается безопасным."
        ),
    ),
    FileFormat.ZIP_ARCHIVE: FormatSafetySpec(
        safety=FormatSafety.UNDETERMINED,
        title="ZIP-контейнер",
        rationale=(
            "Контейнер общего назначения: класс безопасности задаёт вложенное "
            "содержимое (.npz — массивы, .pt — pickle, .keras — конфигурация "
            "слоёв), а не сам ZIP. Решение принимается по формату сканера, "
            "который развернул контейнер."
        ),
    ),
    FileFormat.TAR_ARCHIVE: FormatSafetySpec(
        safety=FormatSafety.UNDETERMINED,
        title="TAR-архив",
        rationale=(
            "Контейнер общего назначения — как и ZIP, сам по себе кода не несёт; "
            "класс определяется развёрнутым содержимым."
        ),
    ),
    FileFormat.UNKNOWN: FormatSafetySpec(
        safety=FormatSafety.UNDETERMINED,
        title="неопознанный формат",
        rationale=(
            "Формат определить не удалось. Это зона MLS-PARSE-001 / "
            "«неподдерживаемый формат»: файл не проверен, но утверждать про него "
            "«формат исполняет код» оснований нет."
        ),
    ),
}


#: Формат, соответствующий каждому зарегистрированному сканеру. Второй источник
#: для классификации: сканер знает то, чего не видно по первым байтам (сжатый
#: joblib, .keras внутри ZIP). Мета-тест сверяет реестр с ScannerRegistry —
#: новый сканер обязан появиться здесь.
SCANNER_FORMATS: dict[str, FileFormat] = {
    "pickle": FileFormat.PICKLE,
    "joblib": FileFormat.JOBLIB,
    "pytorch": FileFormat.PYTORCH,
    "keras": FileFormat.KERAS_H5,
    "numpy": FileFormat.NUMPY_NPY,
    "safetensors": FileFormat.SAFETENSORS,
    "gguf": FileFormat.GGUF,
}


#: Что расширение ОБЕЩАЕТ по содержимому. Множество, а не одно значение:
#: у ``.pt`` три исторических представления (ZIP-контейнер, legacy pickle,
#: совсем старый tar), и ни одно из них не является подделкой.
#:
#: Расширения, которых здесь нет, ничего не обещают и не могут дать расхождения.
#: Намеренно отсутствует ``.bin``: это универсальное имя для любых бинарных
#: весов (HuggingFace pytorch_model.bin, сырые тензоры, произвольные дампы) —
#: ожидание для него было бы выдумкой и дало бы шум на легитимных файлах.
EXPECTED_CONTENT_FORMATS: dict[str, frozenset[FileFormat]] = {
    ".safetensors": frozenset({FileFormat.SAFETENSORS}),
    ".pkl": frozenset({FileFormat.PICKLE}),
    ".pickle": frozenset({FileFormat.PICKLE}),
    ".dill": frozenset({FileFormat.PICKLE}),
    # Несжатый joblib — обычный pickle. Сжатый начинается с заголовка кодека,
    # который FormatDetector не опознаёт (UNKNOWN) — расхождением это не будет.
    ".joblib": frozenset({FileFormat.PICKLE}),
    ".npy": frozenset({FileFormat.NUMPY_NPY}),
    ".npz": frozenset({FileFormat.ZIP_ARCHIVE}),
    ".gguf": frozenset({FileFormat.GGUF}),
    ".ggml": frozenset({FileFormat.GGUF}),
    ".pt": frozenset(
        {FileFormat.ZIP_ARCHIVE, FileFormat.PICKLE, FileFormat.TAR_ARCHIVE}
    ),
    ".pth": frozenset(
        {FileFormat.ZIP_ARCHIVE, FileFormat.PICKLE, FileFormat.TAR_ARCHIVE}
    ),
    ".ckpt": frozenset(
        {FileFormat.ZIP_ARCHIVE, FileFormat.PICKLE, FileFormat.TAR_ARCHIVE}
    ),
    ".keras": frozenset({FileFormat.ZIP_ARCHIVE}),
    ".h5": frozenset({FileFormat.KERAS_H5}),
    ".hdf5": frozenset({FileFormat.KERAS_H5}),
}


#: Флаги metadata, при которых формально безопасный формат становится
#: code-bearing: внутри массива лежит pickle. Список декларативный — новый
#: подобный факт добавляется одной строкой.
_CODE_BEARING_METADATA_FLAGS: tuple[tuple[str, str], ...] = (
    (
        META_OBJECT_DTYPE,
        "массив имеет dtype=object, а такие массивы NumPy сериализует через "
        "pickle — формат перестаёт быть безопасным",
    ),
    (
        META_PICKLE_PAYLOAD,
        "внутри данных массива обнаружен pickle-поток",
    ),
)


# ---------------------------------------------------------------------------
# Факты о конкретном файле
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatFacts:
    """Разобранные факты о формате одного файла.

    Собираются :func:`build_format_facts`, кладутся в ``RawScanData.metadata``
    и читаются обратно :func:`format_facts_of`. Детектор работает с этой
    структурой, а не с сырыми ключами словаря.

    :ivar declared_ext: Расширение файла (в нижнем регистре, с точкой).
    :ivar detected_format: Формат по содержимому.
    :ivar detection_basis: Сырой признак определения формата.
    :ivar expected_formats: Форматы, допустимые для расширения (пусто — если
        расширение ничего не обещает).
    :ivar mismatch: Содержимое не совпало с обещанием расширения.
    :ivar safety: Класс безопасности с учётом обоих источников.
    :ivar safety_title: Имя формата, определившего класс.
    :ivar safety_rationale: Обоснование класса.
    :ivar scanner_name: Имя сканера, обработавшего файл.
    """

    declared_ext: str
    detected_format: FileFormat
    detection_basis: str
    expected_formats: frozenset[FileFormat]
    mismatch: bool
    safety: FormatSafety
    safety_title: str
    safety_rationale: str
    scanner_name: str


def build_format_facts(path: Path, raw_data: RawScanData) -> FormatFacts:
    """Собирает факты о формате файла: расхождение и класс безопасности.

    Чистая функция: ничего не пишет и не эмитит, только считает. Запись в
    ``RawScanData`` делает :func:`attach_format_facts`.

    :param path: Реальный путь к файлу (читаются первые байты заголовка).
    :param raw_data: Результат сканирования — источник имени сканера и фактов,
        уточняющих классификацию (object-dtype у NumPy).
    :return: Факты о формате.
    """
    declared_ext = path.suffix.lower()
    evidence: FormatEvidence = FormatDetector.detect_content(path)
    expected = EXPECTED_CONTENT_FORMATS.get(declared_ext, frozenset())

    # Расхождение — только когда формат ОПРЕДЕЛЁН и не входит в ожидаемые.
    # Неопознанный формат расхождением не считается: это граница с
    # MLS-PARSE-001 / «неподдерживаемый формат».
    mismatch = bool(
        expected
        and evidence.format is not FileFormat.UNKNOWN
        and evidence.format not in expected
    )

    safety, title, rationale = _classify_safety(evidence.format, raw_data)

    return FormatFacts(
        declared_ext=declared_ext,
        detected_format=evidence.format,
        detection_basis=evidence.basis,
        expected_formats=expected,
        mismatch=mismatch,
        safety=safety,
        safety_title=title,
        safety_rationale=rationale,
        scanner_name=raw_data.scanner_name,
    )


def attach_format_facts(
    raw_data: RawScanData,
    path: Path,
    *,
    unsupported: bool = False,
) -> None:
    """Записывает факты о формате в ``raw_data.metadata``.

    Вызывается агрегатором (CLI и Scanner-фасад) сразу после ``scanner.scan()``
    и до запуска детекторов. Мутирует ``raw_data`` на месте — симметрично
    ``annotate_suppressed_parse_error`` в слое детекторов.

    Никогда не бросает исключений наружу: недоступный файл даёт формат
    ``UNKNOWN`` (см. :meth:`FormatDetector.detect_content`), а не ошибку.

    :param raw_data: Результат сканирования; дополняется фактами.
    :param path: Реальный путь к файлу.
    :param unsupported: ``True``, если ни один сканер не взялся за файл. Флаг
        нужен ParseErrorDetector: неподдерживаемый формат — это не сбой разбора,
        и MLS-PARSE-001 по нему не эмитится (файл виден через
        ``FileResult.error`` и отдельный канал файловых фактов).
    """
    facts = build_format_facts(path, raw_data)

    metadata: dict[str, Any] = raw_data.metadata if raw_data.metadata is not None else {}
    metadata[META_DECLARED_EXT] = facts.declared_ext
    metadata[META_DETECTED_FORMAT] = facts.detected_format.value
    metadata[META_DETECTION_BASIS] = facts.detection_basis
    metadata[META_EXPECTED_FORMATS] = ", ".join(
        sorted(fmt.value for fmt in facts.expected_formats)
    )
    metadata[META_FORMAT_MISMATCH] = facts.mismatch
    metadata[META_FORMAT_SAFETY] = facts.safety.value
    metadata[META_SAFETY_FORMAT_TITLE] = facts.safety_title
    metadata[META_SAFETY_RATIONALE] = facts.safety_rationale
    metadata[META_FORMAT_SCANNER] = facts.scanner_name
    if unsupported:
        metadata[META_FORMAT_UNSUPPORTED] = True
    raw_data.metadata = metadata


def format_facts_of(raw_data: RawScanData) -> FormatFacts | None:
    """Читает факты о формате из ``raw_data.metadata``.

    :param raw_data: Результат сканирования.
    :return: Факты или ``None``, если :func:`attach_format_facts` не вызывалась
        (например, сканер вызван напрямую, в обход пайплайна) — детектор в этом
        случае молчит, а не гадает.
    """
    metadata = raw_data.metadata
    if not metadata or META_DETECTED_FORMAT not in metadata:
        return None

    expected_raw = str(metadata.get(META_EXPECTED_FORMATS, ""))
    expected = frozenset(
        FileFormat(value)
        for value in (part.strip() for part in expected_raw.split(","))
        if value
    )
    return FormatFacts(
        declared_ext=str(metadata.get(META_DECLARED_EXT, "")),
        detected_format=FileFormat(metadata[META_DETECTED_FORMAT]),
        detection_basis=str(metadata.get(META_DETECTION_BASIS, "")),
        expected_formats=expected,
        mismatch=bool(metadata.get(META_FORMAT_MISMATCH)),
        safety=FormatSafety(
            metadata.get(META_FORMAT_SAFETY, FormatSafety.UNDETERMINED.value)
        ),
        safety_title=str(metadata.get(META_SAFETY_FORMAT_TITLE, "")),
        safety_rationale=str(metadata.get(META_SAFETY_RATIONALE, "")),
        scanner_name=str(metadata.get(META_FORMAT_SCANNER, raw_data.scanner_name)),
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _classify_safety(
    detected: FileFormat,
    raw_data: RawScanData,
) -> tuple[FormatSafety, str, str]:
    """Определяет класс безопасности по содержимому И по формату сканера.

    Берётся строгий из двух источников (CODE_BEARING > SAFE > UNDETERMINED):

    * содержимое строже сканера, когда pickle замаскирован под ``.safetensors``
      (сканер выбран по расширению и считает файл безопасным);
    * сканер строже содержимого, когда joblib сжат (по первым байтам это
      неопознанный поток кодека, а сканер знает, что внутри pickle).

    Отдельно учитываются metadata-факты, превращающие формально безопасный
    формат в code-bearing (``dtype=object`` у NumPy).

    :return: Класс, имя формата и обоснование — всё для сообщения детектора.
    """
    scanner_format = SCANNER_FORMATS.get(raw_data.scanner_name)
    candidates = [detected]
    if scanner_format is not None and scanner_format is not detected:
        candidates.append(scanner_format)

    for wanted in (FormatSafety.CODE_BEARING, FormatSafety.SAFE):
        for candidate in candidates:
            spec = FORMAT_SAFETY[candidate]
            if spec.safety is wanted:
                if wanted is FormatSafety.SAFE:
                    return _apply_metadata_overrides(spec, raw_data)
                return spec.safety, spec.title, spec.rationale

    spec = FORMAT_SAFETY[candidates[0]]
    return spec.safety, spec.title, spec.rationale


def _apply_metadata_overrides(
    spec: FormatSafetySpec,
    raw_data: RawScanData,
) -> tuple[FormatSafety, str, str]:
    """Переводит безопасный формат в code-bearing по фактам сканера.

    Единственный такой случай сегодня — NumPy с ``dtype=object``: формат
    хранения безопасен, но конкретный массив сериализован через pickle.
    """
    metadata = raw_data.metadata or {}
    for flag, reason in _CODE_BEARING_METADATA_FLAGS:
        if metadata.get(flag):
            return (
                FormatSafety.CODE_BEARING,
                spec.title,
                f"{spec.rationale} В этом файле {reason}.",
            )
    return spec.safety, spec.title, spec.rationale

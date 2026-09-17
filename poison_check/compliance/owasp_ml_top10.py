"""Маппинг кодов угроз poison-check на OWASP Machine Learning Security Top 10.

Источник: https://owasp.org/www-project-machine-learning-security-top-10/
Версия списка: 2023 (актуальная на момент написания).

OWASP ML Top 10 описывает десять наиболее критичных категорий угроз
безопасности ML-систем. Каждая категория имеет идентификатор ML01–ML10.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Final

from poison_check.core.result import Issue, ScanResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Справочник OWASP ML Top 10
# ---------------------------------------------------------------------------

#: Полное описание категорий OWASP ML Top 10 на русском языке.
OWASP_ML_TOP10: Final[dict[str, dict[str, str]]] = {
    "ML01": {
        "title_ru": "Атаки на входные данные (Input Manipulation Attack)",
        "description_ru": (
            "Злоумышленник подаёт на вход модели специально подготовленные данные "
            "(adversarial examples), чтобы вызвать неправильную классификацию или "
            "обход средств защиты. Включает атаки уклонения (evasion attacks)."
        ),
        "examples_ru": (
            "Adversarial perturbations в изображениях; специально сформированный "
            "текст для обхода NLP-классификаторов; аудио-атаки на ASR-системы."
        ),
    },
    "ML02": {
        "title_ru": "Атаки отравления данных (Data Poisoning Attack)",
        "description_ru": (
            "Внедрение вредоносных или специально подготовленных образцов в обучающую "
            "выборку с целью изменения поведения модели: снижение точности, создание "
            "бэкдора или смещение классификации для определённых входов."
        ),
        "examples_ru": (
            "Backdoor-атаки через отравленные обучающие примеры; смещение "
            "тональности через манипуляцию данными; атаки на федеративное обучение."
        ),
    },
    "ML03": {
        "title_ru": "Атаки на цепочку поставок (ML Supply Chain Attack)",
        "description_ru": (
            "Внедрение вредоносного кода через ML-модели, предобученные веса, "
            "зависимости или инструменты разработки. Злоумышленник компрометирует "
            "компоненты на этапах до или во время развёртывания модели. "
            "Pickle-based RCE — наиболее распространённый вектор в этой категории."
        ),
        "examples_ru": (
            "Вредоносные .pkl/.pt файлы на HuggingFace; компрометация pip-пакетов; "
            "встроенные исполняемые файлы (PE/ELF) в архивах моделей."
        ),
    },
    "ML04": {
        "title_ru": "Атаки на инверсию модели (Model Inversion Attack)",
        "description_ru": (
            "Восстановление обучающих данных или чувствительной информации путём "
            "многократных запросов к модели. Злоумышленник использует выходные данные "
            "модели для восстановления входных данных, использованных при обучении."
        ),
        "examples_ru": (
            "Восстановление лиц из модели распознавания; извлечение медицинских "
            "данных из клинических моделей; атаки на языковые модели с PII."
        ),
    },
    "ML05": {
        "title_ru": "Атаки на извлечение модели (Model Theft)",
        "description_ru": (
            "Воссоздание функциональности или архитектуры модели через систематические "
            "запросы к API. Злоумышленник создаёт суррогатную модель, близкую к "
            "оригиналу, без доступа к весам или обучающим данным."
        ),
        "examples_ru": (
            "Кража коммерческих ML-API через query-based extraction; воссоздание "
            "архитектуры через активационные паттерны."
        ),
    },
    "ML06": {
        "title_ru": "Атаки на AI-систему (AI-Specific Attack)",
        "description_ru": (
            "Атаки, специфичные для AI-компонентов: манипуляции с моделью через "
            "её интерфейс, эксплуатация специфических свойств нейронных сетей, "
            "атаки через промпты (prompt injection) для LLM-систем."
        ),
        "examples_ru": (
            "Prompt injection в LLM-агентах; джейлбрейк через специальные "
            "токены; атаки на embedding-пространство."
        ),
    },
    "ML07": {
        "title_ru": "Уязвимости передачи данных (Transfer Learning Attack)",
        "description_ru": (
            "Эксплуатация уязвимостей, внесённых при transfer learning или "
            "fine-tuning. Предобученная база может содержать бэкдоры или смещения, "
            "которые сохраняются после дообучения на новых данных."
        ),
        "examples_ru": (
            "Скрытые бэкдоры в предобученных весах; сохранение бэкдора после "
            "fine-tuning; атаки через отравленные foundation models."
        ),
    },
    "ML08": {
        "title_ru": "Уязвимости модели (Model Vulnerability)",
        "description_ru": (
            "Технические уязвимости в самой модели или инфраструктуре: небезопасные "
            "форматы сериализации, уязвимые зависимости, небезопасные конфигурации "
            "фреймворков. Включает известные CVE в ML-фреймворках."
        ),
        "examples_ru": (
            "CVE-2025-32434 (PyTorch arbitrary code execution); небезопасный "
            "torch.load без weights_only; уязвимые версии TensorFlow/sklearn."
        ),
    },
    "ML09": {
        "title_ru": "Нарушение целостности вывода (Output Integrity Attack)",
        "description_ru": (
            "Манипуляция с выходными данными модели или хранимыми результатами "
            "для достижения нужного злоумышленнику результата. Включает утечку "
            "чувствительных данных через выходы модели (memorization attacks)."
        ),
        "examples_ru": (
            "Извлечение секретов из обученной LLM; API-ключи и токены, "
            "memorized моделью; персональные данные в весах модели."
        ),
    },
    "ML10": {
        "title_ru": "Атаки на среду выполнения (Model Runtime Attack)",
        "description_ru": (
            "Атаки на инфраструктуру развёртывания и среду выполнения модели: "
            "несанкционированный доступ к API, атаки на сервер вывода, "
            "эксплуатация runtime-уязвимостей при загрузке модели."
        ),
        "examples_ru": (
            "RCE через pickle при torch.load; выполнение кода при десериализации "
            "joblib-файлов; сетевые callback'и при загрузке модели."
        ),
    },
}

# ---------------------------------------------------------------------------
# Маппинг: Issue.code → список категорий OWASP ML Top 10
# ---------------------------------------------------------------------------

#: Маппинг Issue.code → список идентификаторов OWASP ML Top 10.
ISSUE_TO_OWASP: Final[dict[str, list[str]]] = {
    # --- AllowlistDetector ---
    # Неразрешённый глобал → supply chain + model vulnerability
    "MLS-ALW-001": ["ML03", "ML08"],

    # --- BlocklistDetector ---
    # Запрещённый глобал → supply chain + runtime attack
    "MLS-PKL-001": ["ML03", "ML10"],

    # --- SecretsDetector ---
    # Секреты в модели → нарушение целостности вывода (memorization)
    "MLS-SEC-001": ["ML09"],

    # --- NetworkDetector ---
    # Whitelisted домен → supply chain (информационно)
    "MLS-NET-001": ["ML03"],
    # Неизвестный домен → runtime attack (callback при загрузке)
    "MLS-NET-002": ["ML10", "ML03"],
    # Приватный IP → runtime attack
    "MLS-NET-003": ["ML10"],
    # Публичный IP → runtime attack
    "MLS-NET-004": ["ML10"],
    # Голый публичный IP → runtime attack (network callback)
    "MLS-NET-005": ["ML10"],
    # Опасная URL-схема — supply chain через манипулируемые URL
    "MLS-NET-006": ["ML03", "ML10"],

    # --- ExecutableDetector ---
    # Встроенный исполняемый файл → supply chain + runtime
    "MLS-EXE-001": ["ML03", "ML10"],

    # --- CompressionDetector ---
    # Архивная бомба → runtime attack (DoS при загрузке)
    "MLS-CMP-001": ["ML10"],
    # Подозрительное сжатие → runtime attack
    "MLS-CMP-002": ["ML10"],
    # Глубокая вложенность → runtime attack
    "MLS-CMP-003": ["ML10"],

    # --- JoblibScanner: decompression bomb ---
    "MLS-JOBLIB-003": ["ML10"],

    # --- FormatPolicyDetector: политика формата ---
    "MLS-FMT-001": ["ML03", "ML10"],
    "MLS-FMT-002": ["ML03", "ML08"],

    # --- NumpyScanner: object-dtype с pickle ---
    "MLS-NPY-001": ["ML03", "ML10"],

    # --- GGUFMetadataDetector ---
    "MLS-GGUF-001": ["ML03"],
    "MLS-GGUF-002": ["ML10", "ML03"],
    "MLS-GGUF-003": ["ML10"],
    "MLS-GGUF-004": ["ML03"],

    # --- CVEDetector: конкретные CVE и паттерны ---
    "MLS-CVE-2025-32434": ["ML08", "ML10"],
    "MLS-PATTERN-OS-SYSTEM": ["ML03", "ML10"],
    "MLS-PATTERN-SUBPROCESS": ["ML03", "ML10"],
    "MLS-PATTERN-BUILTINS-EVAL": ["ML03", "ML10"],
    "MLS-PATTERN-BUILTINS-EXEC": ["ML03", "ML10"],
    "MLS-PATTERN-DILL-BYPASS": ["ML03", "ML08"],
    "MLS-GHSA-83PF-V6QQ-PWMR": ["ML08", "ML10"],
    "MLS-PATTERN-NUMPY-FROMPYFUNC": ["ML03", "ML10"],
    "MLS-PATTERN-CLOUDPICKLE-FUNCTION": ["ML03", "ML10"],
    "MLS-PATTERN-CTYPES-EXEC": ["ML03", "ML10"],
}


class OwaspMapper:
    """Маппер находок poison-check на категории OWASP Machine Learning Security Top 10.

    Логика маппинга:
    1. Точное совпадение Issue.code с ключом ISSUE_TO_OWASP.
    2. Prefix-matching для динамических кодов CVEDetector (MLS-CVE-*, MLS-PATTERN-*).
    3. Fallback: пустой список + предупреждение в лог.
    """

    def map_issue(self, issue: Issue) -> list[str]:
        """Возвращает список категорий OWASP ML Top 10 для данного Issue.

        Параметры:
            issue: Issue, полученный от любого детектора poison-check.

        Возвращает:
            Список идентификаторов вида ['ML03', 'ML10'].
            Пустой список если маппинг не найден.
        """
        # 1. Точное совпадение
        if issue.code in ISSUE_TO_OWASP:
            return list(ISSUE_TO_OWASP[issue.code])

        # 2. Prefix-matching для динамических кодов CVEDetector.
        #    Longest-first: специфичный паттерн перебивает общий префикс.
        for key in sorted(ISSUE_TO_OWASP, key=len, reverse=True):
            if issue.code.startswith(key):
                return list(ISSUE_TO_OWASP[key])

        logger.debug(
            "OwaspMapper: нет маппинга OWASP для кода %r",
            issue.code,
        )
        return []

    def generate_report(self, result: ScanResult) -> dict[str, Any]:
        """Формирует структурированный отчёт по OWASP ML Top 10.

        Отчёт содержит все десять категорий ML01–ML10. Для каждой категории
        указаны: название, описание, список найденных Issues и статус (hit/clean).

        Параметры:
            result: Итоговый результат прогона сканирования.

        Возвращает:
            Словарь со структурой:
            {
                "ML01": {
                    "title_ru": ...,
                    "description_ru": ...,
                    "status": "hit" | "clean",
                    "issues": [Issue, ...]
                },
                ...
                "_summary": {
                    "hit_categories": ["ML03", "ML10"],
                    "clean_categories": ["ML01", ...],
                    "total_issues": N
                }
            }
        """
        # Инициализируем все категории как «чистые»
        report: dict[str, Any] = {
            cat_id: {
                "title_ru": cat_info["title_ru"],
                "description_ru": cat_info["description_ru"],
                "status": "clean",
                "issues": [],
            }
            for cat_id, cat_info in OWASP_ML_TOP10.items()
        }

        # Группируем Issues по категориям
        issues_by_category: dict[str, list[Issue]] = defaultdict(list)
        total_issues = 0

        for file_result in result.results_per_file.values():
            for issue in file_result.issues:
                owasp_cats = self.map_issue(issue)
                for cat_id in owasp_cats:
                    if cat_id in report:
                        issues_by_category[cat_id].append(issue)
                total_issues += 1

        # Заполняем Issues и статусы
        hit_categories: list[str] = []
        clean_categories: list[str] = []

        for cat_id in OWASP_ML_TOP10:
            cat_issues = issues_by_category.get(cat_id, [])
            report[cat_id]["issues"] = cat_issues
            if cat_issues:
                report[cat_id]["status"] = "hit"
                hit_categories.append(cat_id)
            else:
                clean_categories.append(cat_id)

        report["_summary"] = {
            "hit_categories": sorted(hit_categories),
            "clean_categories": sorted(clean_categories),
            "total_issues": total_issues,
        }

        return report

    def compliance_tag(self, owasp_id: str) -> str:
        """Формирует compliance-тег в формате 'owasp-ml:ml03'.

        Параметры:
            owasp_id: Идентификатор вида 'ML03'.

        Возвращает:
            Строка вида 'owasp-ml:ml03'.
        """
        return f"owasp-ml:{owasp_id.lower()}"

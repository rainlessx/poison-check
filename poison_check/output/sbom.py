"""CycloneDX SBOM форматтер для результатов сканирования ML-файлов.

SBOM (Software Bill of Materials) в формате CycloneDX JSON schema v1.4.
Используется для инвентаризации ML-моделей и документирования уязвимостей.

Спецификация: https://cyclonedx.org/specification/overview/
Схема v1.4: https://cyclonedx.org/schema/bom/1.4

Структура вывода:
    bomFormat      = "CycloneDX"
    specVersion    = "1.4"
    components     = каждый проверенный ML-файл (type="machine-learning-model")
    vulnerabilities = найденные Issues с CVE-кодами

Формат SBOM (CycloneDX) — по образцу распространённой практики генерации SBOM.
"""

from __future__ import annotations

import json
import uuid
from datetime import timezone
from pathlib import Path
from typing import Any

from poison_check.core.result import Confidence, FileResult, Issue, ScanResult, Severity
from poison_check.output.report_metadata import report_metadata_sbom_properties
from poison_check.output.severity_threshold import (
    below_threshold_justification,
    count_below_threshold,
    is_below_threshold,
    threshold_of,
)

# ---------------------------------------------------------------------------
# Маппинг severity → CycloneDX severity
# ---------------------------------------------------------------------------

_SEVERITY_TO_CYCLONEDX: dict[Severity, str] = {
    Severity.CRITICAL: "critical",
    Severity.HIGH: "high",
    Severity.MEDIUM: "medium",
    Severity.LOW: "low",
    Severity.INFO: "info",
}

# Маппинг confidence → CVSS score (приблизительный)
_CONFIDENCE_TO_SCORE: dict[Confidence, float] = {
    Confidence.CERTAIN: 9.8,
    Confidence.HIGH: 8.0,
    Confidence.MEDIUM: 5.5,
    Confidence.LOW: 3.0,
}


class SbomFormatter:
    """Форматтер результатов сканирования в CycloneDX JSON SBOM v1.4.

    Компоненты = каждый проверенный ML-файл (type="machine-learning-model").
    Версия компонента = SHA-256 хеш файла (или "unknown" если недоступен).
    Уязвимости = найденные Issue, обогащённые CVE-ссылками если есть.

    Пример использования:
        formatter = SbomFormatter()
        sbom_json = formatter.format(scan_result)
        Path("models.cdx.json").write_text(sbom_json, encoding="utf-8")
    """

    BOM_FORMAT: str = "CycloneDX"
    SPEC_VERSION: str = "1.4"
    TOOL_NAME: str = "poison-check"

    def format(self, result: ScanResult) -> str:
        """Сериализует ScanResult в CycloneDX JSON SBOM v1.4.

        :param result: Результат сканирования.
        :return: Отформатированная JSON-строка CycloneDX SBOM.
        """
        # Генерируем стабильный serial number на основе timestamp и путей
        serial_number = f"urn:uuid:{uuid.uuid4()}"

        # Метаданные SBOM
        metadata = self._build_metadata(result)

        # Компоненты — ML-файлы
        components, component_bom_refs = self._build_components(result)

        # Уязвимости — Issues
        vulnerabilities = self._build_vulnerabilities(result, component_bom_refs)

        doc: dict[str, Any] = {
            "bomFormat": self.BOM_FORMAT,
            "specVersion": self.SPEC_VERSION,
            "serialNumber": serial_number,
            "version": 1,
            "metadata": metadata,
            "components": components,
            "vulnerabilities": vulnerabilities,
        }

        return json.dumps(doc, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Построение секций SBOM
    # ------------------------------------------------------------------

    def _build_metadata(self, result: ScanResult) -> dict[str, Any]:
        """Строит секцию metadata CycloneDX SBOM.

        Пороги политики попадают в ``metadata.properties``: по SBOM должно быть
        видно, при каком ``severity_threshold`` часть уязвимостей помечена
        подпороговыми и какой ключ отвечал за провал гейта.
        """
        timestamp = result.timestamp
        if timestamp.tzinfo is None:
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            timestamp_str = timestamp.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        threshold = threshold_of(result)
        thresholds = result.policy_thresholds
        fail_on = thresholds.fail_on_severity if thresholds is not None else None

        properties: list[dict[str, str]] = [
            {"name": "poison-check:policy", "value": result.policy},
            {
                "name": "poison-check:files_scanned",
                "value": str(len(result.results_per_file)),
            },
        ]
        if threshold is not None:
            properties.append(
                {"name": "poison-check:severity_threshold", "value": threshold.value}
            )
            properties.append(
                {
                    "name": "poison-check:below_threshold_count",
                    "value": str(count_below_threshold(result, threshold)),
                }
            )
        if fail_on is not None:
            properties.append(
                {"name": "poison-check:fail_on_severity", "value": fail_on.value}
            )

        # Шапка отчёта (--client / --auditor). Незаданное поле не добавляется —
        # в инвентаризации пустой реквизит хуже отсутствующего.
        properties.extend(report_metadata_sbom_properties(result))

        return {
            "timestamp": timestamp_str,
            "tools": [
                {
                    "vendor": "poison-check",
                    "name": self.TOOL_NAME,
                    "version": result.tool_version,
                }
            ],
            "properties": properties,
        }

    def _build_components(
        self, result: ScanResult
    ) -> tuple[list[dict[str, Any]], dict[Path, str]]:
        """Строит список компонентов CycloneDX (по одному на ML-файл).

        Возвращает:
            (список компонентов, словарь path → bom-ref)
        """
        components: list[dict[str, Any]] = []
        component_bom_refs: dict[Path, str] = {}

        for idx, (file_path, file_result) in enumerate(
            result.results_per_file.items()
        ):
            bom_ref = f"component-{idx}"
            component_bom_refs[file_path] = bom_ref

            component = self._file_to_component(
                file_path=file_path,
                file_result=file_result,
                bom_ref=bom_ref,
            )
            components.append(component)

        return components, component_bom_refs

    def _file_to_component(
        self,
        file_path: Path,
        file_result: FileResult,
        bom_ref: str,
    ) -> dict[str, Any]:
        """Преобразует FileResult в CycloneDX component.

        Версия компонента = SHA-256 хеш файла.
        Если хеш недоступен, используем "unknown".
        """
        # Пытаемся получить хеш файла — может быть в деталях Issue или вычислим сами
        file_hash = self._get_file_hash(file_path)

        component: dict[str, Any] = {
            "type": "machine-learning-model",
            "bom-ref": bom_ref,
            "name": file_path.name,
            "version": file_hash,
            "description": (
                f"ML-модель: {file_path.name}. "
                f"Сканер: {file_result.scanner_name}."
            ),
            "properties": [
                {"name": "poison-check:scanner", "value": file_result.scanner_name},
                {"name": "poison-check:path", "value": str(file_path)},
            ],
        }

        # Добавляем хеши в секцию hashes если удалось получить
        hashes = self._build_hashes(file_path)
        if hashes:
            component["hashes"] = hashes

        # Ошибка сканирования
        if file_result.error:
            component["properties"].append(
                {"name": "poison-check:scan_error", "value": file_result.error}
            )

        return component

    def _build_vulnerabilities(
        self,
        result: ScanResult,
        component_bom_refs: dict[Path, str],
    ) -> list[dict[str, Any]]:
        """Строит список уязвимостей CycloneDX из Issues.

        Каждый Issue становится vulnerability. Issues с CVE-ссылками
        получают поле id из CVE идентификатора.
        """
        vulnerabilities: list[dict[str, Any]] = []
        vuln_idx = 0
        threshold = threshold_of(result)

        for file_path, file_result in result.results_per_file.items():
            bom_ref = component_bom_refs.get(file_path, "unknown")
            for issue in file_result.issues:
                vuln = self._issue_to_vulnerability(
                    issue=issue,
                    component_bom_ref=bom_ref,
                    vuln_idx=vuln_idx,
                    threshold=threshold,
                )
                vulnerabilities.append(vuln)
                vuln_idx += 1

        return vulnerabilities

    def _issue_to_vulnerability(
        self,
        issue: Issue,
        component_bom_ref: str,
        vuln_idx: int,
        threshold: Severity | None = None,
    ) -> dict[str, Any]:
        """Преобразует Issue в CycloneDX vulnerability.

        Если в references есть CVE — используем его как id.
        Иначе генерируем внутренний идентификатор MLS-XXXXX.

        Порог внимания политики не удаляет уязвимость из SBOM (инвентаризация
        обязана быть полной), а выражается секцией ``analysis``: подпороговая
        запись получает ``state: "in_triage"`` с пояснением. ``ratings`` при
        этом не меняются — severity уязвимости от настройки отчёта не зависит.
        """
        # Ищем CVE среди ссылок
        cve_id: str | None = None
        for ref in issue.references:
            if ref.type.lower() == "cve":
                cve_id = ref.id
                break

        vuln_id = cve_id if cve_id else f"MLS-{issue.code}-{vuln_idx:05d}"
        vuln_bom_ref = f"vuln-{vuln_idx}"

        # Строим ratings (оценка уязвимости)
        ratings = [
            {
                "source": {"name": "poison-check"},
                "score": _CONFIDENCE_TO_SCORE.get(issue.confidence, 5.0),
                "severity": _SEVERITY_TO_CYCLONEDX[issue.severity],
                "method": "other",
                "justification": issue.confidence.value,
            }
        ]

        vuln: dict[str, Any] = {
            "bom-ref": vuln_bom_ref,
            "id": vuln_id,
            "source": {
                "name": "poison-check",
                "url": "https://github.com/poison-check/poison-check",
            },
            "ratings": ratings,
            "description": issue.message,
            "affects": [
                {
                    "ref": component_bom_ref,
                }
            ],
            "properties": [
                {"name": "poison-check:code", "value": issue.code},
                {"name": "poison-check:severity", "value": issue.severity.value},
                {"name": "poison-check:confidence", "value": issue.confidence.value},
                {"name": "poison-check:location", "value": issue.location},
            ],
        }

        # Отметка порога внимания
        below = is_below_threshold(issue, threshold)
        vuln["properties"].append(
            {"name": "poison-check:below_threshold", "value": str(below).lower()}
        )
        if below:
            vuln["analysis"] = {
                "state": "in_triage",
                "detail": below_threshold_justification(threshold),
            }

        # Необязательные поля
        if issue.why:
            vuln["detail"] = issue.why
        if issue.remediation:
            vuln["recommendation"] = issue.remediation

        # Ссылки CWE
        cwes: list[int] = []
        advisories: list[dict[str, str]] = []
        for ref in issue.references:
            if ref.type.lower() == "cwe":
                try:
                    cwe_num = int(ref.id.replace("CWE-", "").replace("cwe-", ""))
                    cwes.append(cwe_num)
                except ValueError:
                    pass
            elif ref.type.lower() != "cve":
                # Прочие ссылки — в advisories
                advisories.append({"title": f"{ref.type.upper()}: {ref.id}"})

        if cwes:
            vuln["cwes"] = cwes
        if advisories:
            vuln["advisories"] = advisories

        # Compliance-теги в properties
        if issue.compliance_tags:
            vuln["properties"].append(
                {
                    "name": "poison-check:compliance_tags",
                    "value": ", ".join(issue.compliance_tags),
                }
            )

        return vuln

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _get_file_hash(path: Path) -> str:
        """Вычисляет SHA-256 хеш файла потоково. При ошибке возвращает 'unknown'."""
        import hashlib

        try:
            h = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return "unknown"

    @staticmethod
    def _build_hashes(path: Path) -> list[dict[str, str]]:
        """Строит список хешей CycloneDX для файла.

        CycloneDX алгоритмы: SHA-256, SHA-512, MD5.
        При ошибке возвращает пустой список.
        """
        import hashlib

        try:
            h_sha256 = hashlib.sha256()
            h_sha512 = hashlib.sha512()
            h_md5 = hashlib.md5(usedforsecurity=False)  # noqa: S324
            with path.open("rb") as f:
                while chunk := f.read(65536):
                    h_sha256.update(chunk)
                    h_sha512.update(chunk)
                    h_md5.update(chunk)
            return [
                {"alg": "SHA-256", "content": h_sha256.hexdigest()},
                {"alg": "SHA-512", "content": h_sha512.hexdigest()},
                {"alg": "MD5", "content": h_md5.hexdigest()},
            ]
        except OSError:
            return []

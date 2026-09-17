"""Compliance-модули: маппинг на ФСТЭК, OWASP ML Top 10, ГОСТ Р 56939-2024."""

from poison_check.compliance.fstec_mapping import UBI_DESCRIPTIONS, UBI_MAPPINGS, FstecMapper
from poison_check.compliance.gost_mapping import GOST_56939_2024, ISSUE_TO_GOST, GostMapper
from poison_check.compliance.owasp_ml_top10 import ISSUE_TO_OWASP, OWASP_ML_TOP10, OwaspMapper

__all__ = [
    "FstecMapper",
    "UBI_MAPPINGS",
    "UBI_DESCRIPTIONS",
    "GostMapper",
    "GOST_56939_2024",
    "ISSUE_TO_GOST",
    "OwaspMapper",
    "OWASP_ML_TOP10",
    "ISSUE_TO_OWASP",
]

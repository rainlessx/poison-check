"""Пакет детекторов угроз ML-файлов.

Импорт этого пакета автоматически регистрирует все доступные детекторы
в DetectorRegistry через декоратор @DetectorRegistry.register.
"""

# Импортируем все детекторы, чтобы их декораторы @DetectorRegistry.register
# выполнились и детекторы оказались в глобальном реестре.
# Порядок важен: BlocklistDetector импортируется первым, так как
# AllowlistDetector зависит от его _HARDCODED_BLOCKLIST.
from poison_check.detectors.allowlist_detector import AllowlistDetector
from poison_check.detectors.blocklist_detector import BlocklistDetector
from poison_check.detectors.bundle_detector import BundleCodeDetector
from poison_check.detectors.compression_detector import CompressionDetector
from poison_check.detectors.cve_detector import CVEDetector
from poison_check.detectors.executable_detector import ExecutableDetector
from poison_check.detectors.format_policy_detector import FormatPolicyDetector
from poison_check.detectors.gguf_metadata_detector import GGUFMetadataDetector
from poison_check.detectors.joblib_metadata_detector import JoblibMetadataDetector
from poison_check.detectors.keras_detector import KerasThreatDetector
from poison_check.detectors.network_detector import NetworkDetector
from poison_check.detectors.numpy_metadata_detector import NumpyMetadataDetector
from poison_check.detectors.parse_error_detector import ParseErrorDetector
from poison_check.detectors.secrets_detector import SecretsDetector

__all__ = [
    "AllowlistDetector",
    "BlocklistDetector",
    "BundleCodeDetector",
    "CompressionDetector",
    "CVEDetector",
    "ExecutableDetector",
    "FormatPolicyDetector",
    "GGUFMetadataDetector",
    "JoblibMetadataDetector",
    "KerasThreatDetector",
    "NetworkDetector",
    "NumpyMetadataDetector",
    "ParseErrorDetector",
    "SecretsDetector",
]

"""Пакет сканеров ML-файлов.

Импорт этого пакета автоматически регистрирует все доступные сканеры
в ScannerRegistry через декоратор @ScannerRegistry.register.
"""

# Импортируем все сканеры, чтобы их декораторы @ScannerRegistry.register
# выполнились и сканеры оказались в глобальном реестре.
from poison_check.scanners.gguf_scanner import GGUFScanner
from poison_check.scanners.joblib_scanner import JoblibScanner
from poison_check.scanners.keras_scanner import KerasScanner
from poison_check.scanners.numpy_scanner import NumpyScanner
from poison_check.scanners.pickle_scanner import PickleScanner
from poison_check.scanners.pytorch_scanner import PyTorchScanner
from poison_check.scanners.safetensors_scanner import SafetensorsScanner

__all__ = [
    "GGUFScanner",
    "JoblibScanner",
    "KerasScanner",
    "NumpyScanner",
    "PickleScanner",
    "PyTorchScanner",
    "SafetensorsScanner",
]

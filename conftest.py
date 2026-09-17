"""Корневой conftest.py — добавляет project root в sys.path для pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

"""Общие фикстуры для тестов."""
import sys
from pathlib import Path

# Позволяем импортировать пакеты engine/strategies напрямую
sys.path.insert(0, str(Path(__file__).parent.parent))

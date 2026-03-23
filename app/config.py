from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
EXPORT_DIR = OUTPUT_DIR / "exports"
MODEL_DIR = BASE_DIR / "models" / "pretrained"
DEFAULT_MODEL_PATH = MODEL_DIR / "generator.pt"
APP_NAME = "基于神经网络的二维多孔介质重构系统"

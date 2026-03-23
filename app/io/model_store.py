from __future__ import annotations

"""兼容层：保留旧模块路径，转发到新的 ModelManager。"""

from pathlib import Path

from app.core.model_manager import ModelManager


def initialize_demo_model(path: Path) -> Path:
    """生成示例 CVAE 模型权重文件。"""
    manager = ModelManager(model_path=path)
    return manager.save_demo_weights(path)

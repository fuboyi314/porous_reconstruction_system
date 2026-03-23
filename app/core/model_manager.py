from __future__ import annotations

"""模型管理模块。"""

from pathlib import Path
from typing import Any

import torch

from app.config import DEFAULT_MODEL_PATH
from app.core.generator import ConditionalVAEReconstructor, build_default_model, load_state_dict_if_available


class ModelManager:
    """负责模型权重检查、加载与版本信息维护。"""

    def __init__(self, model_path: Path | None = None, device: str = "cpu") -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = device
        self.model: ConditionalVAEReconstructor = build_default_model(device=device)
        self.loaded_from_disk = False

    def model_exists(self) -> bool:
        """检查模型文件是否存在。"""
        return self.model_path.exists()

    def load_model(self) -> ConditionalVAEReconstructor:
        """加载模型权重；若不存在则保留示例随机权重。"""
        self.loaded_from_disk = load_state_dict_if_available(self.model, self.model_path, device=self.device)
        return self.model

    def get_version_info(self) -> dict[str, Any]:
        """返回模型版本与加载状态。"""
        return {
            "model_name": "ConditionalVAEReconstructor",
            "model_family": "CVAE",
            "model_path": str(self.model_path),
            "device": self.device,
            "weights_loaded": self.loaded_from_disk,
            "version": "0.3.0-cvae-demo",
        }

    def save_demo_weights(self, target_path: Path | None = None) -> Path:
        """保存当前示例模型权重，便于本地演示。"""
        path = target_path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

from __future__ import annotations

"""条件神经网络重构核心模块。"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from app.core.config import ReconstructionConfig
from app.core.postprocess import PostprocessConfig, postprocess_image

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ForwardOutput:
    """CVAE 前向过程的输出。"""

    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


@dataclass(slots=True)
class InferenceOutput:
    """模型推理结果。"""

    grayscale_image: np.ndarray
    binary_image: np.ndarray
    threshold: float


class ConditionEncoder(torch.nn.Module):
    """将重构条件编码为紧凑条件向量。"""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, embedding_dim: int = 32) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
            torch.nn.ReLU(),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """编码条件张量。"""
        return self.network(condition)


class ConditionalVAEEncoder(torch.nn.Module):
    """CVAE 编码器，将图像与条件嵌入映射为潜变量分布。"""

    def __init__(self, condition_dim: int = 32, latent_dim: int = 32) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.mu_head = torch.nn.Linear(32 * 8 * 8 + condition_dim, latent_dim)
        self.logvar_head = torch.nn.Linear(32 * 8 * 8 + condition_dim, latent_dim)

    def forward(self, image: torch.Tensor, condition_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """输出潜变量分布参数。"""
        feature = self.features(image).flatten(start_dim=1)
        feature = torch.cat([feature, condition_embedding], dim=1)
        return self.mu_head(feature), self.logvar_head(feature)


class ConditionalVAEDecoder(torch.nn.Module):
    """CVAE 解码器，将潜变量与条件嵌入解码为二维灰度图。"""

    def __init__(self, latent_dim: int = 32, condition_dim: int = 32, base_size: int = 16) -> None:
        super().__init__()
        self.base_size = base_size
        self.project = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + condition_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64 * base_size * base_size),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        condition_embedding: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """输出指定尺寸的灰度图。"""
        merged = torch.cat([latent, condition_embedding], dim=1)
        feature_map = self.project(merged)
        feature_map = feature_map.view(-1, 64, self.base_size, self.base_size)
        decoded = self.decoder(feature_map)
        return torch.nn.functional.interpolate(decoded, size=image_size, mode="bilinear", align_corners=False)


class ConditionalVAEReconstructor(torch.nn.Module):
    """用于二维多孔介质重构的条件变分自编码器。"""

    def __init__(self, latent_dim: int = 32, condition_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_encoder = ConditionEncoder(embedding_dim=condition_dim)
        self.encoder = ConditionalVAEEncoder(condition_dim=condition_dim, latent_dim=latent_dim)
        self.decoder = ConditionalVAEDecoder(latent_dim=latent_dim, condition_dim=condition_dim)

    @staticmethod
    def _normalize_condition(config: ReconstructionConfig) -> list[float]:
        """将配置归一化为模型条件向量。"""
        width_norm = float(max(config.image_width, 1))
        return [
            float(config.porosity),
            float(config.pore_size_mean / width_norm),
            float(config.pore_size_std / width_norm),
            float(config.specific_surface_area),
            float(config.coordination_number / 10.0),
        ]

    def encode_condition(
        self,
        condition: ReconstructionConfig | Sequence[float],
        device: torch.device,
    ) -> torch.Tensor:
        """将配置对象或原始条件序列编码为张量嵌入。"""
        if isinstance(condition, ReconstructionConfig):
            vector = self._normalize_condition(condition)
        else:
            vector = [float(value) for value in condition]
        condition_tensor = torch.tensor([vector], dtype=torch.float32, device=device)
        return self.condition_encoder(condition_tensor)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数采样。"""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, image: torch.Tensor, condition: ReconstructionConfig | Sequence[float]) -> ForwardOutput:
        """训练/微调时使用的完整前向过程。"""
        device = image.device
        image_size = (image.shape[-2], image.shape[-1])
        condition_embedding = self.encode_condition(condition, device=device)
        mu, logvar = self.encoder(image, condition_embedding)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(latent, condition_embedding, image_size=image_size)
        return ForwardOutput(reconstruction=reconstruction, mu=mu, logvar=logvar)

    def infer(self, condition: ReconstructionConfig, seed: int | None = None, device: str = "cpu") -> np.ndarray:
        """根据条件参数执行推理并返回二维灰度图。"""
        runtime_device = torch.device(device)
        self.to(runtime_device)
        self.eval()

        actual_seed = condition.seed if seed is None else seed
        torch.manual_seed(actual_seed)
        np.random.seed(actual_seed)

        condition_embedding = self.encode_condition(condition, device=runtime_device)
        latent = torch.randn(1, self.latent_dim, device=runtime_device)
        with torch.no_grad():
            grayscale = self.decoder(
                latent,
                condition_embedding,
                image_size=(condition.image_height, condition.image_width),
            )
        return grayscale.squeeze().detach().cpu().numpy().astype(np.float32)


class PorousMediaInferenceEngine:
    """封装模型推理、灰度图输出与后处理流程。"""

    def __init__(
        self,
        model: ConditionalVAEReconstructor,
        device: str = "cpu",
        postprocess_config: PostprocessConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.postprocess_config = postprocess_config or PostprocessConfig()

    def infer(self, config: ReconstructionConfig, seed: int | None = None) -> InferenceOutput:
        """返回灰度图、稳定二值图及阈值。"""
        grayscale = self.model.infer(condition=config, seed=seed, device=self.device)
        binary, threshold = postprocess_image(
            grayscale,
            config=self.postprocess_config,
            porosity_target=config.porosity,
        )
        return InferenceOutput(grayscale_image=grayscale, binary_image=binary, threshold=threshold)


def load_state_dict_if_available(model: torch.nn.Module, model_path: Path, device: str = "cpu") -> bool:
    """如果存在模型权重则加载。"""
    if not model_path.exists():
        LOGGER.info("模型权重不存在，继续使用随机初始化权重：%s", model_path)
        return False

    state = torch.load(model_path, map_location=torch.device(device))
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    LOGGER.info("模型权重加载成功：%s", model_path)
    return True


def build_default_model(device: str = "cpu") -> ConditionalVAEReconstructor:
    """创建默认的 CVAE 重构模型。"""
    model = ConditionalVAEReconstructor()
    model.to(torch.device(device))
    return model

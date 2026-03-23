from __future__ import annotations

"""重构业务服务。"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from app.core.analyzer import generate_brief_analysis, generate_detailed_analysis
from app.core.config import ReconstructionConfig
from app.core.dto import ReconstructionResult
from app.core.generator import PorousMediaInferenceEngine
from app.core.metrics import compare_with_targets, compute_metrics
from app.core.model_manager import ModelManager
from app.core.postprocess import PostprocessConfig, postprocess_image

LOGGER = logging.getLogger(__name__)


class ReconstructionService:
    """负责串联参数校验、推理、后处理、分析与结果组织。"""

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cpu",
        postprocess_config: PostprocessConfig | None = None,
    ) -> None:
        self.model_manager = ModelManager(model_path=model_path, device=device)
        model = self.model_manager.load_model()
        self.inference_engine = PorousMediaInferenceEngine(
            model=model,
            device=device,
            postprocess_config=postprocess_config,
        )

    def _procedural_grayscale_candidates(self, config: ReconstructionConfig) -> list[np.ndarray]:
        """当缺少正式权重时，生成更贴近目标结构尺度的候选灰度图。"""
        rng = np.random.default_rng(config.seed)
        candidates: list[np.ndarray] = []
        base_noise = rng.random((config.image_height, config.image_width), dtype=np.float32)
        scales = [0.45, 0.65, 0.85, 1.05, 1.25]
        for scale in scales:
            sigma = max(config.pore_size_mean * scale / 6.0, 1.0)
            smoothed = gaussian_filter(base_noise, sigma=sigma)
            smoothed = (smoothed - smoothed.min()) / max(smoothed.ptp(), 1e-6)
            candidates.append(smoothed.astype(np.float32))
        return candidates

    def _candidate_postprocess_configs(self, config: ReconstructionConfig) -> list[PostprocessConfig]:
        """生成用于误差搜索的后处理配置集合。"""
        min_size = max(int(config.pore_size_mean // 2), 4)
        return [
            PostprocessConfig(
                threshold_mode="auto",
                remove_small_objects_enabled=True,
                min_object_size=min_size,
                hole_strategy="preserve",
                smoothing_mode="median",
                median_size=3,
            ),
            PostprocessConfig(
                threshold_mode="auto",
                remove_small_objects_enabled=True,
                min_object_size=min_size * 2,
                hole_strategy="fill_small",
                max_hole_size=max(int(config.pore_size_mean**2), 16),
                smoothing_mode="open_close",
                morph_radius=max(int(config.pore_size_std), 1),
            ),
            PostprocessConfig(
                threshold_mode="otsu",
                remove_small_objects_enabled=True,
                min_object_size=min_size,
                hole_strategy="fill_small",
                max_hole_size=max(int(config.pore_size_mean**2 / 2), 8),
                boundary_mode="crop",
                boundary_width=1,
                smoothing_mode="close_open",
                morph_radius=max(int(config.pore_size_std), 1),
            ),
        ]

    @staticmethod
    def _comparison_score(comparison_dict: dict[str, Any]) -> float:
        """根据偏差结果计算综合评分，分数越小越好。"""
        weights = {
            "porosity": 5.0,
            "pore_size_mean": 2.0,
            "pore_size_std": 1.5,
            "specific_surface_area": 1.5,
            "coordination_number": 1.5,
        }
        score = 0.0
        for key, weight in weights.items():
            item = comparison_dict.get(key, {})
            if hasattr(item, "to_dict"):
                item = item.to_dict()
            relative_error = abs(float(item.get("relative_error", 1.0)))
            score += weight * relative_error
        return score

    def _select_best_structure(
        self,
        config: ReconstructionConfig,
        model_grayscale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
        """在候选灰度图与后处理参数中搜索误差更小的结果。"""
        grayscale_candidates = [model_grayscale]
        if not self.model_manager.loaded_from_disk:
            grayscale_candidates.extend(self._procedural_grayscale_candidates(config))

        best_binary: np.ndarray | None = None
        best_grayscale: np.ndarray | None = None
        best_score = float("inf")
        best_comparison: dict[str, Any] = {}

        for grayscale in grayscale_candidates:
            for postprocess_config in self._candidate_postprocess_configs(config):
                binary, _ = postprocess_image(grayscale, config=postprocess_config, porosity_target=config.porosity)
                metrics = compute_metrics(binary)
                comparison = compare_with_targets(metrics, config).to_dict()
                score = self._comparison_score(comparison)
                if score < best_score:
                    best_score = score
                    best_binary = binary
                    best_grayscale = grayscale
                    best_comparison = comparison

        if best_binary is None or best_grayscale is None:
            raise RuntimeError("未能生成有效的候选重构结构。")
        return best_grayscale, best_binary, best_comparison, best_score

    def run(self, config: ReconstructionConfig) -> ReconstructionResult:
        """执行完整重构业务流程。"""
        LOGGER.info("参数提交：%s", config.to_dict())
        config.validate()

        LOGGER.info("重构开始。")
        inference_output = self.inference_engine.infer(config=config, seed=config.seed)
        grayscale_image, binary_image, comparison_dict, score = self._select_best_structure(
            config=config,
            model_grayscale=inference_output.grayscale_image,
        )
        metrics = compute_metrics(binary_image)
        analysis_text = generate_brief_analysis(config, metrics, comparison_dict)
        detailed_analysis_text = generate_detailed_analysis(config, metrics, comparison_dict)
        model_info = self.model_manager.get_version_info()
        model_info["calibration_score"] = score
        LOGGER.info("分析结果摘要：%s", comparison_dict)
        LOGGER.info("重构结束。")

        return ReconstructionResult(
            config=config,
            metrics=metrics,
            analysis_text=analysis_text,
            detailed_analysis_text=detailed_analysis_text,
            comparison=comparison_dict,
            grayscale_image=grayscale_image,
            binary_image=binary_image,
            model_info=model_info,
        )

    @staticmethod
    def build_export_payload(result: ReconstructionResult) -> dict[str, Any]:
        """构建导出模块可直接使用的完整结果数据结构。"""
        return result.to_dict()

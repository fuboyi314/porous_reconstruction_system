from __future__ import annotations

"""重构业务服务。"""

import logging
from pathlib import Path
from typing import Any

from app.core.analyzer import generate_brief_analysis, generate_detailed_analysis
from app.core.config import ReconstructionConfig
from app.core.dto import ReconstructionResult
from app.core.generator import PorousMediaInferenceEngine
from app.core.metrics import compare_with_targets, compute_metrics
from app.core.model_manager import ModelManager
from app.core.postprocess import PostprocessConfig

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

    def run(self, config: ReconstructionConfig) -> ReconstructionResult:
        """执行完整重构业务流程。"""
        LOGGER.info("参数提交：%s", config.to_dict())
        config.validate()

        LOGGER.info("重构开始。")
        inference_output = self.inference_engine.infer(config=config, seed=config.seed)
        metrics = compute_metrics(inference_output.binary_image)
        comparison = compare_with_targets(metrics, config)
        analysis_text = generate_brief_analysis(config, metrics, comparison)
        detailed_analysis_text = generate_detailed_analysis(config, metrics, comparison)
        model_info = self.model_manager.get_version_info()
        LOGGER.info("分析结果摘要：%s", comparison.to_dict())
        LOGGER.info("重构结束。")

        return ReconstructionResult(
            config=config,
            metrics=metrics,
            analysis_text=analysis_text,
            detailed_analysis_text=detailed_analysis_text,
            comparison=comparison.to_dict(),
            grayscale_image=inference_output.grayscale_image,
            binary_image=inference_output.binary_image,
            model_info=model_info,
        )

    @staticmethod
    def build_export_payload(result: ReconstructionResult) -> dict[str, Any]:
        """构建导出模块可直接使用的完整结果数据结构。"""
        return result.to_dict()

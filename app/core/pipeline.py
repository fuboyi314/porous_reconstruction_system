from __future__ import annotations

from pathlib import Path

from app.core.config import ReconstructionConfig
from app.core.dto import ReconstructionResult
from app.core.postprocess import PostprocessConfig
from app.core.service import ReconstructionService


class ReconstructionPipeline:
    """兼容层：保留旧入口，内部委托给 ReconstructionService。"""

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cpu",
        postprocess_config: PostprocessConfig | None = None,
    ) -> None:
        self.service = ReconstructionService(
            model_path=model_path,
            device=device,
            postprocess_config=postprocess_config,
        )

    def run(self, config: ReconstructionConfig) -> ReconstructionResult:
        return self.service.run(config)

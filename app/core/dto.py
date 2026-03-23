from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.config import ReconstructionConfig


@dataclass(slots=True)
class ComputedMetrics:
    porosity: float
    pore_size_mean: float
    pore_size_std: float
    specific_surface_area: float
    coordination_number: float
    pore_size_histogram: list[float] = field(default_factory=list)
    pore_size_bin_edges: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReconstructionResult:
    config: ReconstructionConfig
    metrics: ComputedMetrics
    analysis_text: str
    detailed_analysis_text: str = ""
    comparison: dict[str, Any] = field(default_factory=dict)
    grayscale_image: Any = None
    binary_image: Any = None
    model_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "analysis_text": self.analysis_text,
            "detailed_analysis_text": self.detailed_analysis_text,
            "comparison": self.comparison,
            "model_info": self.model_info,
        }

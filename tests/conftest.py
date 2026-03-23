from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import ReconstructionConfig
from app.core.dto import ComputedMetrics, ReconstructionResult


@pytest.fixture()
def sample_result() -> ReconstructionResult:
    np = pytest.importorskip("numpy")
    binary = np.zeros((8, 8), dtype=np.uint8)
    binary[2:6, 2:6] = 1
    grayscale = binary.astype(np.float32)
    return ReconstructionResult(
        config=ReconstructionConfig(
            porosity=0.25,
            pore_size_mean=2.0,
            pore_size_std=0.5,
            specific_surface_area=0.3,
            coordination_number=2.0,
            image_width=8,
            image_height=8,
            seed=1,
            sample_count=1,
        ),
        metrics=ComputedMetrics(
            porosity=0.25,
            pore_size_mean=2.0,
            pore_size_std=0.5,
            specific_surface_area=0.3,
            coordination_number=2.0,
            pore_size_histogram=[0.4, 0.6],
            pore_size_bin_edges=[0.0, 1.0, 2.0],
        ),
        analysis_text="简洁分析",
        detailed_analysis_text="详细分析",
        comparison={"porosity": {"delta": 0.0, "status": "matched"}},
        grayscale_image=grayscale,
        binary_image=binary,
        model_info={"model_name": "demo", "version": "0.1"},
    )

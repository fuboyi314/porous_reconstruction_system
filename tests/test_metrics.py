from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from app.core.config import ReconstructionConfig
from app.core.metrics import (
    compare_with_targets,
    compute_metrics,
    compute_porosity,
    compute_specific_surface_area_2d,
    estimate_coordination_number,
    estimate_pore_size_distribution,
)


def test_compute_porosity_returns_fraction() -> None:
    image = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert compute_porosity(image) == pytest.approx(0.5)


def test_estimate_pore_size_distribution_returns_structured_result() -> None:
    image = np.zeros((7, 7), dtype=np.uint8)
    image[2:5, 2:5] = 1
    distribution = estimate_pore_size_distribution(image, bins=4)
    assert distribution.mean > 0.0
    assert distribution.std >= 0.0
    assert len(distribution.histogram) == 4
    assert len(distribution.bin_edges) == 5
    assert sum(distribution.histogram) == pytest.approx(1.0)


def test_compute_specific_surface_area_2d_is_positive_for_isolated_pore() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 1
    assert compute_specific_surface_area_2d(image) > 0.0


def test_estimate_coordination_number_detects_branching() -> None:
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 1:6] = 1
    image[1:6, 3] = 1
    coordination = estimate_coordination_number(image)
    assert coordination >= 3.0


def test_compare_with_targets_returns_structured_mapping() -> None:
    image = np.zeros((7, 7), dtype=np.uint8)
    image[2:5, 2:5] = 1
    actual = compute_metrics(image)
    target = ReconstructionConfig(
        porosity=0.2,
        pore_size_mean=1.0,
        pore_size_std=0.5,
        specific_surface_area=0.1,
        coordination_number=1.0,
        image_width=7,
        image_height=7,
        seed=1,
        sample_count=1,
    )
    comparison = compare_with_targets(actual, target)
    assert set(comparison.items) == {
        "porosity",
        "pore_size_mean",
        "pore_size_std",
        "specific_surface_area",
        "coordination_number",
    }
    assert comparison.items["porosity"].name == "孔隙率"
    assert isinstance(comparison.items["porosity"].delta, float)

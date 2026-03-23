from __future__ import annotations

from app.core.analyzer import generate_brief_analysis, generate_detailed_analysis
from app.core.config import ReconstructionConfig
from app.core.dto import ComputedMetrics


def _sample_actual() -> ComputedMetrics:
    return ComputedMetrics(
        porosity=0.34,
        pore_size_mean=11.5,
        pore_size_std=2.8,
        specific_surface_area=0.17,
        coordination_number=2.6,
        pore_size_histogram=[0.2, 0.3, 0.5],
        pore_size_bin_edges=[0.0, 1.0, 2.0, 3.0],
    )


def _sample_target() -> ReconstructionConfig:
    return ReconstructionConfig(
        porosity=0.35,
        pore_size_mean=12.0,
        pore_size_std=3.0,
        specific_surface_area=0.18,
        coordination_number=2.8,
        image_width=256,
        image_height=256,
        seed=42,
        sample_count=1,
    )


def _sample_comparison() -> dict[str, dict[str, float | str]]:
    return {
        "porosity": {"delta": -0.01, "status": "lower", "relative_error": -0.0286},
        "pore_size_mean": {"delta": -0.5, "status": "lower", "relative_error": -0.0417},
        "pore_size_std": {"delta": -0.2, "status": "lower", "relative_error": -0.0667},
        "specific_surface_area": {"delta": -0.01, "status": "lower", "relative_error": -0.0556},
        "coordination_number": {"delta": -0.2, "status": "lower", "relative_error": -0.0714},
    }


def test_generate_brief_analysis_contains_core_sections() -> None:
    text = generate_brief_analysis(_sample_target(), _sample_actual(), _sample_comparison())
    assert "本次重构面向二维多孔介质目标结构参数匹配任务" in text
    assert "孔隙率目标为" in text
    assert "孔径分布方面" in text
    assert "比表面积" in text
    assert "配位数" in text


def test_generate_detailed_analysis_contains_overall_evaluation() -> None:
    text = generate_detailed_analysis(_sample_target(), _sample_actual(), _sample_comparison())
    assert "重构分析报告：" in text
    assert "目标参数概述" in text
    assert "孔隙率" in text
    assert "平均孔径" in text
    assert "总体评价：" in text


def test_generate_analysis_handles_missing_values_gracefully() -> None:
    actual = {"porosity": 0.31}
    target = {"porosity": 0.35}
    text = generate_brief_analysis(target, actual, comparison={"porosity": {"delta": -0.04, "status": "lower"}})
    assert "缺失" in text or "暂无法" in text or "尚无" in text

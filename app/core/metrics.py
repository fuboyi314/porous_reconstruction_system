from __future__ import annotations

"""二维二值多孔介质结构分析模块。"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter
from skimage.morphology import skeletonize

from app.core.dto import ComputedMetrics


@dataclass(slots=True)
class PoreSizeDistribution:
    """孔径分布统计结果。"""

    mean: float
    std: float
    histogram: list[float]
    bin_edges: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetricComparisonItem:
    """单个指标的目标对比结果。"""

    name: str
    actual: float
    target: float
    delta: float
    relative_error: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetricComparison:
    """多个指标的目标对比结果。"""

    items: dict[str, MetricComparisonItem]

    def to_dict(self) -> dict[str, Any]:
        return {key: item.to_dict() for key, item in self.items.items()}


def _to_binary(image: np.ndarray) -> np.ndarray:
    """确保输入转换为 0/1 二值孔隙图。"""
    return (np.asarray(image) > 0).astype(np.uint8)


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(values.std()) if values.size else 0.0


def compute_porosity(image: np.ndarray) -> float:
    """计算孔隙率。输入中 1 表示孔隙，0 表示固体。"""
    pore = _to_binary(image)
    return float(pore.mean())


def estimate_pore_size_distribution(image: np.ndarray, bins: int = 12) -> PoreSizeDistribution:
    """利用欧氏距离变换估计孔径分布。"""
    pore = _to_binary(image).astype(bool)
    distance_map = ndi.distance_transform_edt(pore)
    pore_radii = distance_map[distance_map > 0]
    histogram, bin_edges = np.histogram(pore_radii, bins=bins, density=False)
    histogram = histogram.astype(float)
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return PoreSizeDistribution(
        mean=_safe_mean(pore_radii),
        std=_safe_std(pore_radii),
        histogram=histogram.tolist(),
        bin_edges=bin_edges.tolist(),
    )


def compute_specific_surface_area_2d(image: np.ndarray) -> float:
    """基于孔隙-固体边界长度近似计算二维比表面积。"""
    pore = _to_binary(image)
    pore_area = max(float(pore.sum()), 1.0)
    boundary_length = perimeter(pore, neighborhood=8)
    return float(boundary_length / pore_area)


def estimate_coordination_number(image: np.ndarray) -> float:
    """基于骨架与 8 邻域连接关系估计配位数。"""
    skeleton = skeletonize(_to_binary(image).astype(bool))
    if not skeleton.any():
        return 0.0

    padded = np.pad(skeleton.astype(np.uint8), 1, mode="constant")
    ys, xs = np.where(padded == 1)
    neighbor_counts: list[int] = []
    for y, x in zip(ys, xs, strict=False):
        window = padded[y - 1 : y + 2, x - 1 : x + 2]
        neighbor_counts.append(int(window.sum()) - 1)

    neighbor_counts_array = np.asarray(neighbor_counts, dtype=float)
    branch_nodes = neighbor_counts_array[neighbor_counts_array >= 3]
    if branch_nodes.size:
        return float(branch_nodes.mean())
    return float(neighbor_counts_array.mean())


def compare_with_targets(
    actual: ComputedMetrics | Mapping[str, float],
    target: Mapping[str, float] | Any,
) -> MetricComparison:
    """比较实际结构指标与目标值。"""
    actual_mapping: Mapping[str, float]
    if isinstance(actual, ComputedMetrics):
        actual_mapping = {
            "porosity": actual.porosity,
            "pore_size_mean": actual.pore_size_mean,
            "pore_size_std": actual.pore_size_std,
            "specific_surface_area": actual.specific_surface_area,
            "coordination_number": actual.coordination_number,
        }
    else:
        actual_mapping = actual

    if hasattr(target, "to_dict"):
        target_mapping = target.to_dict()
    else:
        target_mapping = dict(target)

    label_map = {
        "porosity": "孔隙率",
        "pore_size_mean": "平均孔径",
        "pore_size_std": "孔径标准差",
        "specific_surface_area": "比表面积",
        "coordination_number": "配位数",
    }

    items: dict[str, MetricComparisonItem] = {}
    for key, label in label_map.items():
        actual_value = float(actual_mapping[key])
        target_value = float(target_mapping[key])
        delta = actual_value - target_value
        relative_error = delta / target_value if abs(target_value) > 1e-12 else 0.0
        if abs(delta) < 1e-3:
            status = "matched"
        elif delta > 0:
            status = "higher"
        else:
            status = "lower"
        items[key] = MetricComparisonItem(
            name=label,
            actual=actual_value,
            target=target_value,
            delta=delta,
            relative_error=relative_error,
            status=status,
        )
    return MetricComparison(items=items)


def compute_metrics(binary_image: np.ndarray, bins: int = 12) -> ComputedMetrics:
    """汇总计算 GUI 需要展示的结构分析指标。"""
    distribution = estimate_pore_size_distribution(binary_image, bins=bins)
    return ComputedMetrics(
        porosity=compute_porosity(binary_image),
        pore_size_mean=distribution.mean,
        pore_size_std=distribution.std,
        specific_surface_area=compute_specific_surface_area_2d(binary_image),
        coordination_number=estimate_coordination_number(binary_image),
        pore_size_histogram=distribution.histogram,
        pore_size_bin_edges=distribution.bin_edges,
    )

from __future__ import annotations

"""二维多孔介质重构结果后处理模块。"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects

ThresholdMode = Literal["auto", "otsu", "manual"]
HoleStrategy = Literal["preserve", "fill_small", "fill_all"]
BoundaryMode = Literal["none", "crop", "protect"]
SmoothMode = Literal["none", "median", "open_close", "close_open"]


@dataclass(slots=True)
class PostprocessConfig:
    """后处理参数，适合直接映射到 GUI 控件。"""

    threshold_mode: ThresholdMode = "auto"
    manual_threshold: float = 0.5
    remove_small_objects_enabled: bool = True
    min_object_size: int = 16
    hole_strategy: HoleStrategy = "fill_small"
    max_hole_size: int = 64
    boundary_mode: BoundaryMode = "none"
    boundary_width: int = 0
    smoothing_mode: SmoothMode = "median"
    median_size: int = 3
    morph_radius: int = 1
    preserve_pore_phase: bool = True
    force_threshold: bool = False


def _is_binary_image(image: np.ndarray) -> bool:
    values = np.unique(image)
    return values.size <= 2 and np.all(np.isin(values, [0, 1, False, True]))


def _normalize_binary(image: np.ndarray) -> np.ndarray:
    return (image > 0).astype(np.uint8)


def threshold_image(
    image: np.ndarray,
    config: PostprocessConfig,
    porosity_target: float | None = None,
) -> tuple[np.ndarray, float]:
    """对二维灰度图进行阈值分割。"""
    image = np.asarray(image, dtype=np.float32)
    if _is_binary_image(image) and not config.force_threshold:
        binary = _normalize_binary(image)
        threshold = 0.5
        return binary, threshold

    if config.threshold_mode == "manual":
        threshold = float(config.manual_threshold)
    elif config.threshold_mode == "otsu":
        threshold = float(threshold_otsu(image))
    else:
        if porosity_target is None:
            threshold = float(threshold_otsu(image))
        else:
            quantile = float(np.clip(1.0 - porosity_target, 0.01, 0.99))
            threshold = float(np.quantile(image, quantile))

    binary = (image < threshold).astype(np.uint8)
    return binary, threshold


def remove_small_connected_domains(binary: np.ndarray, min_size: int, pore_value: int = 1) -> np.ndarray:
    """去除小连通域噪声。"""
    phase = binary == pore_value
    cleaned = remove_small_objects(phase, min_size=max(min_size, 1))
    result = np.zeros_like(binary, dtype=np.uint8)
    result[cleaned] = pore_value
    return result


def apply_hole_strategy(binary: np.ndarray, strategy: HoleStrategy, max_hole_size: int) -> np.ndarray:
    """根据策略填补或保留小孔洞。"""
    pore = binary.astype(bool)
    if strategy == "preserve":
        result = pore
    elif strategy == "fill_all":
        result = remove_small_holes(pore, area_threshold=pore.size)
    else:
        result = remove_small_holes(pore, area_threshold=max(max_hole_size, 1))
    return result.astype(np.uint8)


def apply_boundary_mode(binary: np.ndarray, reference: np.ndarray, mode: BoundaryMode, width: int) -> np.ndarray:
    """执行边界裁剪或边界保护。"""
    if mode == "none" or width <= 0:
        return _normalize_binary(binary)

    result = _normalize_binary(binary)
    width = int(width)
    if mode == "crop":
        result[:width, :] = 0
        result[-width:, :] = 0
        result[:, :width] = 0
        result[:, -width:] = 0
        return result

    reference = _normalize_binary(reference)
    result[:width, :] = reference[:width, :]
    result[-width:, :] = reference[-width:, :]
    result[:, :width] = reference[:, :width]
    result[:, -width:] = reference[:, -width:]
    return result


def smooth_binary(binary: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    """使用中值滤波或形态学开闭操作平滑结构。"""
    pore = binary.astype(bool)
    radius = max(int(config.morph_radius), 1)
    footprint = disk(radius)

    if config.smoothing_mode == "median":
        size = max(int(config.median_size), 1)
        filtered = median_filter(binary.astype(np.uint8), size=size)
        return _normalize_binary(filtered)
    if config.smoothing_mode == "open_close":
        return binary_closing(binary_opening(pore, footprint), footprint).astype(np.uint8)
    if config.smoothing_mode == "close_open":
        return binary_opening(binary_closing(pore, footprint), footprint).astype(np.uint8)
    return _normalize_binary(binary)


def postprocess_image(
    image: np.ndarray,
    config: PostprocessConfig | None = None,
    porosity_target: float | None = None,
) -> tuple[np.ndarray, float]:
    """对神经网络输出图像执行完整后处理。"""
    runtime_config = config or PostprocessConfig()
    binary, threshold = threshold_image(image=image, config=runtime_config, porosity_target=porosity_target)
    reference = binary.copy()

    if runtime_config.remove_small_objects_enabled:
        pore_value = 1 if runtime_config.preserve_pore_phase else 0
        binary = remove_small_connected_domains(binary, min_size=runtime_config.min_object_size, pore_value=pore_value)
        if pore_value == 0:
            binary = 1 - binary

    binary = apply_hole_strategy(binary, runtime_config.hole_strategy, runtime_config.max_hole_size)
    binary = smooth_binary(binary, runtime_config)
    binary = apply_boundary_mode(binary, reference=reference, mode=runtime_config.boundary_mode, width=runtime_config.boundary_width)
    return _normalize_binary(binary), threshold


if __name__ == "__main__":
    example = np.random.rand(128, 128).astype(np.float32)
    config = PostprocessConfig(
        threshold_mode="auto",
        remove_small_objects_enabled=True,
        min_object_size=24,
        hole_strategy="fill_small",
        max_hole_size=48,
        boundary_mode="crop",
        boundary_width=2,
        smoothing_mode="median",
        median_size=3,
    )
    binary, threshold = postprocess_image(example, config=config, porosity_target=0.35)
    print("postprocess done", binary.shape, binary.dtype, int(binary.min()), int(binary.max()), threshold)

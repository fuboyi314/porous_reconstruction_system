from __future__ import annotations

"""任务配置与参数校验模块。"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class ConfigValidationError(ValueError):
    """参数配置不合法时抛出的异常。"""


@dataclass(slots=True)
class ReconstructionConfig:
    """二维多孔介质重构任务配置。"""

    porosity: float = 0.35
    pore_size_mean: float = 12.0
    pore_size_std: float = 3.0
    specific_surface_area: float = 0.18
    coordination_number: float = 2.8
    image_width: int = 256
    image_height: int = 256
    seed: int = 42
    sample_count: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """校验配置参数是否合法。"""
        if not (0.0 < float(self.porosity) < 1.0):
            raise ConfigValidationError("孔隙率 porosity 必须位于 (0, 1) 范围内。")
        if self.pore_size_mean <= 0:
            raise ConfigValidationError("孔径均值 pore_size_mean 必须为正数。")
        if self.pore_size_std <= 0:
            raise ConfigValidationError("孔径标准差 pore_size_std 必须为正数。")
        if self.pore_size_std > self.pore_size_mean * 10:
            raise ConfigValidationError("孔径标准差 pore_size_std 非法，不能显著大于孔径均值。")
        if self.specific_surface_area <= 0:
            raise ConfigValidationError("比表面积 specific_surface_area 必须为正数。")
        if self.coordination_number <= 0:
            raise ConfigValidationError("配位数 coordination_number 必须为正数。")
        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ConfigValidationError("图像宽度 image_width 必须为正整数。")
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ConfigValidationError("图像高度 image_height 必须为正整数。")
        if not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ConfigValidationError("样本数量 sample_count 必须大于等于 1。")
        if not isinstance(self.seed, int):
            raise ConfigValidationError("随机种子 seed 必须为整数。")

    def to_dict(self) -> dict[str, Any]:
        """导出为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReconstructionConfig":
        """从字典恢复配置对象。"""
        fields = {
            "porosity",
            "pore_size_mean",
            "pore_size_std",
            "specific_surface_area",
            "coordination_number",
            "image_width",
            "image_height",
            "seed",
            "sample_count",
        }
        payload = {key: data[key] for key in fields if key in data}
        return cls(**payload)

    def condition_vector(self) -> list[float]:
        """返回模型推理使用的条件向量。"""
        return [
            float(self.porosity),
            float(self.pore_size_mean),
            float(self.pore_size_std),
            float(self.specific_surface_area),
            float(self.coordination_number),
        ]

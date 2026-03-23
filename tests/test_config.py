from __future__ import annotations

import re

import pytest

from app.core.config import ConfigValidationError, ReconstructionConfig


def test_default_config_is_valid() -> None:
    config = ReconstructionConfig()
    assert config.porosity == pytest.approx(0.35)
    assert config.image_width == 256
    assert config.sample_count == 1


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("porosity", 0.0, "孔隙率 porosity 必须位于 (0, 1) 范围内。"),
        ("porosity", 1.0, "孔隙率 porosity 必须位于 (0, 1) 范围内。"),
        ("image_width", 0, "图像宽度 image_width 必须为正整数。"),
        ("image_height", -1, "图像高度 image_height 必须为正整数。"),
        ("sample_count", 0, "样本数量 sample_count 必须大于等于 1。"),
        ("specific_surface_area", 0.0, "比表面积 specific_surface_area 必须为正数。"),
        ("coordination_number", 0.0, "配位数 coordination_number 必须为正数。"),
        ("pore_size_mean", 0.0, "孔径均值 pore_size_mean 必须为正数。"),
        ("pore_size_std", 0.0, "孔径标准差 pore_size_std 必须为正数。"),
    ],
)
def test_invalid_values_raise_clear_errors(field_name: str, value: float, message: str) -> None:
    payload = ReconstructionConfig().to_dict()
    payload[field_name] = value
    with pytest.raises(ConfigValidationError, match=re.escape(message)):
        ReconstructionConfig.from_dict(payload)


def test_excessive_std_raises_error() -> None:
    with pytest.raises(ConfigValidationError, match="孔径标准差 pore_size_std 非法"):
        ReconstructionConfig(pore_size_mean=1.0, pore_size_std=20.0)


def test_to_dict_and_from_dict_roundtrip() -> None:
    config = ReconstructionConfig(
        porosity=0.41,
        pore_size_mean=9.5,
        pore_size_std=2.1,
        specific_surface_area=0.23,
        coordination_number=3.4,
        image_width=128,
        image_height=192,
        seed=123,
        sample_count=4,
    )
    restored = ReconstructionConfig.from_dict(config.to_dict())
    assert restored == config


def test_condition_vector_order() -> None:
    config = ReconstructionConfig()
    assert config.condition_vector() == [0.35, 12.0, 3.0, 0.18, 2.8]

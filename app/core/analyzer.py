from __future__ import annotations

"""分析文字自动生成模块。"""

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from app.core.config import ReconstructionConfig
from app.core.dto import ComputedMetrics


def _to_mapping(value: Any) -> dict[str, Any]:
    """将 dataclass / 映射对象统一转换为字典。"""
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value.__dict__.copy() if hasattr(value, "__dict__") else {}


def _comparison_mapping(comparison: Any, actual: Any, target: Any) -> dict[str, Any]:
    """统一获取偏差结果字典；若缺失则尝试按需计算。"""
    if comparison is not None:
        if hasattr(comparison, "to_dict"):
            return dict(comparison.to_dict())
        if isinstance(comparison, Mapping):
            return dict(comparison)

    try:
        from app.core.metrics import compare_with_targets
    except Exception:
        return {}

    try:
        computed = compare_with_targets(actual, target)
        if hasattr(computed, "to_dict"):
            return dict(computed.to_dict())
        if isinstance(computed, Mapping):
            return dict(computed)
    except Exception:
        return {}
    return {}


def _metric_entry(
    key: str,
    label: str,
    target_mapping: Mapping[str, Any],
    actual_mapping: Mapping[str, Any],
    comparison_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """获取单个指标的目标、实际与偏差信息。"""
    target_value = target_mapping.get(key)
    actual_value = actual_mapping.get(key)

    comparison_item = comparison_mapping.get(key, {}) if comparison_mapping else {}
    if hasattr(comparison_item, "to_dict"):
        comparison_item = comparison_item.to_dict()

    delta = comparison_item.get("delta")
    status = comparison_item.get("status")
    relative_error = comparison_item.get("relative_error")
    return {
        "label": label,
        "target": target_value,
        "actual": actual_value,
        "delta": delta,
        "status": status,
        "relative_error": relative_error,
    }


def _format_value(value: Any) -> str:
    """格式化数值，缺失时返回占位文本。"""
    if value is None:
        return "未提供"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _status_cn(status: Any) -> str:
    mapping = {
        "matched": "与目标基本一致",
        "higher": "高于目标",
        "lower": "低于目标",
    }
    return mapping.get(str(status), "缺少偏差评价")


def _brief_metric_sentence(entry: Mapping[str, Any], focus: str) -> str:
    """生成简洁版单项描述。"""
    if entry["actual"] is None and entry["target"] is None:
        return f"{entry['label']}信息缺失，暂无法完成{focus}评价。"
    if entry["actual"] is None:
        return f"{entry['label']}仅提供了目标值 {_format_value(entry['target'])}，尚无实际结果用于{focus}评价。"
    if entry["target"] is None:
        return f"{entry['label']}实际值为 {_format_value(entry['actual'])}，但缺少目标值，暂无法判断{focus}偏差。"

    sentence = (
        f"{entry['label']}目标为 {_format_value(entry['target'])}，实际为 {_format_value(entry['actual'])}，"
        f"总体上{_status_cn(entry['status'])}"
    )
    if entry["delta"] is not None:
        sentence += f"，偏差为 {abs(float(entry['delta'])):.4f}"
    sentence += "。"
    return sentence


def _detailed_metric_paragraph(entry: Mapping[str, Any], scientific_note: str) -> str:
    """生成详细版单项分析。"""
    if entry["actual"] is None and entry["target"] is None:
        return f"- {entry['label']}：目标值与实际值均缺失，当前无法进行结构解释。"
    if entry["actual"] is None:
        return f"- {entry['label']}：目标值为 {_format_value(entry['target'])}，但尚未获得实际测量结果，因此该指标暂不参与本次结构判断。"
    if entry["target"] is None:
        return f"- {entry['label']}：实际结果为 {_format_value(entry['actual'])}，但缺少目标值，当前只能记录结果而不能进行目标匹配评价。"

    line = (
        f"- {entry['label']}：目标值 {_format_value(entry['target'])}，实际值 {_format_value(entry['actual'])}，"
        f"判断为“{_status_cn(entry['status'])}”。"
    )
    if entry["delta"] is not None:
        line += f" 绝对偏差 {abs(float(entry['delta'])):.4f}。"
    if entry["relative_error"] is not None:
        line += f" 相对误差 {abs(float(entry['relative_error'])) * 100:.2f}%。"
    line += f" {scientific_note}"
    return line


def _overall_evaluation(entries: list[dict[str, Any]]) -> str:
    """根据可用指标生成总体评价。"""
    available = [entry for entry in entries if entry["status"] in {"matched", "higher", "lower"}]
    if not available:
        return "总体上，由于有效指标不足，当前仅能完成结果记录，尚不宜给出完整结构评价。"

    matched = sum(entry["status"] == "matched" for entry in available)
    higher = sum(entry["status"] == "higher" for entry in available)
    lower = sum(entry["status"] == "lower" for entry in available)

    if matched >= max(1, len(available) - 1):
        return "总体上，本次重构结果与目标结构参数较为接近，可作为后续模型调优与样本筛选的候选结果。"
    if higher > lower:
        return "总体上，重构结果在多个指标上偏高，说明生成结构可能较为疏松或界面复杂度偏大，建议加强目标约束或调整后处理参数。"
    if lower > higher:
        return "总体上，重构结果在多个指标上偏低，说明生成结构可能偏致密或连通性不足，建议提高模型对目标条件的响应能力。"
    return "总体上，各指标偏差方向不完全一致，说明结构特征已部分接近目标，但仍需结合训练样本和后处理策略进一步修正。"


def generate_brief_analysis(
    target: ReconstructionConfig | Mapping[str, Any] | Any,
    actual: ComputedMetrics | Mapping[str, Any] | Any,
    comparison: Mapping[str, Any] | Any | None = None,
) -> str:
    """生成适合 GUI 显示的简洁中文分析文本。"""
    target_mapping = _to_mapping(target)
    actual_mapping = _to_mapping(actual)
    comparison_mapping = _comparison_mapping(comparison, actual, target)

    entries = [
        _metric_entry("porosity", "孔隙率", target_mapping, actual_mapping, comparison_mapping),
        _metric_entry("pore_size_mean", "平均孔径", target_mapping, actual_mapping, comparison_mapping),
        _metric_entry("specific_surface_area", "比表面积", target_mapping, actual_mapping, comparison_mapping),
        _metric_entry("coordination_number", "配位数", target_mapping, actual_mapping, comparison_mapping),
    ]
    pore_std_entry = _metric_entry("pore_size_std", "孔径标准差", target_mapping, actual_mapping, comparison_mapping)

    lines = [
        "本次重构面向二维多孔介质目标结构参数匹配任务，重点关注孔隙率、孔径分布、比表面积与连通性特征。",
        _brief_metric_sentence(entries[0], "孔隙率"),
        f"孔径分布方面，{_brief_metric_sentence(entries[1], '平均孔径')} { _brief_metric_sentence(pore_std_entry, '孔径离散性') }",
        _brief_metric_sentence(entries[2], "比表面积"),
        _brief_metric_sentence(entries[3], "连通性"),
        _overall_evaluation(entries + [pore_std_entry]),
    ]
    return "\n".join(lines)


def generate_detailed_analysis(
    target: ReconstructionConfig | Mapping[str, Any] | Any,
    actual: ComputedMetrics | Mapping[str, Any] | Any,
    comparison: Mapping[str, Any] | Any | None = None,
) -> str:
    """生成适合 TXT 导出的详细中文分析文本。"""
    target_mapping = _to_mapping(target)
    actual_mapping = _to_mapping(actual)
    comparison_mapping = _comparison_mapping(comparison, actual, target)

    porosity = _metric_entry("porosity", "孔隙率", target_mapping, actual_mapping, comparison_mapping)
    pore_mean = _metric_entry("pore_size_mean", "平均孔径", target_mapping, actual_mapping, comparison_mapping)
    pore_std = _metric_entry("pore_size_std", "孔径标准差", target_mapping, actual_mapping, comparison_mapping)
    surface = _metric_entry("specific_surface_area", "比表面积", target_mapping, actual_mapping, comparison_mapping)
    coordination = _metric_entry("coordination_number", "配位数", target_mapping, actual_mapping, comparison_mapping)

    lines = [
        "重构分析报告：",
        (
            f"本次任务的目标参数概述为：孔隙率 {_format_value(target_mapping.get('porosity'))}，"
            f"平均孔径 {_format_value(target_mapping.get('pore_size_mean'))}，"
            f"孔径标准差 {_format_value(target_mapping.get('pore_size_std'))}，"
            f"比表面积 {_format_value(target_mapping.get('specific_surface_area'))}，"
            f"配位数 {_format_value(target_mapping.get('coordination_number'))}。"
        ),
        _detailed_metric_paragraph(porosity, "该指标反映孔隙相体积分数，是评价宏观疏松程度的核心参数。"),
        _detailed_metric_paragraph(pore_mean, "该指标体现主导孔尺度是否接近目标分布中心。"),
        _detailed_metric_paragraph(pore_std, "该指标用于表征孔径离散程度和分布宽度。"),
        _detailed_metric_paragraph(surface, "在二维条件下，该值可近似表征孔隙-固体界面复杂度。"),
        _detailed_metric_paragraph(coordination, "该指标与骨架连通性和传输通道连续性密切相关。"),
        "总体评价：" + _overall_evaluation([porosity, pore_mean, pore_std, surface, coordination]),
    ]
    return "\n".join(lines)


def generate_analysis(
    target: ReconstructionConfig | Mapping[str, Any] | Any,
    actual: ComputedMetrics | Mapping[str, Any] | Any,
    comparison: Mapping[str, Any] | Any | None = None,
) -> str:
    """兼容旧调用，默认返回简洁版分析文本。"""
    return generate_brief_analysis(target=target, actual=actual, comparison=comparison)

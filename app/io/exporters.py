from __future__ import annotations

"""结果导出模块。"""

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import EXPORT_DIR
from app.core.dto import ReconstructionResult

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExportPaths:
    """记录本次导出的文件路径。"""

    png_path: Path
    csv_path: Path
    txt_path: Path
    json_path: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_export_path(
    path: Path | None,
    suffix: str,
    prefix: str = "reconstruction_result",
    add_timestamp: bool = True,
) -> Path:
    """生成带时间戳的导出路径。"""
    if path is None:
        directory = EXPORT_DIR
        name = prefix
    elif path.suffix:
        directory = path.parent
        name = path.stem
    else:
        directory = path
        name = prefix

    directory.mkdir(parents=True, exist_ok=True)
    stamped_name = f"{name}_{_timestamp()}" if add_timestamp else name
    return directory / f"{stamped_name}{suffix}"


def _comparison_rows(result: ReconstructionResult) -> list[list[Any]]:
    target = result.config.to_dict()
    actual = result.metrics.to_dict()
    rows = [["metric", "target", "actual", "delta"]]
    for key in [
        "porosity",
        "pore_size_mean",
        "pore_size_std",
        "specific_surface_area",
        "coordination_number",
    ]:
        target_value = float(target.get(key, 0.0))
        actual_value = float(actual.get(key, 0.0))
        rows.append([key, target_value, actual_value, actual_value - target_value])
    return rows


def export_png(path: Path | None, image: np.ndarray, add_timestamp: bool = True) -> Path:
    """导出重构图像 PNG。"""
    output_path = _resolve_export_path(path, ".png", add_timestamp=add_timestamp)
    array = (np.asarray(image) > 0).astype(np.uint8) * 255
    Image.fromarray(array, mode="L").save(output_path)
    LOGGER.info("导出 PNG 成功：%s", output_path)
    return output_path


def export_csv(path: Path | None, result: ReconstructionResult, add_timestamp: bool = True) -> Path:
    """导出参数对比表 CSV。"""
    output_path = _resolve_export_path(path, ".csv", add_timestamp=add_timestamp)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_comparison_rows(result))
    LOGGER.info("导出 CSV 成功：%s", output_path)
    return output_path


def export_txt(path: Path | None, result: ReconstructionResult, add_timestamp: bool = True) -> Path:
    """导出分析说明 TXT。"""
    output_path = _resolve_export_path(path, ".txt", add_timestamp=add_timestamp)
    content = (
        f"重构任务时间：{datetime.now().isoformat(timespec='seconds')}\n"
        f"目标参数：{json.dumps(result.config.to_dict(), ensure_ascii=False)}\n"
        f"模型信息：{json.dumps(result.model_info, ensure_ascii=False)}\n"
        "分析说明（简洁版）：\n"
        f"{result.analysis_text}\n\n"
        "分析说明（详细版）：\n"
        f"{result.detailed_analysis_text or result.analysis_text}\n"
    )
    output_path.write_text(content, encoding="utf-8")
    LOGGER.info("导出 TXT 成功：%s", output_path)
    return output_path


def export_json(path: Path | None, result: ReconstructionResult, add_timestamp: bool = True) -> Path:
    """导出完整结果 JSON。"""
    output_path = _resolve_export_path(path, ".json", add_timestamp=add_timestamp)
    payload = {
        "config": result.config.to_dict(),
        "metrics": result.metrics.to_dict(),
        "analysis_text": result.analysis_text,
        "detailed_analysis_text": result.detailed_analysis_text,
        "comparison": result.comparison,
        "model_info": result.model_info,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("导出 JSON 成功：%s", output_path)
    return output_path


def export_all(export_dir: Path | None, result: ReconstructionResult) -> ExportPaths:
    """一次性导出 PNG / CSV / TXT / JSON。"""
    directory = export_dir or EXPORT_DIR
    png_path = export_png(directory, result.binary_image)
    csv_path = export_csv(directory, result)
    txt_path = export_txt(directory, result)
    json_path = export_json(directory, result)
    summary = ExportPaths(
        png_path=png_path,
        csv_path=csv_path,
        txt_path=txt_path,
        json_path=json_path,
    )
    LOGGER.info("导出完成：%s", summary.to_dict())
    return summary

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from app.io.exporters import export_csv, export_json, export_png, export_txt


def test_export_png_creates_timestamped_file(tmp_path: Path, sample_result) -> None:
    output_path = export_png(tmp_path / "result.png", sample_result.binary_image)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert "result_" in output_path.name


def test_export_csv_creates_comparison_table(tmp_path: Path, sample_result) -> None:
    output_path = export_csv(tmp_path / "result.csv", sample_result)
    content = output_path.read_text(encoding="utf-8-sig")
    assert "metric,target,actual,delta" in content
    assert "porosity" in content


def test_export_txt_contains_analysis_text(tmp_path: Path, sample_result) -> None:
    output_path = export_txt(tmp_path / "result.txt", sample_result)
    content = output_path.read_text(encoding="utf-8")
    assert "分析说明（简洁版）" in content
    assert "详细分析" in content


def test_export_json_contains_complete_payload(tmp_path: Path, sample_result) -> None:
    output_path = export_json(tmp_path / "result.json", sample_result)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["analysis_text"] == "简洁分析"
    assert payload["detailed_analysis_text"] == "详细分析"
    assert "comparison" in payload

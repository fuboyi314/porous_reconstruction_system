from __future__ import annotations

import os

import pytest

pytest.importorskip("numpy")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def test_main_window_smoke_startup() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.windowTitle()
        assert window.run_button.text() == "开始重构"
        assert window.export_button.text() == "导出结果"
    finally:
        window.close()
        app.quit()

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

import logging

from app.gui.main_window import MainWindow
from app.io.logging_setup import setup_logging


def main() -> int:
    logger = setup_logging()
    logger.info("软件启动：进入主界面初始化阶段。")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

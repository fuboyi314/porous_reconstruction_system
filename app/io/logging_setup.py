from __future__ import annotations

"""日志初始化模块。"""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import APP_NAME, LOG_DIR


def setup_logging(log_dir: Path | None = None) -> logging.Logger:
    """初始化控制台与文件日志，并记录软件启动信息。"""
    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    log_file = target_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    app_logger = logging.getLogger(APP_NAME)
    app_logger.info("软件启动完成。日志文件：%s", log_file)
    return app_logger

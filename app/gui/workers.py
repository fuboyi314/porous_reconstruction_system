from __future__ import annotations

"""GUI 线程工作器。"""

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.core.config import ReconstructionConfig
from app.core.service import ReconstructionService


class ReconstructionWorker(QObject):
    """在后台线程中执行重构任务，避免阻塞界面。"""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: ReconstructionConfig, model_path: Path | None = None) -> None:
        super().__init__()
        self.config = config
        self.model_path = model_path

    @Slot()
    def run(self) -> None:
        """执行重构并发出成功或失败信号。"""
        try:
            service = ReconstructionService(model_path=self.model_path)
            result = service.run(self.config)
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - 线程中的兜底保护
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)

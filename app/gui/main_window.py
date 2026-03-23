from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, QThread
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.config import APP_NAME, DEFAULT_MODEL_PATH, EXPORT_DIR
from app.core.config import ConfigValidationError, ReconstructionConfig
from app.core.dto import ReconstructionResult
from app.gui.workers import ReconstructionWorker
from app.io.exporters import export_csv, export_json, export_png, export_txt

LOGGER = logging.getLogger(__name__)


class LogEmitter(QObject):
    """日志转 Qt 信号。"""

    message_emitted = Signal(str)


class QtLogHandler(logging.Handler):
    """将 logging 输出转发到 GUI 日志窗口。"""

    def __init__(self, emitter: LogEmitter) -> None:
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self.emitter.message_emitted.emit(message)


class MainWindow(QMainWindow):
    """主图形界面窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 860)

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.current_result: ReconstructionResult | None = None
        self.current_model_path: Path | None = DEFAULT_MODEL_PATH
        self.worker_thread: QThread | None = None
        self.worker: ReconstructionWorker | None = None

        self.log_emitter = LogEmitter()
        self.log_handler = QtLogHandler(self.log_emitter)
        self._install_log_handler()

        self._build_ui()

    def _build_ui(self) -> None:
        self._build_toolbar()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        root_layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([340, 540, 520])
        root_layout.addWidget(splitter, stretch=1)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主操作")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        load_action = QAction("加载模型", self)
        load_action.triggered.connect(self.load_model)
        toolbar.addAction(load_action)

        run_action = QAction("开始重构", self)
        run_action.triggered.connect(self.start_reconstruction)
        toolbar.addAction(run_action)

        export_action = QAction("导出结果", self)
        export_action.triggered.connect(self.export_result)
        toolbar.addAction(export_action)

        clear_log_action = QAction("清空日志", self)
        clear_log_action.triggered.connect(self.clear_logs)
        toolbar.addAction(clear_log_action)

    def _build_input_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("参数输入区")
        form = QFormLayout(group)

        self.porosity_input = self._double_spin(0.35, 0.0001, 0.9999, 4)
        self.pore_mean_input = self._double_spin(12.0, 0.1, 500.0, 2)
        self.pore_std_input = self._double_spin(3.0, 0.01, 500.0, 2)
        self.ssa_input = self._double_spin(0.18, 0.0001, 100.0, 4)
        self.coord_input = self._double_spin(2.8, 0.0001, 20.0, 2)
        self.width_input = self._spin(256, 1, 2048)
        self.height_input = self._spin(256, 1, 2048)
        self.seed_input = self._spin(42, 0, 999999)
        self.samples_input = self._spin(1, 1, 128)

        form.addRow("孔隙率", self.porosity_input)
        form.addRow("孔径均值", self.pore_mean_input)
        form.addRow("孔径标准差", self.pore_std_input)
        form.addRow("比表面积", self.ssa_input)
        form.addRow("配位数", self.coord_input)
        form.addRow("图像宽度", self.width_input)
        form.addRow("图像高度", self.height_input)
        form.addRow("随机种子", self.seed_input)
        form.addRow("样本数量", self.samples_input)

        self.model_path_label = QLabel(f"当前模型：{self.current_model_path}")
        self.model_path_label.setWordWrap(True)

        button_row = QHBoxLayout()
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.clicked.connect(self.load_model)
        self.run_button = QPushButton("开始重构")
        self.run_button.clicked.connect(self.start_reconstruction)
        button_row.addWidget(self.load_model_button)
        button_row.addWidget(self.run_button)

        layout.addWidget(group)
        layout.addWidget(self.model_path_label)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return widget

    def _build_image_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        title = QLabel("图像显示区")
        title.setStyleSheet("font-weight: 600;")

        self.image_label = QLabel("等待重构结果")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.image_label.setStyleSheet(
            "background-color: #f5f6f8; border: 1px solid #d9dce1; border-radius: 6px;"
        )

        layout.addWidget(title)
        layout.addWidget(self.image_label, stretch=1)
        return widget

    def _build_result_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("结果显示区")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self.comparison_table = QTableWidget(5, 4)
        self.comparison_table.setHorizontalHeaderLabels(["指标", "目标值", "实际值", "偏差"])
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(QLabel("参数对比表"))
        layout.addWidget(self.comparison_table)

        self.analysis_text = QPlainTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlaceholderText("分析文字将在此显示。")
        layout.addWidget(QLabel("分析文字"))
        layout.addWidget(self.analysis_text, stretch=1)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("运行日志将在此显示。")
        layout.addWidget(QLabel("日志窗口"))
        layout.addWidget(self.log_text, stretch=1)

        result_button_row = QHBoxLayout()
        self.export_button = QPushButton("导出结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        self.clear_log_button = QPushButton("清空日志")
        self.clear_log_button.clicked.connect(self.clear_logs)
        result_button_row.addWidget(self.export_button)
        result_button_row.addWidget(self.clear_log_button)
        layout.addLayout(result_button_row)
        return widget

    def _install_log_handler(self) -> None:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        self.log_handler.setFormatter(formatter)
        self.log_emitter.message_emitted.connect(self.append_log)
        logging.getLogger().addHandler(self.log_handler)

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI 生命周期钩子
        logging.getLogger().removeHandler(self.log_handler)
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        super().closeEvent(event)

    def append_log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def clear_logs(self) -> None:
        self.log_text.clear()

    def load_model(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型权重文件",
            str(self.current_model_path or DEFAULT_MODEL_PATH),
            "PyTorch Weights (*.pt *.pth);;All Files (*)",
        )
        if not path_str:
            return
        self.current_model_path = Path(path_str)
        self.model_path_label.setText(f"当前模型：{self.current_model_path}")
        LOGGER.info("已选择模型文件：%s", self.current_model_path)

    def _gather_config(self) -> ReconstructionConfig:
        return ReconstructionConfig(
            porosity=self.porosity_input.value(),
            pore_size_mean=self.pore_mean_input.value(),
            pore_size_std=self.pore_std_input.value(),
            specific_surface_area=self.ssa_input.value(),
            coordination_number=self.coord_input.value(),
            image_width=self.width_input.value(),
            image_height=self.height_input.value(),
            seed=self.seed_input.value(),
            sample_count=self.samples_input.value(),
        )

    def start_reconstruction(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.information(self, "提示", "后台任务正在运行，请稍候。")
            return

        try:
            config = self._gather_config()
        except ConfigValidationError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "错误", f"读取参数失败：{exc}")
            return

        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.worker_thread = QThread(self)
        self.worker = ReconstructionWorker(config=config, model_path=self.current_model_path)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_reconstruction_finished)
        self.worker.failed.connect(self._on_reconstruction_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._on_thread_finished)
        self.worker_thread.start()
        LOGGER.info("参数提交成功：%s", config.to_dict())
        LOGGER.info("已启动后台重构线程。")

    def _on_reconstruction_finished(self, result: ReconstructionResult) -> None:
        self.current_result = result
        self._show_image(result.binary_image)
        self._fill_comparison_table(result)
        self.analysis_text.setPlainText(result.analysis_text)
        self.export_button.setEnabled(True)
        LOGGER.info("重构任务完成，界面已刷新。")

    def _on_reconstruction_failed(self, detail: str) -> None:
        LOGGER.error("重构任务失败：%s", detail)
        QMessageBox.critical(self, "重构失败", detail)

    def _on_thread_finished(self) -> None:
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 1)
        self.run_button.setEnabled(True)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def _show_image(self, image: np.ndarray) -> None:
        array = (np.asarray(image) > 0).astype(np.uint8) * 255
        height, width = array.shape
        qimage = QImage(array.data, width, height, array.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage.copy())
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI 尺寸事件
        super().resizeEvent(event)
        if self.current_result is not None:
            self._show_image(self.current_result.binary_image)

    def _fill_comparison_table(self, result: ReconstructionResult) -> None:
        target = result.config.to_dict()
        actual = result.metrics.to_dict()
        rows = [
            ("孔隙率", "porosity"),
            ("平均孔径", "pore_size_mean"),
            ("孔径标准差", "pore_size_std"),
            ("比表面积", "specific_surface_area"),
            ("配位数", "coordination_number"),
        ]
        self.comparison_table.setRowCount(len(rows))
        for row, (label, key) in enumerate(rows):
            target_value = float(target.get(key, 0.0))
            actual_value = float(actual.get(key, 0.0))
            delta = actual_value - target_value
            values = [label, f"{target_value:.4f}", f"{actual_value:.4f}", f"{delta:+.4f}"]
            for col, value in enumerate(values):
                self.comparison_table.setItem(row, col, QTableWidgetItem(value))

    def export_result(self) -> None:
        if not self.current_result:
            QMessageBox.information(self, "提示", "请先运行重构。")
            return

        path_str, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            str(EXPORT_DIR / "reconstruction_result.json"),
            "JSON (*.json);;TXT (*.txt);;CSV (*.csv);;PNG (*.png)",
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            suffix = path.suffix.lower()
            if suffix == ".json" or "JSON" in selected_filter:
                if suffix != ".json":
                    path = path.with_suffix(".json")
                path = export_json(path, self.current_result)
            elif suffix == ".txt" or "TXT" in selected_filter:
                if suffix != ".txt":
                    path = path.with_suffix(".txt")
                path = export_txt(path, self.current_result)
            elif suffix == ".csv" or "CSV" in selected_filter:
                if suffix != ".csv":
                    path = path.with_suffix(".csv")
                path = export_csv(path, self.current_result)
            else:
                if suffix != ".png":
                    path = path.with_suffix(".png")
                path = export_png(path, self.current_result.binary_image)
        except Exception as exc:  # pragma: no cover - 文件系统/第三方库异常
            QMessageBox.critical(self, "导出失败", f"导出结果失败：{exc}")
            return

        LOGGER.info("结果已导出至：%s", path)
        QMessageBox.information(self, "导出完成", f"文件已保存到：{path}")

    @staticmethod
    def _double_spin(value: float, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(0.01)
        widget.setAlignment(Qt.AlignRight)
        return widget

    @staticmethod
    def _spin(value: int, minimum: int, maximum: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setAlignment(Qt.AlignRight)
        return widget

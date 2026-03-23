# 基于神经网络的二维多孔介质重构系统

本项目是一个基于 **PySide6 + PyTorch** 的桌面图形界面程序，用于根据目标参数快速重构二维多孔介质二值图像，并自动完成结构指标分析、结果对比、日志记录与多格式导出。

## 功能概览

- 输入目标参数：孔隙率、孔径分布、比表面积、配位数、图像尺寸、随机种子、样本数量。
- 使用预训练风格的 PyTorch 生成器重构二维多孔介质二值图像。
- 自动计算实际孔隙率、孔径分布、比表面积、配位数。
- 自动生成目标值与实际值对比分析文字。
- 记录运行日志到 `outputs/logs/app.log`。
- 支持导出 PNG、CSV、TXT、JSON。
- 模块化结构，便于继续扩展和申请软件著作权。

## 环境要求

- Python 3.10+
- 推荐使用虚拟环境

## 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

## 目录结构

```text
porous_reconstruction_system/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── config.py
│   │   ├── dto.py
│   │   ├── generator.py
│   │   ├── metrics.py
│   │   ├── model_manager.py
│   │   ├── pipeline.py
│   │   ├── postprocess.py
│   │   └── service.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   └── workers.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── exporters.py
│   │   ├── logging_setup.py
│   │   └── model_store.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── plotting.py
│   ├── __init__.py
│   └── config.py
├── models/
│   └── pretrained/
│       └── README.md
├── outputs/
│   └── .gitkeep
├── main.py
├── README.md
├── requirements.txt
└── tests/
    ├── test_config.py
    ├── test_metrics.py
    ├── test_exporters.py
    ├── test_gui_smoke.py
    └── test_analyzer.py
```

## 说明

当前版本内置了一个 **可本地运行的最小 CVAE 风格神经网络重构流程**：

- 使用 PyTorch 条件变分自编码器风格重构器接受噪声与参数条件；
- 如存在预训练权重文件，可自动加载；
- 如不存在权重，则使用稳定的启发式初始化和随机种子生成结果，确保项目在本地可直接运行；
- 后续可将 `models/pretrained/generator.pt` 替换为真实训练好的模型参数。

## 可扩展方向

- 接入真实训练数据集与训练脚本；
- 优化孔径分布、配位数等指标定义；
- 增加批量重构、结果筛选、多窗口对比等功能；
- 增加软件版本信息与版权登记所需文档模板。


## 运行与测试说明

### 本地运行

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 测试命令

```bash
pytest tests/test_config.py tests/test_analyzer.py
pytest tests/test_metrics.py tests/test_exporters.py tests/test_gui_smoke.py
```

> 说明：`test_metrics.py`、`test_exporters.py`、`test_gui_smoke.py` 依赖科学计算与 Qt 组件，若环境未安装相关依赖会自动跳过。

## 打包说明

### PyInstaller 打包命令

```bash
pyinstaller packaging/porous_reconstruction_system.spec --noconfirm
```

### Windows 兼容性说明

- 建议在 **Windows 10/11 + Python 3.10/3.11 64 位** 环境打包。
- 打包前请在 Windows 本机安装与目标环境一致的依赖：`PySide6`、`torch`、`numpy`、`scipy`、`scikit-image`、`matplotlib`、`Pillow`。
- 若启动后缺少 Qt 平台插件，请确认 PyInstaller 打包结果中包含 `PySide6` 插件目录。
- GUI 烟雾测试使用 `QT_QPA_PLATFORM=offscreen`，在 Windows 正式运行时无需设置该变量。

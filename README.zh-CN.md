# AIRBOT 键盘视觉抓取 Demo

[English](README.md) | [中文](README.zh-CN.md)

这是一个完全运行于 DISCOVERSE 仿真环境中的单臂键盘视觉抓取 Demo。它检测蓝色和绿色积木，确认目标，使用 MobileSAM 分割物体，估计抓取位姿，并执行仿真抓放流程。

本 Demo 不使用实体机器人或实体相机。

## 功能组成

| 组件 | 功能 |
| --- | --- |
| DISCOVERSE 与 MuJoCo | AIRBOT Play 场景、物理仿真与运动执行 |
| YOLO | 积木检测及按颜色确认目标 |
| MobileSAM | 目标分割 |
| SimpleGrasp | 根据仿真腕部 RGB-D 画面估计抓取位姿 |
| 键盘面板 | 使用受白名单限制的中文文本指令控制仿真 |

## 环境要求

- Ubuntu 22.04
- 推荐 Python 3.10；安装脚本支持 Python 3.10、3.11 和 3.12
- 能运行交互式 MuJoCo 窗口的 NVIDIA 图形环境
- 安装期间可联网下载固定版本的 DISCOVERSE 源码和 Python 依赖

键盘仿真本身不需要 AIRBOT SDK 或 `airbot-arm` 服务。5.2.2 仅是与独立真机工作流相关的兼容上下文，不属于本 Demo。

## 快速开始

```bash
git clone https://github.com/DISCOVER-Robotics/AIRBOT-Keyboard-Visual-Grasp-Demo.git
cd AIRBOT-Keyboard-Visual-Grasp-Demo
./install_keyboard.sh
./run_keyboard.sh
```

安装脚本会创建 `venv-keyboard`、检查 Python 版本，并拉取固定版本的 DISCOVERSE。可通过 `GRASP_KEYBOARD_PYTHON` 指定已有 Python 解释器，或在安装前设置 `PYTHON_BIN`。

窗口上方为第三视角，下方为末端相机视角。选择目标或输入以下指令后按回车：

| 指令 | 动作 |
| --- | --- |
| `抓取蓝色积木` | 检测、校验并将蓝色积木移动到放置区 |
| `抓取绿色积木` | 检测、校验并将绿色积木移动到放置区 |
| `打开夹爪` | 打开仿真夹爪 |
| `闭合夹爪` | 闭合仿真夹爪 |
| `回到观察位` | 移动至配置的观察位 |
| `拍照` | 清除当前选择并采集新画面 |
| `预测位姿` | 只估计抓取位姿，不移动机械臂 |

## 仓库结构

```text
app/                           仿真、视觉和命令源码
configs/                       检测与仿真参数
checkpoint/                    已版本化的 YOLO、MobileSAM 和 FastSAM 权重
install_keyboard.sh            键盘仿真环境安装脚本
run_keyboard.sh                键盘仿真启动脚本
requirements-sim-keyboard.txt  键盘 Demo 依赖
tests/                         离线测试
```

## 安全与范围

`run_keyboard.sh` 只启动键盘界面，仓库中不包含实体机器人控制入口。本仓库仅用于仿真演示与开发，不能用于控制实体机器人。

## 许可证与上游组件

DISCOVERSE 将按 `install_keyboard.sh` 中指定的固定版本下载。分发或商业使用前，请审查 DISCOVERSE、YOLO/Ultralytics、MobileSAM、MuJoCo 及所有模型的许可证。

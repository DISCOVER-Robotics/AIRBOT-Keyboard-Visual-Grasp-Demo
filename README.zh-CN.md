# AIRBOT 视觉抓取与键盘控制 Demo

[English](README.md) | [中文](README.zh-CN.md)

单臂视觉抓取与键盘控制 Demo，支持独立 DISCOVERSE 仿真、AIRBOT Play 真机抓放，以及真机反馈的只读仿真镜像。视觉流程包括 YOLO 检测、蓝绿目标确认、MobileSAM 分割、抓取位姿计算和抓放执行。

## 运行模式

| 模式 | 入口 | 硬件连接 |
| --- | --- | --- |
| 独立键盘仿真 | `./run_keyboard.sh` | 不连接真机 |
| 实体机械臂 | `./run_keyboard_robot.sh` | 真实相机及 AIRBOT Play |
| 真机及反馈镜像 | `./run_keyboard_robot.sh --with-sim` | 真机控制，仿真显示反馈 |

## 键盘仿真快速开始

推荐 Ubuntu 22.04 + Python 3.10。键盘安装脚本支持 Python 3.10、3.11 和 3.12。

```bash
git clone https://github.com/DISCOVER-Robotics/AIRBOT-Keyboard-Visual-Grasp-Demo.git
cd AIRBOT-Keyboard-Visual-Grasp-Demo
./install_keyboard.sh
./run_keyboard.sh
```

安装脚本创建独立的 `venv-keyboard` 并下载固定版本的 DISCOVERSE。可通过 `GRASP_KEYBOARD_PYTHON` 指定已有解释器，安装和运行时使用同一路径。此模式不连接真机。

窗口提供第三视角和腕部相机视角。选择目标或在右侧输入指令，按回车或点击执行；抓放后可重置场景。

## 实体机械臂部署

真机流程使用 **arm-sdk 5.2.2 和 airbot-arm 5.2.2**。将适配 Ubuntu 22.04、Python 和平台的厂商包放入 `packages/`：

```text
arm_sdk-5.2.2-py3-none-any.whl
airbot-arm_5.2.2_amd64.deb
```

仓库不包含厂商二进制包。其他安装路径设置见 [packages/README.md](packages/README.md)。

```bash
./install.sh
./run_keyboard_robot.sh --check
```

`--check` 仅检查依赖，不连接设备。启动脚本不会自动启动机械臂服务。

### 终端 A：启动机械臂服务

请按照[官网机械臂服务启动教程](https://docs.discover-robotics.com/document/airbot-play/sdk/quickstart/run-service.html)启动 `airbot-arm`。具体启动命令、CAN 接口选择、硬件类型和服务参数请参阅官网。

服务设置需与 `configs/sam_simplegrasp.yaml` 中的 `ArmParams` 一致。保持服务运行，再在终端 B 启动真机 Demo。

### 终端 B：启动真机 Demo

保持终端 A 运行，在仓库根目录执行下列操作：

**真机 GUI 启动后会自动驱动机械臂到配置的观察位。** 必须先核对标定、观察位和放置位、空间与夹爪设置，并确认急停可用。仓库参数是特定工站的配置，不是适用于任意现场的安全默认值。

```bash
./run_keyboard_robot.sh
```

在 GUI 的“键盘抓取”页输入指令。`./run_grasp.sh` 启动同一真机键盘界面。手动拍照选点和恢复控制连接等功能仍保留。

## 控制指令

| 指令 | 动作 |
| --- | --- |
| `抓取蓝色积木` / `抓取绿色积木` | 确认并抓取指定颜色积木 |
| `打开夹爪` | 打开夹爪 |
| `闭合夹爪` / `关闭夹爪` | 关闭夹爪 |
| `回到观察位` | 移动至观察位 |
| `拍照` | 采集新画面 |
| `预测位姿` | 估计位姿，不执行抓取 |
| `开始抓取` | 抓取已准备的目标 |

指令受目标确认与忙碌状态检查约束。回车和确认按钮使用同一执行入口。

## 真机反馈镜像

在真机环境中安装可选仿真依赖：

```bash
./install_sim.sh
./run_keyboard_robot.sh --with-sim
```

镜像为只读反馈窗口，机械臂仍由真机 GUI 控制。`./run_sim.sh` 在该共享环境中启动独立键盘仿真。

真机反馈镜像模式同样需要保持终端 A 的服务运行。先关闭真机 GUI 和镜像，再停止服务；停止服务的行为及相关参数请参阅官网教程。

通过 `GRASP_KEYBOARD_PYTHON=/path/to/venv/bin/python` 可复用已有真机环境；向该环境安装仿真依赖时设置 `GRASP_SIM_PYTHON`。

## 配置与安全

- `configs/config_file.yaml` 指向真机配置 `configs/sam_simplegrasp.yaml`。
- `configs/sam_simplegrasp.yaml` 包含模型、标定、连接、观察与放置位、运动保护参数；使用前按现场重新标定。
- `configs/live_mirror.yaml` 控制反馈镜像。
- 可使用 RealSense 或完成 USB RGB 模式配置与标定。RGB 模式依赖固定相机与已测量桌面平面。
- 键盘指令不能代替硬件急停，人员及障碍物必须远离机械臂工作空间。

## 仓库结构

```text
app/              视觉、标定、真机控制、键盘界面与仿真
configs/          真机与仿真配置
checkpoint/       YOLO 和分割模型权重
packages/         厂商包安装说明，二进制需单独提供
tests/            离线回归测试
```

## 上游组件

DISCOVERSE 按安装脚本中的固定版本下载。分发或商用前，请审查 DISCOVERSE、YOLO/Ultralytics、MobileSAM、MuJoCo、厂商包及各模型的许可证。

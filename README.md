# AIRBOT Keyboard Visual Grasp Demo

[English](README.md) | [中文](README.zh-CN.md)

This is a keyboard-controlled, single-arm visual grasping demo running entirely in DISCOVERSE simulation. It detects blue and green blocks, confirms the target, segments it with MobileSAM, estimates a grasp pose, and performs a simulated pick-and-place sequence.

No physical robot, camera, microphone, or speech-recognition service is used by this demo.

## Demo capabilities

| Component | Function |
| --- | --- |
| DISCOVERSE and MuJoCo | AIRBOT Play scene, physics, and motion execution |
| YOLO | Block detection and color-aware target confirmation |
| MobileSAM | Target segmentation |
| SimpleGrasp | Metric grasp-pose estimation from the simulated wrist-camera RGB-D frame |
| Keyboard panel | Whitelisted Chinese text commands for the simulation |

## Requirements

- Ubuntu 22.04
- Python 3.10 recommended; Python 3.10, 3.11, and 3.12 are supported by the installer
- An NVIDIA-capable graphics environment for the interactive MuJoCo window
- Network access during installation to fetch the pinned DISCOVERSE revision and Python dependencies

The keyboard simulation itself has no AIRBOT SDK or `airbot-arm` service requirement. Its 5.2.2 compatibility context applies only to the separate physical-robot workflow, which is not part of this demo.

## Quick start

```bash
git clone https://github.com/DISCOVER-Robotics/AIRBOT-Keyboard-Visual-Grasp-Demo.git
cd AIRBOT-Keyboard-Visual-Grasp-Demo
./install_keyboard.sh
./run_keyboard.sh
```

The installer creates `venv-keyboard`, checks the selected Python version, and fetches the pinned DISCOVERSE revision. Set `GRASP_KEYBOARD_PYTHON` to use an existing interpreter, or set `PYTHON_BIN` before installation.

The window shows a third-person scene above an end-effector camera view. Select a target or enter one of the following commands, then press Enter:

| Command | Action |
| --- | --- |
| `抓取蓝色积木` | Detect, validate, and move the blue block to the placement area |
| `抓取绿色积木` | Detect, validate, and move the green block to the placement area |
| `打开夹爪` | Open the simulated gripper |
| `闭合夹爪` | Close the simulated gripper |
| `回到观察位` | Move to the configured observation pose |
| `拍照` | Clear the selection and capture a fresh view |
| `预测位姿` | Estimate a grasp pose without moving the arm |

## Repository layout

```text
app/                           Simulation, vision, and command sources
configs/                       Detection and simulation parameters
checkpoint/                    Versioned YOLO, MobileSAM, and FastSAM weights
install_keyboard.sh            Keyboard-only environment installer
run_keyboard.sh                Keyboard-only simulation launcher
requirements-sim-keyboard.txt  Pinned keyboard-demo dependencies
tests/                         Offline tests
```

## Safety and scope

`run_keyboard.sh` launches only the keyboard interface and does not include a physical-robot control entrypoint. This repository is intended for simulation demonstration and development only; it must not be used to command a physical robot.

## License and upstream components

DISCOVERSE is fetched from the pinned upstream revision specified in `install_keyboard.sh`. Review the licenses of DISCOVERSE, YOLO/Ultralytics, MobileSAM, MuJoCo, and every model before redistributing or using this demo commercially.

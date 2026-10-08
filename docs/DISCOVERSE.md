# DISCOVERSE 末端相机视觉抓取

键盘输入版本使用仓库根目录的 `./install_keyboard.sh` 和 `./run_keyboard.sh`。它启动
文字命令面板，不加载 FunASR、麦克风和音频依赖；详细步骤见 README 顶部“键盘仿真快速开始”。

启动：

```bash
./install_sim.sh  # 新环境或需要补装 MobileSAM 等依赖时执行
./run_sim.sh
./run_sim.sh --device 11         # 指定麦克风
./run_sim.sh --no-voice          # 仅文字输入，仍执行完整视觉流程
./run_sim.sh --yolo-checkpoint /absolute/path/to/best.pt
```

默认使用你配置的 `checkpoint/yolo_blocks_v1.pt`、`checkpoint/mobile_sam.pt` 和
本地 FunASR。模型和阈值来自 `configs/config_file.yaml` 指向的实机配置。
`--yolo-checkpoint` 只覆盖本次仿真。模型类别需与 `RealtimeDetection.candidate_labels`
匹配（当前 `block/cube`）。不需要启动 arm-sdk 服务、CAN 或真实摄像头。

## 当前完整流程

```text
FunASR / 文字 → 白名单与确认
  → 机械臂末端 eye_arm 相机 RGB-D 与同一时刻末端姿态
  → 自训练 YOLO → HSV 蓝绿分类 → 连续帧确认 → 唯一目标与时效检查
  → MobileSAM 框选分割 → 掩码面积/框内比例/有效深度检查
  → 相机点云 → 手眼转换 → SimpleGrasp 已知立方体可见表面拟合、边方向与宽度
  → 预测时效/高度/宽度/逆运动学检查
  → 张爪 → 预抓取位 → 直线下降 → 夹持接触确认/收紧重试
  → 直线抬升 → 预放置位 → 直线下降 → 松爪 → 直线抬升 → 返回观察位
```

**YOLO 和 MobileSAM 都只使用末端相机 `eye_arm`，不再使用外部检测相机。**
上方第三视角只用于观察机械臂动作，下方为第一视角和其检测框。运动时清除旧检测框，
继续显示末端实时画面；分割快照在右侧独立显示，避免把旧掩码叠在移动中的画面上。
右侧同时显示预测位置、物体宽度、角度；日志显示十步动作及失败恢复情况。

实际操作：等待语音模型与检测就绪，说“抓取蓝色积木”或“抓取绿色积木”，核对后确认。
也可直接输入文字。完成后重置场景，再试另一颜色；放置区已有积木时拒绝第二次放置。
手动备用流程为在下拉框选择颜色，然后“预测抓取”或“开始抓取”，仍走同一 YOLO/SAM
链路。“拍照”清除旧目标；尚未移植实机鼠标正负点选与持续自动抓取 UI。

## 与真机共用及适配的部分

| 部分 | 当前实现 |
| --- | --- |
| 语音 | 共用 FunASR 子进程、VoiceCommandPanel、口令白名单和确认规则 |
| 检测 | 共用 AirbotYolo、HSV 分类、DetectionStabilizer 和唯一目标筛选 |
| 分割与准备 | 实机和仿真共同调用 `grasp_preparation.prepare_color_grasp` |
| 抓取预测 | 共用 AirbotSegment、SimpleGrasp 和已知立方体可见表面拟合；仿真尺寸为 40 mm |
| RGB-D | MuJoCo 米制深度，适配成 RealSense 风格的 CV 相机点云；帧中冻结末端姿态 |
| 手眼 | 从仿真相机与末端变换得到 camera-to-end，并转换 OpenGL/CV 坐标；不是实机外参 |
| 运动 | 与真机相同十步顺序和 10 cm 接近高度；MuJoCo 关节控制/IK 代替 SDK/MoveIt |
| 夹持 | 沿用实机宽度余量、压缩量、收紧次数、力上限、采样间隔、超时与连续确认参数；实际双侧接触、速度和力检查 |
| 失败恢复 | 抓取区未夹住时松爪上抬；放置区夹持时松爪上抬；再尝试返回观察位，保留原始错误 |

仿真位置伺服的力反馈是 MuJoCo actuator force，不等价于 SDK 电机 `eef_eff`。
没有伪造电机温度、故障码、控制租约或通信异常。运动没有复现 MoveIt 碰撞规划器。
夹爪靠实际碰撞和摩擦搬运，不使用焊接、物体坐标跟随或真值抓取回退。
真值仅用于放置区占用及实际抬升/落地的评估。

仿真单独使用 `configs/discoverse_sim.yaml`：观察位接近实机位置，姿态调整为可看清目标；
DISCOVERSE link6 原点位于手指附近，和实机工具原点不同，因此单独配置等效
`gripper_length=0.01 m` 与 `min_grasp_z=0.028 m`，防止手指碰桌面。
保留 SimpleGrasp 的原始高度公式与最低高度钳制，**不修改实机配置文件**。
仿真预测的方向四元数用于抓取段的 IK，放置位和观察位使用各自姿态。

这些适配使代码流程更接近真机，但尚不是数字孪生或真机安全验收标准。真实镜头畸变、
深度噪声、光照、机器人动力学和控制延迟尚未逐项标定。当前几何模型针对已知尺寸立方体，任意相机姿态、遮挡和复杂物体下仍需评估。
默认桌面的蓝绿积木也需在当前权重和配置下运行集成验证；仿真成功不能保证真机成功。

## 验证命令

```bash
./run_sim.sh --headless --text '抓取蓝色积木'
./run_sim.sh --headless --text '抓取绿色积木'
./run_tests.sh
# 加载真实权重、末端相机 RGB-D、SAM 并实际执行物理抓放：
GRASP_TEST_YOLO_SIM=1 MUJOCO_GL=egl MPLBACKEND=Agg PYTHONPATH=app \
  ./venv/bin/python -m unittest -v tests.test_discoverse_vision tests.test_discoverse_sim
```

无窗口视觉运行默认选 EGL，需要可用 OpenGL/EGL。每次 CLI 启动创建新场景，失败退出码
非零；GUI 中推理与分割在后台线程，MuJoCo 与渲染只在主线程操作。
`run_tests.sh` 默认跳过真实模型/渲染集成测试，需以上环境变量开启。
检测过期、漏检、多目标、分割失败、准备超时均不启动运动。

仅物理调试、不使用视觉和 GL：

```bash
./run_sim.sh --ground-truth --headless --text '抓取蓝色积木'
```

这是显式调试模式，不代表视觉成功，也不会作为默认流程的自动备用方式。

## 安装与排障

安装脚本复用项目 `venv`，缺少时创建。DISCOVERSE 保存在被 Git 忽略的
`vendor/DISCOVERSE`，必须保留其模型文件。版本锁定为
`d67f47c084aba0e0cf422a8725235f8b9238655a`（1.9.0），MuJoCo 3.13.0，课程验证环境为 Ubuntu 22.04/Python 3.10；项目运行时兼容 Python 3.10–3.12。
可用 `DISCOVERSE_SOURCE=/path/to/DISCOVERSE ./install_sim.sh` 指定已有源码，脚本不覆盖它。
`GRASP_SIM_PYTHON` 可指定另一现有 Python 环境，安装和启动应保持一致。

语音首次使用可能下载模型；Ubuntu 录音需要 `libportaudio2`。桌面显示需要 Qt/OpenGL，
离屏 GUI 可设 `QT_QPA_PLATFORM=offscreen MUJOCO_GL=egl`。
上游可能打印缺少 `gaussian_renderer` 的回溯和 warning，是可选 3DGS 探测，不影响本入口。
实机训练模型有仿真域差异，更换场景/姿态后仍可能漏检；不能靠真值绕过。

## 主要文件

- `app/discoverse_vision.py`：末端相机快照、YOLO、异步共用抓取准备、仿真手眼和点云适配。
- `app/grasp_preparation.py`：实机/仿真共用的分割验证与抓取信息计算。
- `app/airbot_grasp_simple.py` / `app/airbot_segment.py`：原有预测与分割实现。
- `app/discoverse_sim.py`：仿真模型、轨迹、接触确认、失败恢复和成功检查。
- `app/discoverse_voice.py`：上下双视角、语音/文字控制、分割快照和日志。
- `configs/discoverse_sim.yaml`：仅仿真的几何与控制适配参数。

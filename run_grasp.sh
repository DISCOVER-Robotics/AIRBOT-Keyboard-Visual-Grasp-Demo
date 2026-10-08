#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRASP_PYTHON="$PROJECT_DIR/venv/bin/python"
VOICE_PYTHON="${GRASP_VOICE_PYTHON:-$GRASP_PYTHON}"
VOICE_DEVICE="${GRASP_VOICE_DEVICE:-11}"
CHECK_ONLY=false
WITH_SIM=false

usage() {
    cat <<'EOF'
用法：
  ./run_grasp.sh                 启动抓取 GUI
  ./run_grasp.sh --with-sim      真机 GUI + 只读同步仿真窗口
  ./run_grasp.sh --check         只检查运行环境，不连接硬件
  ./run_grasp.sh --device 编号   使用指定麦克风启动

默认使用项目 venv 同时运行视觉和语音。也可通过 GRASP_VOICE_PYTHON
和 GRASP_VOICE_DEVICE 覆盖默认值。
EOF
}

while (($# > 0)); do
    case "$1" in
        --with-sim)
            WITH_SIM=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --device)
            if (($# < 2)) || [[ ! "$2" =~ ^[0-9]+$ ]]; then
                echo "错误：--device 后必须提供非负整数设备编号。" >&2
                exit 2
            fi
            VOICE_DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误：未知参数 $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "错误：找不到可执行文件 $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "错误：缺少文件 $1" >&2
        exit 1
    fi
}

require_executable "$GRASP_PYTHON"
require_executable "$VOICE_PYTHON"
require_file "$PROJECT_DIR/app/airbot_interface.py"
require_file "$PROJECT_DIR/app/voice_asr_worker.py"
require_file "$PROJECT_DIR/configs/sam_simplegrasp.yaml"
require_file "$PROJECT_DIR/checkpoint/mobile_sam.pt"
require_file "$PROJECT_DIR/checkpoint/yolo_blocks_v1.pt"

echo "正在检查抓取环境..."
PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONWARNINGS=ignore "$GRASP_PYTHON" - <<'PY'
import sys

if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
    raise SystemExit(
        f"错误：抓取环境需要 Python 3.10、3.11 或 3.12，当前为 {sys.version.split()[0]}"
    )

import arm_sdk
import cv2
import PyQt6
import torch
import mobile_sam

if arm_sdk.__version__ != "5.2.2":
    raise SystemExit(
        f"错误：arm-sdk 需要 5.2.2，当前为 {arm_sdk.__version__}"
    )
print(f"抓取环境：OK（Python {sys.version.split()[0]}，arm-sdk {arm_sdk.__version__}）")
PY

echo "正在检查语音环境..."
PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}" "$VOICE_PYTHON" - <<'PY'
import sys
import sounddevice
import soundfile
import funasr

print(f"语音环境：OK（Python {sys.version.split()[0]}）")
PY

if command -v ss >/dev/null 2>&1 && ! ss -ltn 2>/dev/null | awk '$4 ~ /:50051$/ { found = 1 } END { exit !found }'; then
    echo "提示：未检测到 50051 端口监听。正常启动前请先在另一个终端启动 airbot-arm。"
fi

if "$WITH_SIM"; then
    "$GRASP_PYTHON" -c 'import mujoco, discoverse, OpenGL'
    export GRASP_LIVE_MIRROR=1
fi

if "$CHECK_ONLY"; then
    echo "环境检查完成；--check 未连接相机或机械臂。"
    exit 0
fi

cd "$PROJECT_DIR"
export GRASP_VOICE_PYTHON="$VOICE_PYTHON"
export GRASP_VOICE_DEVICE="$VOICE_DEVICE"
export PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}"

"$GRASP_PYTHON" - <<'PY_CHECK'
from airbot_camera import check_camera_connection, CameraConnectionError
try:
    check_camera_connection()
except CameraConnectionError as exc:
    raise SystemExit('启动取消：' + str(exc))
PY_CHECK

echo "语音输入设备：$GRASP_VOICE_DEVICE"
echo "警告：即将启动 GUI；程序启动后机械臂会自动移动到配置的观察位。"
echo "请确认 airbot-arm 已启动，观察位、标定参数和机械臂周围空间均安全。"
exec "$GRASP_PYTHON" "$PROJECT_DIR/app/airbot_interface.py"

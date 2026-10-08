#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${GRASP_KEYBOARD_PYTHON:-$PROJECT_DIR/venv/bin/python}"
WITH_SIM=false
CHECK_ONLY=false

usage() {
    cat <<'EOF'
用法：
  ./run_keyboard_robot.sh             启动实体机械臂键盘控制 GUI
  ./run_keyboard_robot.sh --with-sim  实体机械臂 GUI + 只读仿真反馈窗口
  ./run_keyboard_robot.sh --check     只检查依赖，不连接设备
EOF
}

while (($# > 0)); do
    case "$1" in
        --with-sim) WITH_SIM=true; shift ;;
        --check) CHECK_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "找不到 Python 环境：$PYTHON。请先完成项目安装。" >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - <<'PY'
import sys
if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
    raise SystemExit(f"需要 Python 3.10、3.11 或 3.12，当前为 {sys.version.split()[0]}")
import arm_sdk, cv2, PyQt6, torch, mobile_sam
if arm_sdk.__version__ != "5.2.2":
    raise SystemExit(f"arm-sdk 需要 5.2.2，当前为 {arm_sdk.__version__}")
print(f"键盘真机环境：OK（Python {sys.version.split()[0]}，arm-sdk {arm_sdk.__version__}）")
PY

if "$WITH_SIM"; then
    "$PYTHON" -c 'import mujoco, discoverse, OpenGL'
    export GRASP_LIVE_MIRROR=1
fi

if "$CHECK_ONLY"; then
    echo "环境检查完成；未连接相机或机械臂。"
    exit 0
fi

"$PYTHON" - <<'PY'
from airbot_camera import check_camera_connection, CameraConnectionError
try:
    check_camera_connection()
except CameraConnectionError as exc:
    raise SystemExit('启动取消：' + str(exc))
PY

echo "警告：即将启动键盘真机 GUI；程序启动后机械臂会自动移动到配置的观察位。"
echo "请确认 airbot-arm 已启动，观察位、标定参数和机械臂周围空间均安全。"
exec "$PYTHON" "$PROJECT_DIR/app/airbot_interface.py"

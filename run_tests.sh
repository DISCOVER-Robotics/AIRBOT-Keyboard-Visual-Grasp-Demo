#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "错误：找不到项目环境 $PYTHON，请先执行 ./install.sh。" >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

"$PYTHON" -m unittest -v \
    tests.test_discoverse_sim \
    tests.test_live_mirror \
    tests.test_camera_connection \
    tests.test_camera_geometry \
    tests.test_discoverse_vision \
    tests.test_airbot_arm \
    tests.test_airbot_yolo \
    tests.test_grasp_recovery \
    tests.test_block_geometry \
    tests.test_grasp_clearance \
    tests.test_grasp_safety \
    tests.test_grasp_preparation \
    tests.test_grasp_preview \
    tests.test_place_pose \
    tests.test_realtime_pipeline \
    tests.test_voice_asr_worker \
    tests.test_voice_commands

# Keep Qt in its own process so QApplication is initialized before widgets.
QT_QPA_PLATFORM=offscreen "$PYTHON" -m unittest -v \
    tests.test_voice_panel

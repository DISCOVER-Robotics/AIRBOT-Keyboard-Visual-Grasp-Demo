#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_SOURCE="${DISCOVERSE_SOURCE:-$PROJECT_DIR/vendor/DISCOVERSE}"
SIM_REVISION=d67f47c084aba0e0cf422a8725235f8b9238655a
PYTHON="${GRASP_SIM_PYTHON:-$PROJECT_DIR/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    python3 -m venv "$PROJECT_DIR/venv"
    PYTHON="$PROJECT_DIR/venv/bin/python"
fi
if [[ ! -d "$SIM_SOURCE" ]]; then
    git init "$SIM_SOURCE"
    git -C "$SIM_SOURCE" remote add origin https://github.com/discoverse-dev/DISCOVERSE.git
    git -C "$SIM_SOURCE" fetch --depth 1 origin "$SIM_REVISION"
    git -C "$SIM_SOURCE" checkout --detach FETCH_HEAD
elif [[ ! -f "$SIM_SOURCE/discoverse/robots_env/airbot_play_base.py" ]]; then
    echo "DISCOVERSE_SOURCE 不是兼容的 DISCOVERSE 源码目录：$SIM_SOURCE" >&2
    exit 1
fi
"$PYTHON" -m pip install -e "$SIM_SOURCE" -r "$PROJECT_DIR/requirements-sim.txt"
echo "仿真已安装。启动：./run_sim.sh；无麦克风演示：./run_sim.sh --no-voice"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_SOURCE="${DISCOVERSE_SOURCE:-$PROJECT_DIR/vendor/DISCOVERSE}"
SIM_REVISION=d67f47c084aba0e0cf422a8725235f8b9238655a
PYTHON="${GRASP_KEYBOARD_PYTHON:-$PROJECT_DIR/venv-keyboard/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3 || true)"
    fi
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "未找到 Python，请安装 Python 3.10 或设置 PYTHON_BIN。" >&2
        exit 1
    fi
    VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    case "$VERSION" in 3.10|3.11|3.12) ;; *)
        echo "键盘演示需要 Python 3.10、3.11 或 3.12，当前为 $VERSION。" >&2
        exit 1
    esac
    "$PYTHON_BIN" -m venv "$PROJECT_DIR/venv-keyboard"
    PYTHON="$PROJECT_DIR/venv-keyboard/bin/python"
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

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install -e "$SIM_SOURCE" -r "$PROJECT_DIR/requirements-sim-keyboard.txt"
echo "键盘仿真环境已安装。启动：./run_keyboard.sh"

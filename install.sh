#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
SDK_WHEEL="${ARM_SDK_WHEEL:-}"
ARM_DEB="${AIRBOT_ARM_DEB:-}"
SKIP_SYSTEM=false

usage() {
    cat <<'EOF'
用法：
  ./install.sh [--sdk-wheel FILE] [--arm-deb FILE] [--skip-system]

推荐把 5.2.2 软件包放到项目 packages/ 目录后直接运行 ./install.sh。
也可通过 ARM_SDK_WHEEL、AIRBOT_ARM_DEB、PYTHON_BIN、PIP_INDEX_URL 设置路径。
EOF
}

while (($# > 0)); do
    case "$1" in
        --sdk-wheel) SDK_WHEEL="${2:?--sdk-wheel 缺少路径}"; shift 2 ;;
        --arm-deb) ARM_DEB="${2:?--arm-deb 缺少路径}"; shift 2 ;;
        --skip-system) SKIP_SYSTEM=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
    esac
done

find_one() {
    local pattern="$1"
    find "$PROJECT_DIR/packages" -maxdepth 1 -type f -name "$pattern" \
        -print -quit 2>/dev/null || true
}

[[ -n "$SDK_WHEEL" ]] || SDK_WHEEL="$(find_one 'arm_sdk-5.2.2-*.whl')"
[[ -n "$ARM_DEB" ]] || ARM_DEB="$(find_one 'airbot-arm_5.2.2_*.deb')"

if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in /usr/bin/python3.10 /usr/bin/python3.11 /usr/bin/python3.12; do
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
fi
if [[ "$PYTHON_BIN" != */* ]] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "错误：找不到支持的 Python。请安装 Python 3.10，或使用 PYTHON_BIN=/path/to/python。" >&2
    exit 1
fi
PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
    3.10|3.11|3.12) ;;
    *)
        echo "错误：项目需要 Python 3.10、3.11 或 3.12，当前为 Python $PYTHON_VERSION ($PYTHON_BIN)。" >&2
        exit 1
        ;;
esac
PYTHON_VENV_PACKAGE="python${PYTHON_VERSION}-venv"

SDK_ALREADY_INSTALLED=false
if [[ -x "$PROJECT_DIR/venv/bin/python" ]] \
        && "$PROJECT_DIR/venv/bin/python" -c \
        "import arm_sdk; raise SystemExit(arm_sdk.__version__ != '5.2.2')" \
        >/dev/null 2>&1; then
    SDK_ALREADY_INSTALLED=true
fi
if [[ ! -f "$SDK_WHEEL" && "$SDK_ALREADY_INSTALLED" == false ]]; then
    echo "错误：缺少 arm-sdk 5.2.2 wheel。请放入 packages/ 或使用 --sdk-wheel 指定。" >&2
    exit 1
fi
if [[ ! -f "$ARM_DEB" && "$SKIP_SYSTEM" == false ]]; then
    echo "错误：缺少 airbot-arm 5.2.2 deb。请放入 packages/、使用 --arm-deb 指定，" >&2
    echo "或在已安装服务的机器上增加 --skip-system。" >&2
    exit 1
fi

if [[ "$SKIP_SYSTEM" == false ]]; then
    sudo apt-get update
    sudo apt-get install -y "$PYTHON_VENV_PACKAGE" libxcb-cursor0
    sudo apt-get install -y "$ARM_DEB"
fi

if [[ ! -x "$PROJECT_DIR/venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$PROJECT_DIR/venv"
else
    VENV_VERSION="$($PROJECT_DIR/venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$VENV_VERSION" != "$PYTHON_VERSION" ]]; then
        echo "错误：现有 venv 使用 Python $VENV_VERSION，但当前指定的是 Python $PYTHON_VERSION。" >&2
        echo "请移除或重命名 venv 后重新运行，例如：mv venv venv-python${VENV_VERSION}-old" >&2
        exit 1
    fi
fi

PYTHON="$PROJECT_DIR/venv/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel -i "$PIP_INDEX_URL"
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt" -i "$PIP_INDEX_URL"
if [[ -f "$SDK_WHEEL" ]]; then
    "$PYTHON" -m pip install --force-reinstall "$SDK_WHEEL"
fi



PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
import arm_sdk
import cv2
import mobile_sam
import PyQt6
import torch
import ultralytics

assert arm_sdk.__version__ == "5.2.2", arm_sdk.__version__
print("统一环境安装完成：Python、视觉、键盘控制、arm-sdk 5.2.2 均可导入")
PY

echo "下一步：./run_grasp.sh --check"

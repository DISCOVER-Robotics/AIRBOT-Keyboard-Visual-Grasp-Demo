#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${GRASP_KEYBOARD_PYTHON:-$PROJECT_DIR/venv-keyboard/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: keyboard environment not found at $PYTHON. Run ./install_keyboard.sh first." >&2
    exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

"$PYTHON" -m unittest -v \
    tests.test_discoverse_sim \
    tests.test_discoverse_vision \
    tests.test_airbot_yolo \
    tests.test_block_geometry \
    tests.test_grasp_safety \
    tests.test_grasp_preparation \
    tests.test_grasp_preview \
    tests.test_voice_commands

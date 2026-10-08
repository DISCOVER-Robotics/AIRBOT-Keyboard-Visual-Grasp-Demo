#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${GRASP_SIM_PYTHON:-$PROJECT_DIR/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "找不到 Python 环境，请先执行 ./install_sim.sh" >&2
    exit 1
fi
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" app/discoverse_voice.py "$@"

#!/usr/bin/env bash
set -euo pipefail
RDP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$RDP_DIR/.venv/bin/python" ]]; then
    export PYTHON_BIN="$RDP_DIR/.venv/bin/python"
  else
    export PYTHON_BIN=/home/ljl/RDP_vitamin/.venv/bin/python
  fi
fi
exec "$PYTHON_BIN" "$RDP_DIR/rdp_baseline/server.py" "$@"

#!/usr/bin/env bash
# One Optiver campaign: freeze, run, seat the incumbent, read the board.
# Usage: examples/optiver/campaign.sh [stocks]   ("all", a count such as "20", or "0,1,2,3")
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/fy ] && PATH=".venv/bin:$PATH"
command -v fy >/dev/null || { echo "fy not found: uv venv .venv && uv pip install -e '.[dev]'" >&2; exit 1; }
export PYTHONPATH=examples FORESTRY_ROOT=.forestry-optiver
export FORESTRY_ACTOR="${FORESTRY_ACTOR:-$USER}"

echo "== freeze"
DS=$(fy freeze optiver.capture:freeze "${1:-20}")
C="optiver.declarations --dataset $DS"
echo "dataset $DS"

echo "== run"
fy run $C

echo "== seat the incumbent, then read every pipeline against it"
fy decide $C ridge_3m --kind promote --why "incumbent: the model in production" >/dev/null
fy board $C

echo "== history"
fy history $C

echo
echo "Next: fy board $C <pipeline> for one in detail, then"
echo "      fy decide $C <pipeline> --kind promote|reject --why '...'"

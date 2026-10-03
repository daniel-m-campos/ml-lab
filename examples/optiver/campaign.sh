#!/usr/bin/env bash
# One Optiver campaign: freeze, run, seat the incumbent, compare the rest.
# Usage: examples/optiver/campaign.sh [stocks]   ("all", a count such as "20", or "0,1,2,3")
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/fy ] && PATH=".venv/bin:$PATH"
command -v fy >/dev/null || { echo "fy not found: uv venv .venv && uv pip install -e '.[dev]'" >&2; exit 1; }
export PYTHONPATH=examples FORESTRY_ROOT=.forestry-optiver

echo "== freeze"
DS=$(fy freeze optiver.capture:freeze "${1:-20}")
C="optiver.declarations --dataset $DS"
echo "dataset $DS"

echo "== run"
fy run $C
fy board $C

echo "== seat the incumbent"
fy decide $C ridge_3m --kind promote --why "incumbent: the model in production" >/dev/null

echo "== compare every scored candidate to it"
fy compare $C

echo "== history"
fy history $C

echo
echo "Next: read the comparisons, then  fy decide $C <pipeline> --kind promote|reject --why '...'"

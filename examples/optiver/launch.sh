#!/usr/bin/env bash
# The Optiver example: ingest, run, read the scores.
# Usage: examples/optiver/launch.sh [stocks]   ("all", a count such as "20", or "0,1,2,3")
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/lab ] && PATH=".venv/bin:$PATH"
command -v lab >/dev/null || { echo "lab not found: uv venv .venv && uv pip install -e '.[dev]'" >&2; exit 1; }
export ML_LAB_ROOT="${ML_LAB_ROOT:-.ml-lab-optiver}"
export ML_LAB_ACTOR="${ML_LAB_ACTOR:-$USER}"
D=examples/optiver/experiment.py

echo "== ingest"
lab ingest examples/optiver/dataset.py "${1:-20}"

echo "== run"
lab run $D

echo "== latest aggregate scores, horizon 1"
sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite "
SELECT l.name, s.metric, round(s.value, 2) AS value FROM latest_score l
JOIN aggregate_score s ON s.score = l.score
WHERE s.window = '1' ORDER BY s.metric, s.value DESC"

echo
echo "Next: export ML_LAB_ROOT=$ML_LAB_ROOT, add a Pipeline to $D (or a file of pipelines), lab run $D <file>,"
echo "      then sqlite3 -box \$ML_LAB_ROOT/ml_lab.sqlite over fold_score, aggregate_score, fit, failure."

#!/usr/bin/env bash
# One Optiver campaign: ingest, run, read the scores.
# Usage: examples/optiver/campaign.sh [stocks]   ("all", a count such as "20", or "0,1,2,3")
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/fy ] && PATH=".venv/bin:$PATH"
command -v fy >/dev/null || { echo "fy not found: uv venv .venv && uv pip install -e '.[dev]'" >&2; exit 1; }
export FORESTRY_ROOT=.forestry-optiver
export FORESTRY_ACTOR="${FORESTRY_ACTOR:-$USER}"
D=examples/optiver/experiment.py

echo "== ingest"
fy ingest examples/optiver/capture.py "${1:-20}"

echo "== run"
fy run $D

echo "== latest aggregate scores, age 1"
sqlite3 -box $FORESTRY_ROOT/forestry.sqlite "
SELECT p.name, s.metric, round(s.value, 2) AS value FROM latest_score l
JOIN aggregate_score s ON s.score = l.score JOIN pipeline p ON p.id = l.pipeline
WHERE s.age = 1 ORDER BY s.metric, s.value DESC"

echo
echo "Next: export FORESTRY_ROOT=$FORESTRY_ROOT, add a Pipeline to $D (or a file of pipelines), fy run $D <file>,"
echo "      then sqlite3 -box \$FORESTRY_ROOT/forestry.sqlite over fold_score, aggregate_score, fit, failure."

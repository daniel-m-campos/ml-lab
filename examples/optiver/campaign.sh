#!/usr/bin/env bash
# One Optiver campaign, start to seal. Usage: examples/optiver/campaign.sh [stocks]
# stocks: "all", a count ("20", the default) or a list ("0,1,2,3").
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=examples FORESTRY_ROOT=.forestry-optiver
DECL=optiver.declarations

echo "== freeze"
DS=$(fy freeze optiver.capture:freeze "${1:-20}")
echo "dataset $DS"

echo "== screen: fit metrics, human gate"
fy run "$DECL" --dataset "$DS"
fy board "$DECL" --dataset "$DS"

# A person reads the board and gates each row: fy gate <id> --advance|--stop --why "...".
# This template advances every pending row with one recorded reason.
fy gate "$DECL" --dataset "$DS" --pending --advance --why "screening: every pipeline clears corr 0.15"

echo "== funnel: quick sim grid, full sim against the baseline"
fy run "$DECL" --dataset "$DS"
fy board "$DECL" --dataset "$DS"

echo "== history"
fy history "$DECL" --dataset "$DS"

echo "== seal"
fy seal "$DECL" --dataset "$DS" || true

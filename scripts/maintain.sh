#!/usr/bin/env bash
# ChalkIQ maintenance — delegates to autonomous daily pipeline.
# Production runs via GitHub Actions (.github/workflows/daily.yml) at midnight ET.
# Use this script only for local recovery or testing.
#
# Usage:
#   bash scripts/maintain.sh
#   bash scripts/maintain.sh --game-day    # odds + newsletter only
#   bash scripts/maintain.sh --skip-odds

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin/python3"
LOGS="$PROJECT_DIR/logs"
GAME_DAY=false
SKIP_ODDS=false

for arg in "$@"; do
  case "$arg" in
    --game-day) GAME_DAY=true ;;
    --skip-odds) SKIP_ODDS=true ;;
  esac
done

mkdir -p "$LOGS"
cd "$PROJECT_DIR"

echo "=== ChalkIQ Maintenance  $(date) ==="
echo "  (Production: GitHub Actions daily.yml at midnight ET)"

EXTRA=()
[[ "$GAME_DAY" == true ]] && EXTRA+=(--game-day)
[[ "$SKIP_ODDS" == true ]] && EXTRA+=(--skip-odds)

$VENV scripts/daily_collect.py "${EXTRA[@]}" \
  >> "$LOGS/daily_collect.log" 2>&1 \
  && echo "  done (see logs/daily_collect.log)" \
  || echo "  daily_collect failed — check logs/daily_collect.log"

echo "=== Done ==="

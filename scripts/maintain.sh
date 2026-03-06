#!/usr/bin/env bash
# ChalkIQ maintenance script
# Run manually or via cron. Pulls latest code, refreshes box scores,
# and keeps polling daemons running in the background.
#
# Usage:
#   bash scripts/maintain.sh           # full refresh + restart daemons
#   bash scripts/maintain.sh --no-pull # skip git pull (use local code)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin/python3"
LOGS="$PROJECT_DIR/logs"
NO_PULL=false

for arg in "$@"; do
  [[ "$arg" == "--no-pull" ]] && NO_PULL=true
done

mkdir -p "$LOGS"
cd "$PROJECT_DIR"

echo "=== ChalkIQ Maintenance  $(date) ==="

# ── 1. Pull latest code ───────────────────────────────────────────────────────
if [ "$NO_PULL" = false ]; then
  echo "[1/5] git pull..."
  git pull --ff-only
else
  echo "[1/5] skipping git pull"
fi

# ── 2. Kill stale polling daemons ─────────────────────────────────────────────
echo "[2/5] killing old poll processes..."
pkill -f "poll_odds.py"   2>/dev/null && echo "  killed poll_odds.py"   || echo "  poll_odds.py not running"
pkill -f "poll_props.py"  2>/dev/null && echo "  killed poll_props.py"  || echo "  poll_props.py not running"
sleep 1

# ── 3. Fetch new box scores (incremental — skips already-cached games) ────────
echo "[3/5] fetching box scores..."
$VENV scripts/fetch_boxscores.py --seasons 2026 \
  >> "$LOGS/boxscores.log" 2>&1 \
  && echo "  done (see logs/boxscores.log)" \
  || echo "  fetch_boxscores failed — check logs/boxscores.log"

# ── 4. Start poll_odds.py daemon ─────────────────────────────────────────────
echo "[4/5] starting poll_odds.py (every 30 min)..."
nohup $VENV scripts/poll_odds.py --interval 30 \
  >> "$LOGS/poll_odds.log" 2>&1 &
echo "  PID $!  ->  logs/poll_odds.log"

# ── 5. Start poll_props.py daemon ────────────────────────────────────────────
echo "[5/5] starting poll_props.py (every 60 min)..."
nohup $VENV scripts/poll_props.py --interval 60 --hours 12 \
  >> "$LOGS/poll_props.log" 2>&1 &
echo "  PID $!  ->  logs/poll_props.log"

echo ""
echo "=== Done. Running daemons:"
pgrep -a -f "poll_odds.py\|poll_props.py" || echo "  (none found)"
echo ""
echo "Tail logs with:"
echo "  tail -f $LOGS/poll_odds.log"
echo "  tail -f $LOGS/poll_props.log"

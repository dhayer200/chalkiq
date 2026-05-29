#!/usr/bin/env bash
# ChalkIQ maintenance script — multi-sport
# Run manually or via cron. Pulls latest code, refreshes box scores,
# and keeps polling daemons running in the background.
#
# Usage:
#   bash scripts/maintain.sh                   # full refresh + restart daemons
#   bash scripts/maintain.sh --no-pull         # skip git pull (use local code)
#   DIVISIONS=mens,nba bash scripts/maintain.sh  # custom sport selection

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin/python3"
LOGS="$PROJECT_DIR/logs"
NO_PULL=false

# Default divisions — add/remove as needed
DIVISIONS="${DIVISIONS:-mens,womens}"

for arg in "$@"; do
  [[ "$arg" == "--no-pull" ]] && NO_PULL=true
done

mkdir -p "$LOGS"
cd "$PROJECT_DIR"

echo "=== ChalkIQ Maintenance  $(date) ==="
echo "  Divisions: $DIVISIONS"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
if [ "$NO_PULL" = false ]; then
  echo "[1/7] git pull..."
  git pull --ff-only
else
  echo "[1/7] skipping git pull"
fi

# ── 2. Kill stale polling daemons ─────────────────────────────────────────────
echo "[2/7] killing old poll processes..."
pkill -f "poll_odds.py"      2>/dev/null && echo "  killed poll_odds.py"      || echo "  poll_odds.py not running"
pkill -f "poll_props.py"     2>/dev/null && echo "  killed poll_props.py"     || echo "  poll_props.py not running"
pkill -f "poll_injuries.py"  2>/dev/null && echo "  killed poll_injuries.py"  || echo "  poll_injuries.py not running"
sleep 1

# ── 3. Fetch new box scores (incremental — skips already-cached games) ────────
echo "[3/7] fetching box scores..."
$VENV scripts/fetch_boxscores.py --seasons 2026 \
  >> "$LOGS/boxscores.log" 2>&1 \
  && echo "  done (see logs/boxscores.log)" \
  || echo "  fetch_boxscores failed — check logs/boxscores.log"

# ── 4. Start poll_odds.py daemon (multi-sport) ────────────────────────────────
echo "[4/7] starting poll_odds.py (all sports, every 30 min)..."
nohup $VENV -u scripts/poll_odds.py --divisions "$DIVISIONS" --interval 30 \
  >> "$LOGS/poll_odds.log" 2>&1 &
echo "  PID $!  ->  logs/poll_odds.log"

# ── 5. Start poll_props.py daemon (multi-sport, budget-limited) ───────────────
echo "[5/7] starting poll_props.py (all sports, every 60 min, max 3 games)..."
nohup $VENV -u scripts/poll_props.py --divisions "$DIVISIONS" --interval 60 --hours 4 --max-games 3 \
  >> "$LOGS/poll_props.log" 2>&1 &
echo "  PID $!  ->  logs/poll_props.log"

# ── 6. Start poll_injuries.py daemon (multi-sport) ────────────────────────────
echo "[6/7] starting poll_injuries.py (all sports, every 10 min)..."
nohup $VENV -u scripts/poll_injuries.py --divisions "$DIVISIONS" --interval 10 \
  >> "$LOGS/poll_injuries.log" 2>&1 &
echo "  PID $!  ->  logs/poll_injuries.log"

# ── 7. Generate newsletter data (for 2PM UTC cron send) ──────────────────────
echo "[7/7] generating newsletter data..."
$VENV scripts/generate_newsletter_data.py \
  >> "$LOGS/newsletter_data.log" 2>&1 \
  && echo "  done (see logs/newsletter_data.log)" \
  || echo "  newsletter data generation failed — check logs/newsletter_data.log"

echo ""
echo "=== Done. Running daemons:"
pgrep -a -f "poll_odds.py\|poll_props.py\|poll_injuries.py" || echo "  (none found)"
echo ""
echo "Tail logs with:"
echo "  tail -f $LOGS/poll_odds.log"
echo "  tail -f $LOGS/poll_props.log"
echo "  tail -f $LOGS/poll_injuries.log"

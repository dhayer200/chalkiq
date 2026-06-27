#!/usr/bin/env bash
# Kapare step 9 — Verifier: lint + test + import smoke
# See /Users/mbp/brain/projects/deep-onboarding.md

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt"
  exit 1
fi

# pytest via module (works even if venv was relocated)
run_pytest() {
  "$PYTHON" -m pytest "$@" 2>/dev/null || {
    echo "  installing pytest..."
    "$PYTHON" -m pip install -q -r requirements-dev.txt
    "$PYTHON" -m pytest "$@"
  }
}

echo "=== ChalkIQ pre-ship ==="

echo "[1/3] compileall..."
"$PYTHON" -m compileall -q src api scripts

echo "[2/3] import smoke..."
"$PYTHON" -c "
from src.ratings.elo import EloEngine, SPORT_CONFIGS
from src.slate.generate import load_engine, ODDS_SPORT
from src.automation.daily import run_daily_collect
assert 'mens' in SPORT_CONFIGS
assert 'cfb' in SPORT_CONFIGS
print('  imports OK')
"

echo "[3/3] pytest..."
run_pytest tests/ -q

echo "=== pre-ship passed ==="

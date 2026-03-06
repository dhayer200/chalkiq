"""
Persistent storage for odds snapshots and CLV records.

Uses JSONL (one JSON object per line) — simple, appendable, grep-friendly.

Files:
    data/odds/snapshots.jsonl  — every odds fetch snapshot
    data/odds/clv.jsonl        — final CLV records (written when game closes)
    data/odds/alerts.jsonl     — line movement alerts
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "odds"


def _path(filename: str, data_dir: Path = _DEFAULT_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / filename


def append(record: dict, filename: str, data_dir: Path = _DEFAULT_DIR) -> None:
    """Append a single record to a JSONL file."""
    with open(_path(filename, data_dir), "a") as f:
        f.write(json.dumps(record) + "\n")


def read_all(filename: str, data_dir: Path = _DEFAULT_DIR) -> list[dict]:
    """Read all records from a JSONL file. Returns [] if file doesn't exist."""
    p = _path(filename, data_dir)
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_snapshot(odds_record: dict, data_dir: Path = _DEFAULT_DIR) -> None:
    """Save a single odds snapshot (one game, one bookmaker)."""
    append(odds_record, "snapshots.jsonl", data_dir)


def save_clv(clv_record: dict, data_dir: Path = _DEFAULT_DIR) -> None:
    """
    Save a CLV record after a game closes.

    clv_record should contain:
    {
        "game_id":        str,
        "game_date":      str,
        "home_espn_id":   str,
        "away_espn_id":   str,
        "home_team":      str,
        "away_team":      str,
        "bookmaker":      str,
        "model_prob_home": float,   # our Elo probability at bet time
        "opening_home_ml": int,
        "closing_home_ml": int,
        "opening_home_prob": float,
        "closing_home_prob": float,
        "clv":            float,    # model_prob_home - closing_home_prob
        "home_won":       bool | None,
        "recorded_at":    str,
    }
    """
    append(clv_record, "clv.jsonl", data_dir)


def save_alert(alert: dict, data_dir: Path = _DEFAULT_DIR) -> None:
    """Save a line movement alert."""
    append(alert, "alerts.jsonl", data_dir)


def load_snapshots(data_dir: Path = _DEFAULT_DIR) -> list[dict]:
    return read_all("snapshots.jsonl", data_dir)


def load_clv_records(data_dir: Path = _DEFAULT_DIR) -> list[dict]:
    return read_all("clv.jsonl", data_dir)


def load_historical_clv_records(data_dir: Path = _DEFAULT_DIR) -> list[dict]:
    return read_all("historical_clv.jsonl", data_dir)


def load_alerts(data_dir: Path = _DEFAULT_DIR) -> list[dict]:
    return read_all("alerts.jsonl", data_dir)


def get_opening_line(
    game_id: str,
    bookmaker: str,
    data_dir: Path = _DEFAULT_DIR,
) -> dict | None:
    """
    Return the first-ever snapshot for this game+bookmaker (the opening line).
    """
    for record in read_all("snapshots.jsonl", data_dir):
        if record.get("game_id") == game_id and record.get("bookmaker") == bookmaker:
            return record
    return None


def get_line_history(
    game_id: str,
    bookmaker: str,
    data_dir: Path = _DEFAULT_DIR,
) -> list[dict]:
    """Return all snapshots for a game+bookmaker in chronological order."""
    return [
        r for r in read_all("snapshots.jsonl", data_dir)
        if r.get("game_id") == game_id and r.get("bookmaker") == bookmaker
    ]

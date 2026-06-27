"""
Position effectiveness score: 0–100 rating for how effective a player is at their position.

CBB: derived from Hollinger Game Score rolling average, normalized per position.
CFB: position-specific stat composites from box score data.

50 = league-average starter at position
70+ = very good | 85+ = elite | below 40 = replacement level
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.players.gamescores import game_score

_ROOT = Path(__file__).resolve().parents[2]

# Position group normalization (CBB)
_CBB_POS_GROUPS = {
    "G": "G", "PG": "G", "SG": "G", "G/F": "G",
    "F": "F", "SF": "F", "PF": "F", "F/C": "F",
    "C": "C", "C/F": "C",
}


def _elo_style_100(raw: float, mean: float, std: float) -> int:
    """Map a stat to 0–100 using z-score → logistic."""
    if std <= 0:
        std = 1.0
    z = (raw - mean) / std
    # logistic: 50 at mean, ~75 at +1σ, ~25 at -1σ
    import math
    score = 100 / (1 + math.exp(-z * 0.9))
    return max(0, min(100, round(score)))


@dataclass
class PlayerEffectiveness:
    player_id: str
    name: str
    team_id: str
    position: str
    division: str
    games: int = 0
    score: int = 50          # 0–100 effectiveness
    raw_avg: float = 0.0     # underlying metric
    trend: str = "stable"    # up | down | stable


@dataclass
class EffectivenessEngine:
    """Aggregate box score files into 0–100 position ratings."""

    division: str
    ratings: dict[str, PlayerEffectiveness] = field(default_factory=dict)

    def load_boxscores(self, limit_files: int = 0) -> int:
        """Process cached box score JSON files. Returns files read."""
        box_dir = _ROOT / "data" / "raw" / self.division / "boxscores"
        if not box_dir.exists():
            return 0

        files = sorted(box_dir.glob("*.json"))
        if limit_files:
            files = files[-limit_files:]

        # Collect per-player game scores for normalization
        by_pos: dict[str, list[float]] = defaultdict(list)
        player_games: dict[str, list[dict]] = defaultdict(list)

        for f in files:
            try:
                game = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for p in game.get("players", []):
                pid = p.get("player_id", "")
                if not pid:
                    continue
                pos = _normalize_pos(p.get("position", ""), self.division)
                raw = _game_metric(p, self.division)
                if raw is None:
                    continue
                by_pos[pos].append(raw)
                player_games[pid].append({**p, "_metric": raw, "_pos": pos})

        # Position means/stds
        pos_stats: dict[str, tuple[float, float]] = {}
        for pos, vals in by_pos.items():
            if len(vals) < 5:
                pos_stats[pos] = (statistics.mean(vals), max(statistics.stdev(vals), 0.01) if len(vals) > 1 else 1.0)
            else:
                pos_stats[pos] = (statistics.mean(vals), max(statistics.stdev(vals), 0.01))

        for pid, games in player_games.items():
            if not games:
                continue
            metrics = [g["_metric"] for g in games]
            recent = metrics[-5:]
            avg = statistics.mean(metrics)
            pos = games[-1]["_pos"]
            mean, std = pos_stats.get(pos, (avg, 1.0))
            score = _elo_style_100(avg, mean, std)

            if len(recent) >= 3:
                first_half = statistics.mean(recent[: len(recent) // 2 or 1])
                second_half = statistics.mean(recent[len(recent) // 2 :])
                if second_half - first_half > std * 0.3:
                    trend = "up"
                elif first_half - second_half > std * 0.3:
                    trend = "down"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            last = games[-1]
            self.ratings[pid] = PlayerEffectiveness(
                player_id=pid,
                name=last.get("name", pid),
                team_id=last.get("team_id", ""),
                position=pos,
                division=self.division,
                games=len(games),
                score=score,
                raw_avg=round(avg, 2),
                trend=trend,
            )

        return len(files)

    def top_players(self, n: int = 10, position: str | None = None) -> list[PlayerEffectiveness]:
        players = list(self.ratings.values())
        if position:
            players = [p for p in players if p.position == position]
        players.sort(key=lambda p: p.score, reverse=True)
        return players[:n]

    def export_json(self, path: Path | None = None) -> Path:
        out = path or _ROOT / "data" / "players" / f"{self.division}_effectiveness.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "division": self.division,
            "n_players": len(self.ratings),
            "players": [
                {
                    "player_id": p.player_id,
                    "name": p.name,
                    "team_id": p.team_id,
                    "position": p.position,
                    "games": p.games,
                    "effectiveness": p.score,
                    "raw_avg": p.raw_avg,
                    "trend": p.trend,
                }
                for p in sorted(self.ratings.values(), key=lambda x: x.score, reverse=True)
            ],
        }
        out.write_text(json.dumps(payload, indent=2))
        return out


def _normalize_pos(pos: str, division: str) -> str:
    pos = (pos or "?").upper().strip()
    if division == "mens":
        return _CBB_POS_GROUPS.get(pos, pos[:1] if pos else "?")
    # CFB positions
    if pos in ("QB",):
        return "QB"
    if pos in ("RB", "FB", "HB"):
        return "RB"
    if pos in ("WR", "TE", "SE", "FL"):
        return "WR"
    if pos in ("LB", "OLB", "ILB", "MLB"):
        return "LB"
    if pos in ("DB", "CB", "S", "SS", "FS"):
        return "DB"
    if pos in ("DL", "DE", "DT", "NT"):
        return "DL"
    return pos[:2] if len(pos) >= 2 else pos


def _game_metric(player: dict, division: str) -> float | None:
    """Single-game raw metric for normalization."""
    if division == "mens":
        gs = player.get("game_score")
        if gs is not None:
            return float(gs)
        # Recompute from stats if present
        try:
            return game_score(
                int(player.get("pts", 0)),
                int(player.get("fgm", 0)), int(player.get("fga", 0)),
                int(player.get("ftm", 0)), int(player.get("fta", 0)),
                int(player.get("oreb", 0)), int(player.get("dreb", 0)),
                int(player.get("stl", 0)), int(player.get("ast", 0)),
                int(player.get("blk", 0)), int(player.get("pf", 0)),
                int(player.get("to", 0)),
            )
        except (TypeError, ValueError):
            return None

    if division == "cfb":
        return player.get("cfb_composite")
    return None


def update_player_ratings(divisions: list[str] | None = None) -> dict[str, int]:
    """Rebuild effectiveness ratings for all divisions with box score data."""
    divisions = divisions or ["mens", "cfb"]
    counts = {}
    for div in divisions:
        engine = EffectivenessEngine(div)
        n_files = engine.load_boxscores()
        if engine.ratings:
            out = engine.export_json()
            print(f"  [players/{div}] {len(engine.ratings)} players from {n_files} games → {out.name}")
            counts[div] = len(engine.ratings)
        else:
            print(f"  [players/{div}] no box score data yet ({n_files} files scanned)")
            counts[div] = 0
    return counts

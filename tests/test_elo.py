"""Baseline regression tests — Kapare T3 foundation."""

from __future__ import annotations

import pytest

from src.ratings.elo import EloEngine, SPORT_CONFIGS, DEFAULT_RATING


class TestEloBasics:
    def test_default_rating(self):
        engine = EloEngine()
        assert engine.rating("999") == DEFAULT_RATING

    def test_home_favorite_wins_increases_rating(self):
        engine = EloEngine(k=24, home_advantage=100, scale=300)
        before = engine.rating("1")
        engine.update("1", "2", home_score=80, away_score=70, neutral=False)
        assert engine.rating("1") > before

    def test_neutral_home_advantage_zero_effect(self):
        engine = EloEngine(home_advantage=100, scale=300)
        p_home = engine.win_prob("1", "2", neutral=True)
        p_away = engine.win_prob("2", "1", neutral=True)
        assert abs(p_home + p_away - 1.0) < 1e-9

    def test_sport_configs_active_divisions(self):
        assert "mens" in SPORT_CONFIGS
        assert "cfb" in SPORT_CONFIGS
        assert SPORT_CONFIGS["cfb"]["season_boundary"] == "cfb"


class TestCfbSeasonBoundary:
    def test_january_bowl_games_same_season(self):
        engine = EloEngine(season_boundary="cfb", season_regress=0.4)
        engine.update("1", "2", 28, 14, date="2026-01-01")
        assert engine._last_season_year == 2025

    def test_august_starts_new_season(self):
        engine = EloEngine(season_boundary="cfb", season_regress=0.4)
        engine.update("1", "2", 28, 14, date="2025-08-30")
        assert engine._last_season_year == 2025
        engine.update("1", "2", 35, 10, date="2026-08-29")
        assert engine._last_season_year == 2026

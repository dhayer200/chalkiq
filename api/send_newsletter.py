"""POST /api/send_newsletter — Daily newsletter cron (protected by CRON_SECRET).

Reads pre-generated content from newsletter_content table (written by
scripts/generate_newsletter_data.py) and sends personalized emails via Resend.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api._shared.db import get_active_subscribers, get_latest_newsletter_content
from api._shared.email import RESEND_FROM

CRON_SECRET = os.environ.get("CRON_SECRET", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
SITE_URL = os.environ.get("SITE_URL", "https://chalkiq.com")


def _fmt_ml(ml: int | None) -> str:
    if ml is None:
        return "?"
    return f"+{ml}" if ml > 0 else str(ml)


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1%}"


def _generate_picks_html(tier: str) -> tuple[str, str]:
    """Generate newsletter HTML from live model data. Returns (subject, html_body)."""
    today = date.today()
    content = get_latest_newsletter_content()

    # Determine which sports are in today's content
    sport_labels = []
    if content and content.get("sports"):
        sport_labels = [s["label"] for s in content["sports"] if s.get("n_bets", 0) > 0]

    sport_tag = ", ".join(sport_labels) if sport_labels else "No Games"
    subject = f"ChalkIQ Picks — {today.strftime('%B %d, %Y')} ({sport_tag})"

    # Header
    header = f"""
    <div style="background:#0f172a;color:#f1f5f9;padding:32px;font-family:system-ui,sans-serif;max-width:600px;margin:0 auto">
        <div style="text-align:center;margin-bottom:24px">
            <span style="background:#22c55e;color:#0a1628;padding:4px 10px;border-radius:6px;font-weight:800;font-size:14px">C</span>
            <span style="font-weight:700;font-size:18px;margin-left:8px">ChalkIQ</span>
        </div>
        <h1 style="text-align:center;font-size:20px;margin-bottom:8px">{today.strftime('%A, %B %d')}</h1>
        <p style="text-align:center;color:#94a3b8;font-size:14px;margin-bottom:24px">Daily Picks — {sport_tag}</p>
        <div style="border-top:1px solid #1e293b;padding-top:20px">
    """

    body = ""

    if not content or not content.get("sports"):
        body = """
            <p style="color:#94a3b8;font-size:14px;text-align:center">No games today. See you tomorrow.</p>
        """
    else:
        for sport in content["sports"]:
            bets = sport.get("bet_sheet", [])
            clv = sport.get("clv_stats", {})
            label = sport.get("label", sport.get("division", ""))

            # Sport header
            body += f"""
                <div style="margin-bottom:24px">
                <p style="color:#22c55e;font-weight:600;font-size:14px;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">{label}</p>
            """

            # Season stats from real CLV data
            if clv:
                avg_clv = clv.get("avg_clv", 0)
                beat_pct = clv.get("beat_closing_pct", 0)
                win_rate = clv.get("win_rate")
                n_games = clv.get("n_games", 0)
                stats_parts = [
                    f"<span style='color:#22c55e;font-weight:600'>{avg_clv:+.2%} CLV</span>",
                    f"<span style='font-weight:600'>{beat_pct:.1%} beat close</span>",
                ]
                if win_rate is not None:
                    stats_parts.append(f"<span style='font-weight:600'>{win_rate:.1%} SU</span>")
                stats_parts.append(f"{n_games} games")
                body += f"""
                    <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:12px">
                        <p style="color:#f1f5f9;font-size:13px;margin:0">{' | '.join(stats_parts)}</p>
                    </div>
                """

            if not bets:
                body += '<p style="color:#64748b;font-size:13px">No edges found today.</p>'
            else:
                # How many picks to show per tier
                max_picks = len(bets) if tier == "paid" else min(2, len(bets))
                shown_bets = bets[:max_picks]

                for bet in shown_bets:
                    edge = bet.get("edge")
                    edge_str = f"{edge:+.1%}" if edge else "?"
                    ml_str = _fmt_ml(bet.get("bet_ml"))
                    prob_str = _fmt_pct(bet.get("model_prob"))

                    # Edge strength indicator
                    if edge and edge >= 0.08:
                        strength = "🔥"
                    elif edge and edge >= 0.05:
                        strength = "✅"
                    else:
                        strength = "⚡"

                    body += f"""
                        <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:8px">
                            <p style="margin:0;font-size:14px;color:#f1f5f9">
                                {strength} <strong>{bet['bet_name']}</strong>
                                <span style="color:#94a3b8"> vs {bet['opp_name']}</span>
                            </p>
                            <p style="margin:4px 0 0 0;font-size:12px;color:#94a3b8">
                                ML: {ml_str} &nbsp;|&nbsp; Model: {prob_str} &nbsp;|&nbsp;
                                Edge: <span style="color:#22c55e;font-weight:600">{edge_str}</span>
                            </p>
                        </div>
                    """

                if tier == "free" and len(bets) > max_picks:
                    remaining = len(bets) - max_picks
                    body += f"""
                        <p style="color:#64748b;font-size:13px;margin-top:8px">
                            + {remaining} more pick{'s' if remaining > 1 else ''} available for Premium subscribers
                        </p>
                    """

            body += "</div>"

        # Upgrade CTA for free tier
        if tier == "free":
            body += f"""
                <div style="text-align:center;margin:24px 0">
                    <a href="{SITE_URL}/#subscribe" style="background:#22c55e;color:#0a1628;padding:10px 24px;border-radius:8px;font-weight:700;text-decoration:none;font-size:14px">Upgrade to Premium — $9/mo</a>
                </div>
            """

    footer = """
        </div>
        <div style="border-top:1px solid #1e293b;margin-top:24px;padding-top:16px;text-align:center">
            <p style="color:#475569;font-size:12px">Not financial or gambling advice.</p>
            <p style="color:#475569;font-size:12px"><a href="{unsub_url}" style="color:#64748b">Unsubscribe</a></p>
        </div>
    </div>
    """

    return subject, header + body + footer


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Verify cron secret
        auth = self.headers.get("authorization", "")
        if CRON_SECRET and auth != f"Bearer {CRON_SECRET}":
            return self._json(401, {"error": "Unauthorized"})

        if not RESEND_API_KEY:
            return self._json(500, {"error": "RESEND_API_KEY not configured"})

        import resend
        resend.api_key = RESEND_API_KEY

        results = {"free": 0, "paid": 0, "errors": []}

        for tier in ("paid", "free"):
            subscribers = get_active_subscribers(tier)
            if not subscribers:
                continue

            subject, html = _generate_picks_html(tier)

            for sub in subscribers:
                unsub_url = f"{SITE_URL}/api/unsubscribe?token={sub['unsubscribe_token']}"
                personalized_html = html.replace("{unsub_url}", unsub_url)
                try:
                    resend.Emails.send({
                        "from": RESEND_FROM,
                        "to": sub["email"],
                        "subject": subject,
                        "html": personalized_html,
                    })
                    results[tier] += 1
                except Exception as e:
                    results["errors"].append(f"{sub['email']}: {e}")

        return self._json(200, {"sent": results})

    def _json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

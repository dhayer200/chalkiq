"""POST /api/send_newsletter — Daily newsletter cron (protected by CRON_SECRET)."""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api._shared.db import get_active_subscribers

CRON_SECRET = os.environ.get("CRON_SECRET", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
SITE_URL = os.environ.get("SITE_URL", "https://chalkiq.com")


def _generate_picks_html(tier: str) -> tuple[str, str]:
    """Generate newsletter HTML content. Returns (subject, html_body)."""
    today = date.today()
    subject = f"ChalkIQ Picks — {today.strftime('%B %d, %Y')}"

    # Static header for now — will wire to live model data later
    header = f"""
    <div style="background:#0f172a;color:#f1f5f9;padding:32px;font-family:system-ui,sans-serif;max-width:600px;margin:0 auto">
        <div style="text-align:center;margin-bottom:24px">
            <span style="background:#22c55e;color:#0a1628;padding:4px 10px;border-radius:6px;font-weight:800;font-size:14px">C</span>
            <span style="font-weight:700;font-size:18px;margin-left:8px">ChalkIQ</span>
        </div>
        <h1 style="text-align:center;font-size:20px;margin-bottom:8px">{today.strftime('%A, %B %d')}</h1>
        <p style="text-align:center;color:#94a3b8;font-size:14px;margin-bottom:24px">Daily NCAA Basketball Picks</p>
        <div style="border-top:1px solid #1e293b;padding-top:20px">
    """

    if tier == "paid":
        body = """
            <p style="color:#22c55e;font-weight:600;font-size:14px;text-transform:uppercase;letter-spacing:1px">Premium Picks</p>
            <p style="color:#94a3b8;font-size:14px">Full slate with edges, CLV data, and line movement coming soon. Model is live and tracking.</p>
            <div style="background:#1e293b;border-radius:8px;padding:16px;margin:16px 0">
                <p style="color:#f1f5f9;font-size:14px;margin:0">Season stats: <span style="color:#22c55e;font-weight:600">+3.88% CLV</span> | <span style="font-weight:600">74.7% SU</span> | <span style="color:#22c55e;font-weight:600">57.3% beat close</span></p>
            </div>
            <p style="color:#64748b;font-size:13px">Full picks with book-by-book lines and injury impacts will be delivered once the tournament bracket is set.</p>
        """
    else:
        body = """
            <p style="color:#22c55e;font-weight:600;font-size:14px;text-transform:uppercase;letter-spacing:1px">Free Picks</p>
            <p style="color:#94a3b8;font-size:14px">Top picks and model performance. Upgrade to Premium for full slate, CLV data, and injury analysis.</p>
            <div style="background:#1e293b;border-radius:8px;padding:16px;margin:16px 0">
                <p style="color:#f1f5f9;font-size:14px;margin:0">Season stats: <span style="color:#22c55e;font-weight:600">+3.88% CLV</span> | <span style="font-weight:600">74.7% SU</span></p>
            </div>
            <div style="text-align:center;margin:24px 0">
                <a href="{site}/#subscribe" style="background:#22c55e;color:#0a1628;padding:10px 24px;border-radius:8px;font-weight:700;text-decoration:none;font-size:14px">Upgrade to Premium — $9/mo</a>
            </div>
        """.replace("{site}", SITE_URL)

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
                        "from": "ChalkIQ <picks@chalkiq.com>",
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

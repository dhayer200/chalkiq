"""POST /api/subscribe — Free newsletter signup with welcome email."""

from http.server import BaseHTTPRequestHandler
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api._shared.db import add_subscriber
from api._shared.email import RESEND_FROM

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
SITE_URL = os.environ.get("SITE_URL", "https://chalkiq.com")


def _send_welcome(email: str, unsub_token: str) -> None:
    """Send a welcome email via Resend. Best-effort, never blocks signup."""
    if not RESEND_API_KEY:
        return
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from": RESEND_FROM,
            "to": [email],
            "subject": "Welcome to ChalkIQ",
            "html": f"""
            <div style="font-family:monospace;max-width:560px;margin:0 auto;color:#d8dee9;background:#2e3440;padding:32px;border-radius:8px">
              <h2 style="color:#a3be8c;margin-top:0">Welcome to ChalkIQ</h2>
              <p>You're in. You'll receive our college sports newsletter with:</p>
              <ul>
                <li>Men's college basketball Elo edges and CLV tracking</li>
                <li>College football slates (starting fall 2026)</li>
                <li>Model vs market when we find real edge</li>
              </ul>
              <p style="color:#88c0d0">Stay sharp.</p>
              <hr style="border:none;border-top:1px solid #434c5e;margin:24px 0">
              <p style="font-size:11px;color:#4c566a">
                <a href="{SITE_URL}/api/unsubscribe?token={unsub_token}" style="color:#4c566a">Unsubscribe</a>
              </p>
            </div>
            """,
        })
    except Exception:
        pass  # Never fail signup because of email


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        except Exception:
            return self._json(400, {"error": "Invalid JSON"})

        email = (body.get("email") or "").strip().lower()
        if not email or not _EMAIL_RE.match(email):
            return self._json(400, {"error": "Invalid email address"})

        tier = body.get("tier", "free")
        if tier not in ("free", "paid"):
            tier = "free"

        result = add_subscriber(email, tier)

        # Send welcome email for new signups (not re-subs or existing)
        if result.get("ok") and not result.get("exists"):
            _send_welcome(email, result.get("token", ""))

        return self._json(200, result)

    def _json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

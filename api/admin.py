"""GET/POST /api/admin — Admin dashboard API.

Protected by ADMIN_SECRET env var. Supports:
  GET  ?action=stats         — subscriber counts + latest newsletter info
  GET  ?action=subscribers    — full subscriber list
  GET  ?action=preview&tier=  — newsletter email preview HTML
  GET  ?action=history        — past newsletter sends
  POST ?action=send-test      — send test email to a single address
  POST ?action=send           — send newsletter to all subscribers
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
from urllib.parse import urlparse, parse_qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api._shared.db import (
    get_active_subscribers,
    get_latest_newsletter_content,
)
from api.send_newsletter import _generate_picks_html

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
SITE_URL = os.environ.get("SITE_URL", "https://chalkiq.com")


def _check_auth(headers) -> bool:
    if not ADMIN_SECRET:
        return True  # No secret configured = local dev, allow all
    auth = headers.get("authorization", "")
    return auth == f"Bearer {ADMIN_SECRET}"


def _get_all_subscribers() -> list[dict]:
    """Get ALL subscribers (active and inactive) for admin view."""
    from api._shared.db import _use_postgres, _pg_conn, _sqlite_conn

    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, tier, active, subscribed_at, paid_at, unsubscribed_at "
            "FROM subscribers ORDER BY subscribed_at DESC"
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
    else:
        db = _sqlite_conn()
        rows = [
            dict(r) for r in db.execute(
                "SELECT id, email, tier, active, subscribed_at, paid_at, unsubscribed_at "
                "FROM subscribers ORDER BY subscribed_at DESC"
            ).fetchall()
        ]
        db.close()
    return rows


def _get_send_history() -> list[dict]:
    """Get newsletter send history."""
    from api._shared.db import _use_postgres, _pg_conn, _sqlite_conn

    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, sent_at, tier, subject, recipient_count "
            "FROM newsletter_sends ORDER BY sent_at DESC LIMIT 50"
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
    else:
        db = _sqlite_conn()
        rows = [
            dict(r) for r in db.execute(
                "SELECT id, sent_at, tier, subject, recipient_count "
                "FROM newsletter_sends ORDER BY sent_at DESC LIMIT 50"
            ).fetchall()
        ]
        db.close()
    return rows


def _log_send(tier: str, subject: str, count: int) -> None:
    """Log a newsletter send to the newsletter_sends table."""
    from api._shared.db import _use_postgres, _pg_conn, _sqlite_conn

    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO newsletter_sends (tier, subject, recipient_count) VALUES (%s, %s, %s)",
            (tier, subject, count),
        )
        conn.commit()
        cur.close()
        conn.close()
    else:
        db = _sqlite_conn()
        db.execute(
            "INSERT INTO newsletter_sends (tier, subject, recipient_count) VALUES (?, ?, ?)",
            (tier, subject, count),
        )
        db.commit()
        db.close()


def _get_stats() -> dict:
    all_subs = _get_all_subscribers()
    active = [s for s in all_subs if s.get("active") in (True, 1)]
    inactive = [s for s in all_subs if s.get("active") in (False, 0)]
    free = [s for s in active if s["tier"] == "free"]
    paid = [s for s in active if s["tier"] == "paid"]

    content = get_latest_newsletter_content()

    return {
        "total": len(all_subs),
        "active": len(active),
        "inactive": len(inactive),
        "free": len(free),
        "paid": len(paid),
        "latest_content_date": content.get("date") if content else None,
        "latest_sports": [
            {
                "label": s.get("label", s.get("division", "")),
                "n_upcoming": s.get("n_upcoming", 0),
                "n_bets": s.get("n_bets", 0),
            }
            for s in (content.get("sports", []) if content else [])
        ],
    }


def _inject_custom_message(html: str, message: str) -> str:
    """Inject a custom message block after the email header divider."""
    import html as html_mod
    escaped = html_mod.escape(message).replace("\n", "<br>")
    msg_block = (
        '<div style="background:linear-gradient(135deg,#1e293b,#0f172a);'
        "border:1px solid rgba(34,197,94,0.2);border-radius:10px;"
        'padding:16px 20px;margin-bottom:20px">'
        f'<p style="margin:0;font-size:14px;color:#f1f5f9;line-height:1.6">{escaped}</p>'
        "</div>"
    )
    return html.replace(
        'border-top:1px solid #1e293b;padding-top:20px">',
        'border-top:1px solid #1e293b;padding-top:20px">' + msg_block,
    )


def _send_to_recipients(
    recipients: list[dict],
    tier: str,
    custom_subject: str = "",
    custom_message: str = "",
) -> dict:
    """Send newsletter to a list of recipients. Returns results dict."""
    if not RESEND_API_KEY:
        return {"error": "RESEND_API_KEY not configured"}

    import resend
    resend.api_key = RESEND_API_KEY

    subject, html = _generate_picks_html(tier)
    if custom_subject:
        subject = custom_subject
    if custom_message:
        html = _inject_custom_message(html, custom_message)

    sent = 0
    errors = []

    for sub in recipients:
        token = sub.get("unsubscribe_token", "none")
        unsub_url = f"{SITE_URL}/api/unsubscribe?token={token}"
        personalized_html = html.replace("{unsub_url}", unsub_url)
        try:
            resend.Emails.send({
                "from": "ChalkIQ <onboarding@resend.dev>",
                "to": sub["email"],
                "subject": subject,
                "html": personalized_html,
            })
            sent += 1
        except Exception as e:
            errors.append(f"{sub['email']}: {e}")

    return {"sent": sent, "errors": errors, "subject": subject}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not _check_auth(self.headers):
            return self._json(401, {"error": "Unauthorized"})

        params = parse_qs(urlparse(self.path).query)
        action = params.get("action", ["stats"])[0]

        if action == "stats":
            return self._json(200, _get_stats())

        elif action == "subscribers":
            return self._json(200, {"subscribers": _get_all_subscribers()})

        elif action == "preview":
            tier = params.get("tier", ["paid"])[0]
            subject, html = _generate_picks_html(tier)
            return self._json(200, {"subject": subject, "html": html})

        elif action == "history":
            return self._json(200, {"sends": _get_send_history()})

        else:
            return self._json(400, {"error": f"Unknown action: {action}"})

    def do_POST(self):
        if not _check_auth(self.headers):
            return self._json(401, {"error": "Unauthorized"})

        params = parse_qs(urlparse(self.path).query)
        action = params.get("action", [""])[0]

        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0) or 0)))
        except Exception:
            body = {}

        if action == "send-test":
            email = body.get("email", "").strip()
            if not email:
                return self._json(400, {"error": "Email required"})
            tier = body.get("tier", "paid")
            result = _send_to_recipients(
                [{"email": email, "unsubscribe_token": "test-token"}],
                tier,
                custom_subject=body.get("custom_subject", ""),
                custom_message=body.get("custom_message", ""),
            )
            return self._json(200, result)

        elif action == "send":
            tier = body.get("tier")  # None = both tiers
            custom_subject = body.get("custom_subject", "")
            custom_message = body.get("custom_message", "")
            results = {"paid": 0, "free": 0, "errors": []}

            for t in (["paid", "free"] if not tier else [tier]):
                subs = get_active_subscribers(t)
                if not subs:
                    continue
                r = _send_to_recipients(subs, t, custom_subject, custom_message)
                results[t] = r.get("sent", 0)
                results["errors"].extend(r.get("errors", []))
                _log_send(t, r.get("subject", ""), r.get("sent", 0))

            return self._json(200, {"sent": results})

        else:
            return self._json(400, {"error": f"Unknown action: {action}"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

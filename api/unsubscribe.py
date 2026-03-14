"""GET /api/unsubscribe?token=UUID — One-click unsubscribe."""

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api._shared.db import unsubscribe


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        token = params.get("token", [""])[0]

        if not token:
            return self._html(400, "<h1>Missing token</h1>")

        success = unsubscribe(token)
        if success:
            return self._html(200, """
                <div style="text-align:center;padding:60px 20px;font-family:system-ui">
                    <h1 style="color:#22c55e">Unsubscribed</h1>
                    <p style="color:#94a3b8">You've been removed from ChalkIQ emails.</p>
                    <p style="color:#64748b;margin-top:20px"><a href="/" style="color:#22c55e">Back to ChalkIQ</a></p>
                </div>
            """)
        else:
            return self._html(200, """
                <div style="text-align:center;padding:60px 20px;font-family:system-ui">
                    <h1>Already unsubscribed</h1>
                    <p style="color:#94a3b8">This email is no longer on our list.</p>
                </div>
            """)

    def _html(self, status: int, body: str):
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = f"<!DOCTYPE html><html><head><title>ChalkIQ</title><meta name='viewport' content='width=device-width'></head><body style='background:#0f172a;color:#f1f5f9'>{body}</body></html>"
        self.wfile.write(html.encode())

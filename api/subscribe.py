"""POST /api/subscribe — Free newsletter signup."""

from http.server import BaseHTTPRequestHandler
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api._shared.db import add_subscriber

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


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

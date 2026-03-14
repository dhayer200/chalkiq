"""POST /api/checkout — Create Stripe Checkout session for paid newsletter."""

from http.server import BaseHTTPRequestHandler
import json
import os

import stripe

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
SITE_URL = os.environ.get("SITE_URL", "https://chalkiq.com")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        except Exception:
            return self._json(400, {"error": "Invalid JSON"})

        email = (body.get("email") or "").strip().lower()
        if not email:
            return self._json(400, {"error": "Email required"})

        if not stripe.api_key or not PRICE_ID:
            return self._json(500, {"error": "Stripe not configured"})

        try:
            session = stripe.checkout.Session.create(
                mode="subscription",
                customer_email=email,
                line_items=[{"price": PRICE_ID, "quantity": 1}],
                success_url=f"{SITE_URL}/?paid=1",
                cancel_url=f"{SITE_URL}/#subscribe",
            )
            return self._json(200, {"url": session.url})
        except stripe.StripeError as e:
            return self._json(500, {"error": str(e)})

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

"""POST /api/stripe_webhook — Handle Stripe payment events."""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
from pathlib import Path

import stripe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api._shared.db import upgrade_subscriber, downgrade_subscriber

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = self.rfile.read(int(self.headers.get("content-length", 0)))
        sig = self.headers.get("stripe-signature", "")

        if not WEBHOOK_SECRET:
            return self._json(500, {"error": "Webhook secret not configured"})

        try:
            event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
        except (ValueError, stripe.SignatureVerificationError):
            return self._json(400, {"error": "Invalid signature"})

        event_type = event["type"]

        if event_type == "checkout.session.completed":
            session = event["data"]["object"]
            email = session.get("customer_email", "")
            customer_id = session.get("customer", "")
            subscription_id = session.get("subscription", "")
            if email:
                upgrade_subscriber(email, customer_id, subscription_id)

        elif event_type == "customer.subscription.deleted":
            sub = event["data"]["object"]
            subscription_id = sub.get("id", "")
            if subscription_id:
                downgrade_subscriber(subscription_id)

        return self._json(200, {"received": True})

    def _json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

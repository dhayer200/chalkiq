"""Shared email configuration."""

from __future__ import annotations

import os

RESEND_FROM = os.environ.get(
    "RESEND_FROM",
    "ChalkIQ <onboarding@resend.dev>",
)

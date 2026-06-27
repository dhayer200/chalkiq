"""
Deliver briefings without a dedicated SMS API.

Methods (in order):
  1. Email-to-SMS via Resend → carrier gateway (works from GitHub Actions)
  2. macOS iMessage via osascript (local Mac cron only)
"""

from __future__ import annotations

import os
import platform
import subprocess

# US carrier email-to-SMS gateways (free, no Twilio)
_CARRIER_GATEWAYS = {
    "att": "txt.att.net",
    "tmobile": "tmomail.net",
    "verizon": "vtext.com",
    "sprint": "messaging.sprintpcs.com",
    "uscellular": "email.uscc.net",
}


def _sms_email() -> str | None:
    """Resolve phone → email gateway address."""
    direct = os.environ.get("SMS_EMAIL", "").strip()
    if direct:
        return direct

    phone = os.environ.get("SMS_BRIEFING_TO", "5128508472").strip()
    phone = "".join(c for c in phone if c.isdigit())
    if len(phone) == 10:
        phone = f"1{phone}"

    carrier = os.environ.get("SMS_CARRIER", "att").lower()
    gateway = _CARRIER_GATEWAYS.get(carrier)
    if not gateway:
        return None
    return f"{phone[-10:]}@{gateway}"


def send_via_email_sms(body: str) -> bool:
    """Send briefing as email to carrier SMS gateway using Resend."""
    to_addr = _sms_email()
    if not to_addr:
        print("  [sms] no SMS_EMAIL or SMS_BRIEFING_TO configured")
        return False

    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key:
        print("  [sms] RESEND_API_KEY not set")
        return False

    try:
        import resend
        resend.api_key = api_key
        from_addr = os.environ.get("RESEND_FROM", "ChalkIQ <onboarding@resend.dev>")
        resend.Emails.send({
            "from": from_addr,
            "to": [to_addr],
            "subject": "ChalkIQ",
            "text": body[:1500],  # SMS gateways truncate; keep concise
        })
        print(f"  [sms] sent via email gateway → {to_addr[:6]}***")
        return True
    except Exception as e:
        print(f"  [sms] email gateway failed: {e}")
        return False


def send_via_imessage(body: str, phone: str | None = None) -> bool:
    """Send via macOS Messages (local only, no API key)."""
    if platform.system() != "Darwin":
        return False
    if os.environ.get("IMESSAGE_BRIEFING", "").lower() not in ("1", "true", "yes"):
        return False

    phone = phone or os.environ.get("SMS_BRIEFING_TO", "5128508472")
    digits = "".join(c for c in phone if c.isdigit())
    if len(digits) == 10:
        target = f"+1{digits}"
    elif len(digits) == 11:
        target = f"+{digits}"
    else:
        target = phone

    # Escape for AppleScript
    safe_body = body.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{target}" of targetService
        send "{safe_body}" to targetBuddy
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=30)
        print(f"  [imessage] sent to {target}")
        return True
    except Exception as e:
        print(f"  [imessage] failed: {e}")
        return False


def send_briefing(body: str) -> None:
    """Try all available delivery methods."""
    sent = send_via_email_sms(body)
    if not sent:
        send_via_imessage(body)

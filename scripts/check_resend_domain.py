#!/usr/bin/env python3
"""Check Resend domain verification status for chalkiq.com."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM = os.environ.get("RESEND_FROM", "ChalkIQ <onboarding@resend.dev>")
DOMAIN = os.environ.get("RESEND_DOMAIN", "chalkiq.com")


def main() -> int:
    if not RESEND_API_KEY:
        print("ERROR: RESEND_API_KEY not set")
        return 1

    import resend
    resend.api_key = RESEND_API_KEY

    print(f"Configured FROM: {RESEND_FROM}")
    print(f"Checking domain: {DOMAIN}\n")

    try:
        domains = resend.Domains.list()
    except Exception as e:
        print(f"Failed to list domains: {e}")
        return 1

    items = domains.get("data", []) if isinstance(domains, dict) else getattr(domains, "data", []) or []

    if not items:
        print("No domains registered in Resend.")
        print(f"\nAdd {DOMAIN} at https://resend.com/domains")
        print("Then set DNS records Resend provides (DKIM, SPF, optional DMARC).")
        print(f"Once verified, set RESEND_FROM='ChalkIQ <newsletter@{DOMAIN}>' in Vercel env.")
        return 2

    found = False
    for d in items:
        name = d.get("name") if isinstance(d, dict) else getattr(d, "name", "")
        status = d.get("status") if isinstance(d, dict) else getattr(d, "status", "")
        region = d.get("region") if isinstance(d, dict) else getattr(d, "region", "")
        print(f"  {name}: status={status} region={region}")
        if name == DOMAIN:
            found = True
            if status == "verified":
                print(f"\n✓ {DOMAIN} is verified. Set RESEND_FROM=ChalkIQ <newsletter@{DOMAIN}>")
                return 0
            print(f"\n⚠ {DOMAIN} exists but status is '{status}' — finish DNS setup in Resend dashboard")
            return 3

    if not found:
        print(f"\n{DOMAIN} not found in Resend. Add it at https://resend.com/domains")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

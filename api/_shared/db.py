"""Subscriber database (SQLite for local dev, Turso for production)."""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path

_DB_PATH = os.environ.get("DATABASE_PATH", str(Path(__file__).resolve().parents[2] / "data" / "subscribers.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subscribers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    tier TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    unsubscribe_token TEXT UNIQUE NOT NULL,
    subscribed_at TEXT NOT NULL DEFAULT (datetime('now')),
    paid_at TEXT,
    unsubscribed_at TEXT,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS newsletter_sends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at TEXT NOT NULL DEFAULT (datetime('now')),
    tier TEXT NOT NULL,
    subject TEXT NOT NULL,
    recipient_count INTEGER NOT NULL,
    resend_batch_id TEXT
);
"""


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(_DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(_SCHEMA)
    return db


def add_subscriber(email: str, tier: str = "free") -> dict:
    db = get_db()
    token = str(uuid.uuid4())
    try:
        db.execute(
            "INSERT INTO subscribers (email, tier, unsubscribe_token) VALUES (?, ?, ?)",
            (email, tier, token),
        )
        db.commit()
        return {"ok": True, "token": token}
    except sqlite3.IntegrityError:
        # Already exists — reactivate if inactive
        row = db.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
        if row and not row["active"]:
            db.execute(
                "UPDATE subscribers SET active = 1, unsubscribed_at = NULL, tier = ? WHERE email = ?",
                (tier, email),
            )
            db.commit()
            return {"ok": True, "token": row["unsubscribe_token"], "reactivated": True}
        return {"ok": True, "exists": True, "token": row["unsubscribe_token"] if row else token}
    finally:
        db.close()


def upgrade_subscriber(email: str, stripe_customer_id: str, stripe_subscription_id: str) -> None:
    db = get_db()
    db.execute(
        """UPDATE subscribers
           SET tier = 'paid', stripe_customer_id = ?, stripe_subscription_id = ?,
               paid_at = datetime('now')
           WHERE email = ?""",
        (stripe_customer_id, stripe_subscription_id, email),
    )
    # If subscriber doesn't exist yet, create them as paid
    if db.execute("SELECT changes()").fetchone()[0] == 0:
        token = str(uuid.uuid4())
        db.execute(
            """INSERT INTO subscribers (email, tier, stripe_customer_id, stripe_subscription_id,
               unsubscribe_token, paid_at) VALUES (?, 'paid', ?, ?, ?, datetime('now'))""",
            (email, stripe_customer_id, stripe_subscription_id, token),
        )
    db.commit()
    db.close()


def downgrade_subscriber(stripe_subscription_id: str) -> None:
    db = get_db()
    db.execute(
        "UPDATE subscribers SET tier = 'free' WHERE stripe_subscription_id = ?",
        (stripe_subscription_id,),
    )
    db.commit()
    db.close()


def unsubscribe(token: str) -> bool:
    db = get_db()
    cur = db.execute(
        "UPDATE subscribers SET active = 0, unsubscribed_at = datetime('now') WHERE unsubscribe_token = ? AND active = 1",
        (token,),
    )
    db.commit()
    changed = cur.rowcount > 0
    db.close()
    return changed


def get_active_subscribers(tier: str | None = None) -> list[dict]:
    db = get_db()
    if tier:
        rows = db.execute(
            "SELECT email, tier, unsubscribe_token FROM subscribers WHERE active = 1 AND tier = ?",
            (tier,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT email, tier, unsubscribe_token FROM subscribers WHERE active = 1"
        ).fetchall()
    db.close()
    return [dict(r) for r in rows]

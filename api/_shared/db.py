"""Subscriber database — Neon Postgres in production, SQLite for local dev."""

from __future__ import annotations

import os
import uuid

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _use_postgres() -> bool:
    return DATABASE_URL.startswith("postgres")


# ── Postgres (Neon) ────────────────────────────────────────────────

def _pg_conn():
    import psycopg2
    import psycopg2.extras
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _pg_ensure_schema():
    conn = _pg_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL DEFAULT 'free',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            unsubscribe_token TEXT UNIQUE NOT NULL,
            subscribed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            paid_at TIMESTAMPTZ,
            unsubscribed_at TIMESTAMPTZ,
            active BOOLEAN NOT NULL DEFAULT TRUE
        );
        CREATE TABLE IF NOT EXISTS newsletter_sends (
            id SERIAL PRIMARY KEY,
            sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            tier TEXT NOT NULL,
            subject TEXT NOT NULL,
            recipient_count INTEGER NOT NULL,
            resend_batch_id TEXT
        );
        CREATE TABLE IF NOT EXISTS newsletter_content (
            id SERIAL PRIMARY KEY,
            generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            content JSONB NOT NULL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


# ── SQLite (local dev) ─────────────────────────────────────────────

import sqlite3
from pathlib import Path

_DB_PATH = os.environ.get("DATABASE_PATH", str(Path(__file__).resolve().parents[2] / "data" / "subscribers.db"))

_SQLITE_SCHEMA = """
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

CREATE TABLE IF NOT EXISTS newsletter_content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL DEFAULT (datetime('now')),
    content TEXT NOT NULL
);
"""


def _sqlite_conn() -> sqlite3.Connection:
    db = sqlite3.connect(_DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(_SQLITE_SCHEMA)
    return db


# ── Public API ─────────────────────────────────────────────────────

def add_subscriber(email: str, tier: str = "free") -> dict:
    token = str(uuid.uuid4())

    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO subscribers (email, tier, unsubscribe_token) VALUES (%s, %s, %s)",
                (email, tier, token),
            )
            conn.commit()
            return {"ok": True, "token": token}
        except Exception:
            conn.rollback()
            cur.execute("SELECT * FROM subscribers WHERE email = %s", (email,))
            row = cur.fetchone()
            if row and not row["active"]:
                cur.execute(
                    "UPDATE subscribers SET active = TRUE, unsubscribed_at = NULL, tier = %s WHERE email = %s",
                    (tier, email),
                )
                conn.commit()
                return {"ok": True, "token": row["unsubscribe_token"], "reactivated": True}
            return {"ok": True, "exists": True, "token": row["unsubscribe_token"] if row else token}
        finally:
            cur.close()
            conn.close()
    else:
        db = _sqlite_conn()
        try:
            db.execute(
                "INSERT INTO subscribers (email, tier, unsubscribe_token) VALUES (?, ?, ?)",
                (email, tier, token),
            )
            db.commit()
            return {"ok": True, "token": token}
        except sqlite3.IntegrityError:
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
    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """UPDATE subscribers
               SET tier = 'paid', stripe_customer_id = %s, stripe_subscription_id = %s,
                   paid_at = NOW()
               WHERE email = %s""",
            (stripe_customer_id, stripe_subscription_id, email),
        )
        if cur.rowcount == 0:
            token = str(uuid.uuid4())
            cur.execute(
                """INSERT INTO subscribers (email, tier, stripe_customer_id, stripe_subscription_id,
                   unsubscribe_token, paid_at) VALUES (%s, 'paid', %s, %s, %s, NOW())""",
                (email, stripe_customer_id, stripe_subscription_id, token),
            )
        conn.commit()
        cur.close()
        conn.close()
    else:
        db = _sqlite_conn()
        db.execute(
            """UPDATE subscribers
               SET tier = 'paid', stripe_customer_id = ?, stripe_subscription_id = ?,
                   paid_at = datetime('now')
               WHERE email = ?""",
            (stripe_customer_id, stripe_subscription_id, email),
        )
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
    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE subscribers SET tier = 'free' WHERE stripe_subscription_id = %s",
            (stripe_subscription_id,),
        )
        conn.commit()
        cur.close()
        conn.close()
    else:
        db = _sqlite_conn()
        db.execute(
            "UPDATE subscribers SET tier = 'free' WHERE stripe_subscription_id = ?",
            (stripe_subscription_id,),
        )
        db.commit()
        db.close()


def unsubscribe(token: str) -> bool:
    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE subscribers SET active = FALSE, unsubscribed_at = NOW() WHERE unsubscribe_token = %s AND active = TRUE",
            (token,),
        )
        conn.commit()
        changed = cur.rowcount > 0
        cur.close()
        conn.close()
        return changed
    else:
        db = _sqlite_conn()
        cur = db.execute(
            "UPDATE subscribers SET active = 0, unsubscribed_at = datetime('now') WHERE unsubscribe_token = ? AND active = 1",
            (token,),
        )
        db.commit()
        changed = cur.rowcount > 0
        db.close()
        return changed


def get_active_subscribers(tier: str | None = None) -> list[dict]:
    if _use_postgres():
        conn = _pg_conn()
        cur = conn.cursor()
        if tier:
            cur.execute(
                "SELECT email, tier, unsubscribe_token FROM subscribers WHERE active = TRUE AND tier = %s",
                (tier,),
            )
        else:
            cur.execute("SELECT email, tier, unsubscribe_token FROM subscribers WHERE active = TRUE")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    else:
        db = _sqlite_conn()
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


def save_newsletter_content(data: dict) -> None:
    """Save pre-generated newsletter content (called by generate_newsletter_data.py)."""
    import json
    content_json = json.dumps(data)

    if _use_postgres():
        _pg_ensure_schema()
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO newsletter_content (content) VALUES (%s)",
            (content_json,),
        )
        conn.commit()
        cur.close()
        conn.close()
    else:
        db = _sqlite_conn()
        db.execute(
            "INSERT INTO newsletter_content (content) VALUES (?)",
            (content_json,),
        )
        db.commit()
        db.close()


def get_latest_newsletter_content() -> dict | None:
    """Get the most recent newsletter content."""
    import json

    if _use_postgres():
        _pg_ensure_schema()
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT content FROM newsletter_content ORDER BY generated_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            c = row["content"]
            return c if isinstance(c, dict) else json.loads(c)
        return None
    else:
        db = _sqlite_conn()
        row = db.execute(
            "SELECT content FROM newsletter_content ORDER BY generated_at DESC LIMIT 1"
        ).fetchone()
        db.close()
        if row:
            return json.loads(row["content"])
        return None


import os
import sqlite3
import hashlib
from datetime import datetime

import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp53_app.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password_hash TEXT  NOT NULL,
            created_at  TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            filename    TEXT    NOT NULL,
            upload_date TEXT    NOT NULL,
            results_dir TEXT    NOT NULL,
            num_variants INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def create_user(username: str, password: str) -> bool:
    """Create a new user. Returns True on success, False if username exists."""
    try:
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        conn = _get_conn()
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, pw_hash.decode("utf-8"), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    if bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        return dict(row)
    return None


def save_upload(user_id: int, filename: str, results_dir: str, num_variants: int) -> int:
    """Record an upload. Returns the upload ID."""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO uploads (user_id, filename, upload_date, results_dir, num_variants) VALUES (?, ?, ?, ?, ?)",
        (user_id, filename, datetime.now().isoformat(), results_dir, num_variants),
    )
    upload_id = cur.lastrowid
    conn.commit()
    conn.close()
    return upload_id


def get_user_uploads(user_id: int) -> list[dict]:
    """Get all uploads for a user, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM uploads WHERE user_id = ? ORDER BY upload_date DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_upload(upload_id: int) -> str | None:
    """Delete an upload and its files. Returns the results_dir if found, else None."""
    conn = _get_conn()
    row = conn.execute("SELECT results_dir FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if row is None:
        conn.close()
        return None
    results_dir = row["results_dir"]
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()
    return results_dir

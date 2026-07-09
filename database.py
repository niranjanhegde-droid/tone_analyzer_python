"""
database.py
------------
Lightweight SQLite data access layer -- no ORM, plain sqlite3, so the
project has zero extra dependencies beyond Flask + scikit-learn.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "instance", "app.db")


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            transcript TEXT NOT NULL,
            overall_label TEXT NOT NULL,
            health_score REAL NOT NULL,
            tone_counts_json TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------
def create_user(name, email, password_hash):
    conn = get_db()
    conn.execute(
        "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (name, email, password_hash, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_by_email(email):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def save_analysis(user_id, title, transcript, overall_label, health_score,
                   tone_counts_json, report_json):
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO analyses
           (user_id, title, transcript, overall_label, health_score,
            tone_counts_json, report_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, title, transcript, overall_label, health_score,
         tone_counts_json, report_json, datetime.utcnow().isoformat()),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_analyses_for_user(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


def get_analysis(analysis_id, user_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user_id),
    ).fetchone()
    conn.close()
    return row


def delete_analysis(analysis_id, user_id):
    conn = get_db()
    conn.execute(
        "DELETE FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user_id),
    )
    conn.commit()
    conn.close()

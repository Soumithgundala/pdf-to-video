"""
Local SQLite database layer — replaces Supabase.
Uses Python's built-in sqlite3; no extra dependencies required.
The DB file lives at backend/manga_pipeline.db and is created automatically.
"""
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import config

logger = logging.getLogger(__name__)

DB_PATH = Path(config.DB_PATH)


def get_db() -> sqlite3.Connection:
    """
    Return a SQLite connection with row_factory set so rows behave like dicts.
    check_same_thread=False is intentional: FastAPI background tasks run in
    the same process/thread pool, and we use short-lived connections per call.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read/write
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id              TEXT PRIMARY KEY,
                status          TEXT NOT NULL DEFAULT 'pending',
                pdf_filename    TEXT,
                pdf_path        TEXT,
                total_pages     INTEGER,
                total_panels    INTEGER,
                error_message   TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                completed_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS video_parts (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id           TEXT NOT NULL REFERENCES jobs(id),
                part_number      INTEGER NOT NULL,
                script           TEXT,
                selected_panels  TEXT,
                audio_path       TEXT,
                audio_duration_ms INTEGER,
                video_path       TEXT,
                status           TEXT NOT NULL DEFAULT 'pending',
                created_at       TEXT NOT NULL
            );
        """)
    logger.info(f"Database initialized at {DB_PATH}")


# ---------------------------------------------------------------------------
# Job helpers
# ---------------------------------------------------------------------------

def insert_job(job_id: str, pdf_filename: str, pdf_path: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs (id, status, pdf_filename, pdf_path, created_at, updated_at)
               VALUES (?, 'pending', ?, ?, ?, ?)""",
            (job_id, pdf_filename, pdf_path, now, now),
        )


def get_job(job_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def update_job(job_id: str, **fields) -> None:
    """Update arbitrary columns on a job row."""
    if not fields:
        return
    fields["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    with get_db() as conn:
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)


# ---------------------------------------------------------------------------
# Video-part helpers
# ---------------------------------------------------------------------------

def insert_video_part(
    job_id: str,
    part_number: int,
    script: str,
    selected_panels: list,
    audio_path: str,
    audio_duration_ms: int,
    video_path: str,
) -> None:
    import json
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO video_parts
               (job_id, part_number, script, selected_panels,
                audio_path, audio_duration_ms, video_path, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?)""",
            (
                job_id, part_number, script,
                json.dumps(selected_panels),
                audio_path, audio_duration_ms, video_path, now,
            ),
        )


def get_video_parts(job_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM video_parts WHERE job_id = ? ORDER BY part_number",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]

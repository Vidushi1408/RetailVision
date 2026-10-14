import os
import sqlite3

DB_PATH = os.getenv("RETAILVISION_DB", "store_analytics.db")


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db():
    conn = connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS customer_traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            people_count INTEGER
        );
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, fps REAL, duration_sec REAL, created TEXT
        );
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER, track_id INTEGER,
            first_sec REAL, last_sec REAL, dwell_sec REAL,
            path_length REAL, avg_speed REAL, stationary_ratio REAL,
            zone TEXT, role TEXT, carrying TEXT, segment TEXT
        );
        CREATE TABLE IF NOT EXISTS crossings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER, line_name TEXT, direction TEXT, sec REAL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, kind TEXT, params TEXT, status TEXT,
            created REAL, started REAL, finished REAL,
            result TEXT, error TEXT, upload_path TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE TABLE IF NOT EXISTS count_lines (
            name TEXT PRIMARY KEY,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL, created TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_crossings_video ON crossings(video_id);
        CREATE INDEX IF NOT EXISTS idx_traffic_timestamp ON customer_traffic(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tracks_video_id ON tracks(video_id);
        CREATE INDEX IF NOT EXISTS idx_tracks_segment ON tracks(segment);
    """)
    _ensure_column(conn, "count_lines", "camera", "TEXT DEFAULT 'default'")
    _ensure_column(conn, "videos", "camera", "TEXT DEFAULT 'default'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lines_camera ON count_lines(camera)")
    conn.commit()
    conn.close()


def _ensure_column(conn, table, column, decl):
    """Tiny forward-only migration: add a column to a table created by an older version."""
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


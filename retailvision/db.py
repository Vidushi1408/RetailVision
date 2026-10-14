import sqlite3

DB_PATH = "store_analytics.db"


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
        CREATE TABLE IF NOT EXISTS count_lines (
            name TEXT PRIMARY KEY,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL, created TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_crossings_video ON crossings(video_id);
        CREATE INDEX IF NOT EXISTS idx_traffic_timestamp ON customer_traffic(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tracks_video_id ON tracks(video_id);
        CREATE INDEX IF NOT EXISTS idx_tracks_segment ON tracks(segment);
    """)
    conn.commit()
    conn.close()


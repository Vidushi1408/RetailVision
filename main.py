import os
import shutil
import tempfile

import threading
from contextlib import asynccontextmanager

import datetime
import math
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from retailvision import jobs
from retailvision.db import connect, init_db
from retailvision.forecast import forecast
from retailvision.pipeline import analyze

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("PRELOAD_MODELS", "yolo,llm") not in ("", "none"):
        from retailvision.preload import preload
        threading.Thread(target=preload, daemon=True).start()  # don't delay startup
    yield


app = FastAPI(title="RetailVision", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
init_db()


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")



async def _save_upload(file: UploadFile) -> str:
    """Stream an upload to a temp file, enforcing type and size limits. Caller must delete it."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use one of {sorted(ALLOWED_EXT)}.")
    limit, written = MAX_UPLOAD_MB * 1024 * 1024, 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB limit.")
            tmp.write(chunk)
    return tmp.name


@app.post("/frame/")
async def extract_frame(file: UploadFile = File(...)):
    """Return one JPEG frame from a clip so a counting line can be drawn on it.
    Uses OpenCV (same decoder as the analysis), so it works for codecs a browser cannot play."""
    import cv2
    from fastapi.responses import Response
    path = await _save_upload(file)
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(422, "Could not read a frame from this video.")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    finally:
        os.unlink(path)


@app.post("/analyze_video/")
async def analyze_video(file: UploadFile = File(...), use_clip: bool = True,
                        background: bool = False):
    """Track people through the video, classify, segment, store.

    background=false (default): block and return the summary.
    background=true: return {"job_id"} immediately; poll GET /jobs/{job_id}.
    """
    path = await _save_upload(file)
    if background:
        job_id = jobs.submit(analyze, path, file.filename, use_clip=use_clip,
                             cleanup_path=path)
        return {"job_id": job_id, "status": "queued"}
    try:
        return analyze(path, file.filename, use_clip=use_clip)
    finally:
        os.unlink(path)


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return {k: job[k] for k in ("id", "status", "result", "error")}


@app.get("/customer_traffic/")
def customer_traffic():
    conn = connect()
    rows = conn.execute("SELECT * FROM customer_traffic ORDER BY timestamp DESC").fetchall()
    conn.close()
    return {"customer_traffic": [dict(r) for r in rows]}


@app.get("/segments/")
def segments():
    conn = connect()
    rows = conn.execute("""SELECT segment, role, carrying, COUNT(*) n, ROUND(AVG(dwell_sec),1) avg_dwell
                           FROM tracks GROUP BY segment, role, carrying""").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/forecast/")
def get_forecast(hours: int = 6):
    return forecast(steps=hours)


@app.get("/overview/")
def get_overview():
    conn = connect()
    video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    total_tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    avg_dwell_row = conn.execute("SELECT ROUND(AVG(dwell_sec), 1) FROM tracks").fetchone()
    avg_dwell = avg_dwell_row[0] if avg_dwell_row and avg_dwell_row[0] is not None else 0.0
    
    engaged_row = conn.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN segment = 'engaged-shopper' THEN 1 ELSE 0 END) / MAX(COUNT(*), 1), 1)
        FROM tracks
    """).fetchone()
    engaged_pct = engaged_row[0] if engaged_row and engaged_row[0] is not None else 0.0

    zone_rows = conn.execute("SELECT zone, COUNT(*) count FROM tracks WHERE zone IS NOT NULL GROUP BY zone").fetchall()
    zones = {r["zone"]: r["count"] for r in zone_rows}

    occ = conn.execute("SELECT ROUND(AVG(people_count), 1), MAX(people_count) FROM customer_traffic").fetchone()
    cross = {r["direction"]: r["n"] for r in conn.execute(
        "SELECT direction, COUNT(*) n FROM crossings GROUP BY direction").fetchall()}
    recent_videos = [dict(r) for r in conn.execute("SELECT * FROM videos ORDER BY id DESC LIMIT 5").fetchall()]
    traffic_points = conn.execute("SELECT COUNT(*) FROM customer_traffic").fetchone()[0]
    conn.close()

    return {
        "video_count": video_count,
        "total_tracks": total_tracks,
        "avg_dwell_sec": avg_dwell,
        "engaged_pct": engaged_pct,
        "zones": zones,
        "traffic_points": traffic_points,
        "avg_occupancy": occ[0],
        "peak_occupancy": occ[1],
        "crossings_in": cross.get("in", 0),
        "crossings_out": cross.get("out", 0),
        "recent_videos": recent_videos
    }


@app.get("/crossings/")
def crossings(video_id: int | None = None):
    """Directed line-crossing totals per counting line (optionally for one video)."""
    conn = connect()
    q = "SELECT video_id, line_name, direction, COUNT(*) n FROM crossings"
    args = ()
    if video_id is not None:
        q, args = q + " WHERE video_id = ?", (video_id,)
    rows = conn.execute(q + " GROUP BY video_id, line_name, direction ORDER BY video_id", args).fetchall()
    conn.close()
    out = {}
    for r in rows:
        out.setdefault(r["video_id"], {}).setdefault(r["line_name"], {"in": 0, "out": 0})[r["direction"]] = r["n"]
    return out


MAX_LINES = 8
LINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,31}$")


class LineIn(BaseModel):
    name: str
    x1: float
    y1: float
    x2: float
    y2: float


def _line_rows(conn):
    return [dict(r) for r in conn.execute(
        "SELECT name, x1, y1, x2, y2 FROM count_lines ORDER BY created").fetchall()]


@app.get("/lines/")
def list_lines():
    """Saved counting lines. Coordinates are fractions of the frame (0-1), y down.
    "in" = moving from the left-hand side of (x1,y1)->(x2,y2) to the right-hand side."""
    conn = connect()
    try:
        return {"lines": _line_rows(conn), "using_saved": bool(_line_rows(conn))}
    finally:
        conn.close()


@app.put("/lines/")
def save_line(line: LineIn):
    """Create or replace a counting line by name."""
    if not LINE_NAME.match(line.name.strip()):
        raise HTTPException(422, "Name must be 1-32 characters: letters, digits, space, _ or -.")
    coords = (line.x1, line.y1, line.x2, line.y2)
    if not all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in coords):
        raise HTTPException(422, "Coordinates must be between 0 and 1.")
    if math.hypot(line.x2 - line.x1, line.y2 - line.y1) < 0.05:
        raise HTTPException(422, "Line is too short; draw at least 5% of the frame width.")
    name = line.name.strip()
    conn = connect()
    try:
        exists = conn.execute("SELECT 1 FROM count_lines WHERE name = ?", (name,)).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM count_lines").fetchone()[0]
        if not exists and count >= MAX_LINES:
            raise HTTPException(409, f"At most {MAX_LINES} lines; delete one first.")
        conn.execute("""INSERT INTO count_lines (name, x1, y1, x2, y2, created) VALUES (?,?,?,?,?,?)
                        ON CONFLICT(name) DO UPDATE SET x1=excluded.x1, y1=excluded.y1,
                        x2=excluded.x2, y2=excluded.y2""",
                     (name, *coords, datetime.datetime.now().isoformat()))
        conn.commit()
        return {"lines": _line_rows(conn)}
    finally:
        conn.close()


@app.delete("/lines/{name}")
def delete_line(name: str):
    conn = connect()
    try:
        cur = conn.execute("DELETE FROM count_lines WHERE name = ?", (name,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, "No such line.")
        return {"lines": _line_rows(conn)}
    finally:
        conn.close()


@app.get("/ask/")
def ask(q: str):
    from retailvision.llm_analyst import ask as ask_llm
    return ask_llm(q)


import os
import shutil
import tempfile

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
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



@app.post("/analyze_video/")
async def analyze_video(file: UploadFile = File(...), use_clip: bool = True,
                        background: bool = False):
    """Track people through the video, classify, segment, store.

    background=false (default): block and return the summary.
    background=true: return {"job_id"} immediately; poll GET /jobs/{job_id}.
    """
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
    if background:
        job_id = jobs.submit(analyze, tmp.name, file.filename, use_clip=use_clip,
                             cleanup_path=tmp.name)
        return {"job_id": job_id, "status": "queued"}
    try:
        return analyze(tmp.name, file.filename, use_clip=use_clip)
    finally:
        os.unlink(tmp.name)


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


@app.get("/ask/")
def ask(q: str):
    from retailvision.llm_analyst import ask as ask_llm
    return ask_llm(q)


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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from retailvision import auth, jobs, privacy, streams
from retailvision.db import connect, init_db
from retailvision.forecast import forecast
from retailvision.pipeline import analyze

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")  # background uploads live here until their job ends
MAX_QUESTION_CHARS = 500
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "0"))  # 0 = keep analytics until wiped
CAMERA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,31}$")
ask_limiter = auth.RateLimiter(int(os.getenv("ASK_PER_MINUTE", "20")))


def analyze_job(path, filename, use_clip=True, camera="default"):
    return analyze(path, filename, use_clip=use_clip, camera=camera)


def stream_job(name, seconds, camera="default"):
    path = streams.record(name, seconds)
    try:
        return analyze(path, f"stream:{name}", use_clip=False, camera=camera)
    finally:
        if os.path.exists(path):
            os.unlink(path)


jobs.register("analyze", analyze_job)
jobs.register("stream", stream_job)


def _valid_camera(camera: str) -> str:
    camera = (camera or "default").strip()
    if not CAMERA_NAME.match(camera):
        raise HTTPException(422, "Camera name must be 1-32 characters: letters, digits, space, _ or -.")
    return camera


@asynccontextmanager
async def lifespan(_app):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    jobs.resume(UPLOAD_DIR)  # re-queue analyses interrupted by a restart
    if RETENTION_DAYS > 0:
        privacy.purge_older_than(RETENTION_DAYS)
    if os.getenv("PRELOAD_MODELS", "yolo,llm") not in ("", "none"):
        from retailvision.preload import preload
        threading.Thread(target=preload, daemon=True).start()  # don't delay startup
    yield


app = FastAPI(title="RetailVision", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
init_db()


@app.middleware("http")
async def require_key(request: Request, call_next):
    if auth.is_public(request.url.path) or auth.is_authorized(request):
        return await call_next(request)
    return JSONResponse({"detail": "API key required."}, status_code=401)


class LoginIn(BaseModel):
    key: str


@app.post("/login")
def login(body: LoginIn):
    """Exchange the shared key for an HttpOnly cookie (the dashboard calls this after a 401)."""
    if not auth.configured_key():
        return {"auth": "disabled"}
    if not auth.key_matches(body.key):
        raise HTTPException(401, "Wrong key.")
    resp = JSONResponse({"auth": "ok"})
    resp.set_cookie(auth.COOKIE, body.key, httponly=True, samesite="strict", max_age=12 * 3600)
    return resp


@app.get("/health")
def health():
    from retailvision import llm_analyst
    return {"status": "ok", "auth_required": bool(auth.configured_key()),
            "analyst_ready": llm_analyst.is_ready()}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")



async def _save_upload(file: UploadFile, dest_dir: str | None = None) -> str:
    """Stream an upload to disk, enforcing type and size limits. Caller must delete it."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use one of {sorted(ALLOWED_EXT)}.")
    limit, written = MAX_UPLOAD_MB * 1024 * 1024, 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=dest_dir) as tmp:
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
                        background: bool = False, camera: str = "default"):
    """Track people through the video, classify, segment, store.

    background=false (default): block and return the summary.
    background=true: return {"job_id"} immediately; poll GET /jobs/{job_id}.
    """
    camera = _valid_camera(camera)
    if background:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        path = await _save_upload(file, UPLOAD_DIR)
        try:
            job_id = jobs.submit("analyze", {"path": os.path.abspath(path), "filename": file.filename,
                                             "use_clip": use_clip, "camera": camera}, upload_path=os.path.abspath(path))
        except jobs.QueueFull as e:
            os.unlink(path)
            raise HTTPException(429, str(e))
        return {"job_id": job_id, "status": "queued"}
    path = await _save_upload(file)
    try:
        return analyze(path, file.filename, use_clip=use_clip, camera=camera)
    finally:
        os.unlink(path)


@app.get("/jobs/")
def list_jobs(limit: int = 20):
    return {"jobs": jobs.recent(min(max(limit, 1), 100))}


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
    camera: str = "default"
    x1: float
    y1: float
    x2: float
    y2: float


def _line_rows(conn):
    return [dict(r) for r in conn.execute(
        "SELECT name, camera, x1, y1, x2, y2 FROM count_lines ORDER BY camera, created").fetchall()]


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
        camera = _valid_camera(line.camera)
        conn.execute("""INSERT INTO count_lines (name, x1, y1, x2, y2, created, camera) VALUES (?,?,?,?,?,?,?)
                        ON CONFLICT(name) DO UPDATE SET x1=excluded.x1, y1=excluded.y1,
                        x2=excluded.x2, y2=excluded.y2, camera=excluded.camera""",
                     (name, *coords, datetime.datetime.now().isoformat(), camera))
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


@app.get("/privacy/")
def privacy_info():
    return {
        "retention_days": RETENTION_DAYS or None,
        "stores": ["people counts per second", "per-person track statistics (no images)",
                   "line-crossing events", "counting-line settings"],
        "does_not_store": ["uploaded video (deleted when its job ends)", "face or person images",
                           "stream URLs in responses"],
    }


@app.post("/privacy/purge")
def privacy_purge(days: int):
    if days < 0:
        raise HTTPException(422, "days must be 0 or more.")
    return {"deleted": privacy.purge_older_than(days)}


@app.delete("/privacy/data")
def privacy_wipe(confirm: str = ""):
    if confirm != "yes":
        raise HTTPException(400, "Add ?confirm=yes to delete all analytics data.")
    return {"deleted": privacy.wipe_all()}


@app.get("/streams/")
def list_streams():
    """Names of configured camera sources (STREAM_SOURCES). URLs are never returned."""
    return {"streams": sorted(streams.configured()), "max_seconds": streams.MAX_SECONDS}


@app.post("/analyze_stream/")
def analyze_stream(name: str, seconds: int = 30, camera: str = "default"):
    """Record `seconds` from a configured source, then analyse it as a background job."""
    if name not in streams.configured():
        raise HTTPException(404, "Unknown stream. Configure it in STREAM_SOURCES.")
    if not 1 <= seconds <= streams.MAX_SECONDS:
        raise HTTPException(422, f"seconds must be between 1 and {streams.MAX_SECONDS}.")
    try:
        job_id = jobs.submit("stream", {"name": name, "seconds": seconds, "camera": _valid_camera(camera)})
    except jobs.QueueFull as e:
        raise HTTPException(429, str(e))
    return {"job_id": job_id, "status": "queued"}


@app.get("/ask/")
def ask(q: str, request: Request = None):
    if len(q) > MAX_QUESTION_CHARS:
        raise HTTPException(422, f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
    client = request.client.host if request is not None and request.client else "local"
    if not ask_limiter.allow(client):
        raise HTTPException(429, "Too many questions; wait a minute and try again.")
    from retailvision.llm_analyst import ask as ask_llm
    return ask_llm(q)


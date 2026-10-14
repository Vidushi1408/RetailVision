# RetailVision

Turns store-camera video into analytics, fully locally (no external API):
YOLOv8 + ByteTrack people tracking, CLIP zero-shot classification, KMeans shopper
segments, an LSTM footfall forecaster, and a local LLM (Qwen2.5-1.5B) that answers
questions in plain English over the stored data.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
First use downloads the YOLOv8s and Qwen weights (a few GB; CLIP only if you enable attributes) once; afterwards it runs offline. Weights are not stored in git. Python 3.14 is pinned in `.python-version`.

## Run
```bash
.venv/bin/uvicorn main:app --reload     # dashboard at http://127.0.0.1:8000
.venv/bin/python -m unittest test_retailvision test_api test_counting test_security -v
```
The SQLite database (`store_analytics.db`) is created automatically and is not tracked in git.
Put sample videos in the project root; `*.mp4` and `*.avi` are git-ignored.

## Layout
- `main.py` FastAPI app and endpoints
- `retailvision/` tracker, classifier, forecast, llm_analyst, pipeline, db, device
- `templates/`, `static/` dashboard
- `PLAN.md` analysis and team implementation plan

## Team rules
Work on a branch per workstream; the schema in `retailvision/db.py` is the shared contract.

## Evaluation data (Phase 1)
Real shopping-mall CCTV with per-frame ground-truth counts: the CUHK **Mall dataset**
(research use only, cite Chen et al. BMVC 2012 / Loy et al. ICCV 2013). Not stored in git.
```bash
mkdir -p data && cd data
curl -LO https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip && unzip mall_dataset.zip
```
Then build `data/mall_counts.npy` from `mall_dataset/mall_gt.mat` (`count` array) and run
`python evaluation/count_eval.py yolov8s.pt 1280 0.15`.

## Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `PRELOAD_MODELS` | `yolo,llm` | Models loaded at server start in a background thread. `none` disables (tests/CI). |
| `ENABLE_CLIP_ATTRIBUTES` | `0` | `1` enables CLIP staff/customer + carrying labels. Off because they were not accurate on 640x480 CCTV. |
| `COUNT_LINES` | one horizontal mid-frame line | Fallback only: JSON `{"name": [x1,y1,x2,y2]}` in 0-1 frame coordinates. Lines saved in the dashboard take priority. |
| `MAX_UPLOAD_MB` | `500` | Upload size limit (HTTP 413 above it). |
| `RETAILVISION_API_KEY` | empty (open) | If set, every route except the page, `/static/`, `/login` and `/health` needs this key (`X-API-Key` header or the cookie from `POST /login`). The dashboard asks for it on a 401. |
| `UPLOAD_DIR` | `uploads` | Where background uploads wait for their job. Survives restarts. |
| `MAX_QUEUED_JOBS` | `5` | Queued + running analyses allowed at once (HTTP 429 above it). |
| `ASK_PER_MINUTE` | `20` | Analyst questions per client per minute (HTTP 429 above it). |
| `LOCAL_LLM` | `Qwen/Qwen2.5-1.5B-Instruct` | Local model used by `/ask/`. |

## API notes
- `POST /analyze_video/?background=true` returns `{"job_id"}` at once; poll `GET /jobs/{job_id}`
  (`queued` / `running` / `done` / `error`). Without `background` it blocks and returns the summary.
- `GET /crossings/?video_id=` returns in/out totals per counting line.
- Counting lines: draw them in the dashboard (Counting Lines card). API: `GET /lines/`, `PUT /lines/`
  `{name,x1,y1,x2,y2}` (0-1 fractions, max 8), `DELETE /lines/{name}`. "in" is the right-hand side
  of A to B. `POST /frame/` returns a JPEG frame from a clip (used to draw on).
- Jobs are stored in SQLite: queued or interrupted analyses are re-queued on restart; `GET /jobs/`
  lists recent ones. A job whose upload was lost is marked as an error.
- The analyst can only read `customer_traffic`, `videos`, `tracks` and `crossings` (SQLite
  authorizer allow-list), is read-only, stops after 5 s and returns at most 50 rows.
- Security notes: the shared key is one secret for everyone, not per-user accounts; the cookie is
  HttpOnly and SameSite=Strict but only `Secure` if you add HTTPS in front; rate limits live in
  process memory (reset on restart).
- Accepted video types: mp4, avi, mov, mkv, webm.

## Accuracy checks (need `data/`, see above)
```bash
PYTHONPATH=. python evaluation/count_eval.py yolov8s.pt 1280 0.15   # detection MAE
PYTHONPATH=. python evaluation/forecast_eval.py 10 1                # LSTM vs baselines
PYTHONPATH=. python evaluation/llm_eval.py data/eval.db             # text-to-SQL accuracy
```

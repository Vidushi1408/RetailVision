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
First use downloads YOLOv8n, CLIP and Qwen weights (a few GB) once; afterwards it runs offline.

## Run
```bash
.venv/bin/uvicorn main:app --reload     # dashboard at http://127.0.0.1:8000
.venv/bin/python -m unittest test_retailvision -v
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
| `PRELOAD_MODELS` | `yolo,clip,llm` | Models loaded at server start in a background thread. `none` disables (tests/CI). |
| `MAX_UPLOAD_MB` | `500` | Upload size limit (HTTP 413 above it). |
| `LOCAL_LLM` | `Qwen/Qwen2.5-1.5B-Instruct` | Local model used by `/ask/`. |

## API notes
- `POST /analyze_video/?background=true` returns `{"job_id"}` at once; poll `GET /jobs/{job_id}`
  (`queued` / `running` / `done` / `error`). Without `background` it blocks and returns the summary.
- Accepted video types: mp4, avi, mov, mkv, webm.

## Accuracy checks (need `data/`, see above)
```bash
PYTHONPATH=. python evaluation/count_eval.py yolov8s.pt 1280 0.15   # detection MAE
PYTHONPATH=. python evaluation/forecast_eval.py 10 1                # LSTM vs baselines
PYTHONPATH=. python evaluation/llm_eval.py data/eval.db             # text-to-SQL accuracy
```

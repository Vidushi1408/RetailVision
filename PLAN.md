# RetailVision: analysis and implementation plan

## 1. What the project is
Store-camera video goes in; people are detected and tracked (YOLOv8 + ByteTrack), classified
(CLIP zero-shot + KMeans behaviour segments), stored in SQLite, forecast (PyTorch LSTM) and
queried in plain English through a local LLM (Qwen2.5-1.5B). FastAPI backend, HTML/JS dashboard.
Goal: fully local, no paid or external API.

## 2. Measured findings (run on the two sample videos)

| Area | Finding | Severity |
|---|---|---|
| Tracking | `sample_cctv_vtest.avi`: 91 raw track IDs, 65 kept, but never more than 12 people in view at once. IDs are fragmenting, so unique-visitor counts are inflated roughly 3-5x. | High |
| Classification | 22 of 22 tracks labelled mostly "employee" + "a bag". CLIP prompts are biased, so the output is close to constant and carries no information. | High |
| Segmentation | KMeans always forces 3 clusters, and cluster names are hard-coded. A 1.2 s track can be called "browser". Segments are relative to one video, not meaningful. | Medium |
| Forecast | Under 9 hourly points it returns a sine-wave placeholder. Timestamps are "now + second", so every upload lands in the same hour; real history never builds up. Tests pass only on this placeholder. | High |
| Local-only rule | `llm_analyst.py` silently calls Google Gemini when `GEMINI_API_KEY` is set. | Medium |
| Speed | Tracking is fast (5 s for 80 s of video on Apple MPS). CLIP + load adds ~9 s. Qwen has not been timed. `/analyze_video/` is synchronous and will time out on long videos. | Medium |
| Data/repo hygiene | No `.gitignore`; `store_analytics.db`, `.venv`, weights, videos, `__pycache__` are untracked or tracked. The DB changes on every run, so five people will conflict. Unpinned requirements. | High |
| Tests | 4 smoke tests, no accuracy checks, no tracking/classification/LLM tests. | Medium |
| Dead code | `App.js` (React, superseded by the dashboard), `download_yolo.py`. | Low |
| Security | `/ask/` runs model-written SQL. Guarded by a SELECT-only regex plus `query_only`; acceptable but should add a table allow-list and row/time limits. Uploads go to temp files with no size limit. | Low |

## 3. Plan

### Phase 0: Repo hygiene (Day 1, one person, blocks everyone else)
- Add `.gitignore` (`.venv`, `__pycache__`, `*.db`, `*.pt`, `*.mp4`, `*.avi`, `store_lstm.pt`).
- Untrack `store_analytics.db`; DB is created by `init_db()`.
- Remove the Gemini path from `llm_analyst.py`; delete `App.js` and `download_yolo.py`.
- Pin `requirements.txt` (`pip freeze` of the working venv). Rename `Static` to `static`.
- Write `README.md` (setup, run, tests). Protect `main`; work on branches.
- Exit: fresh clone runs `uvicorn main:app` and the tests.

### Phase 1: Ground truth (Days 1-3, whole team)
- Pick 3-5 short store videos (pedestrian or CCTV). Hand-count unique people per video and
  hand-label 50 people (staff/customer, carrying) in a CSV under `data/labels/`.
- Without this, nothing below can be called "accurate".

### Phase 2: Workstreams (Weeks 1-3, parallel, one owner each)

| # | Owner area | Files | Work | Done when |
|---|---|---|---|---|
| A | Tracking | `tracker.py` | Stride 1-2; try `yolov8s/m`; tune ByteTrack (`track_buffer`, `match_thresh`); add min-track-length + merge fragments by position/time gap; optional appearance re-ID. | Unique count within ±15% of manual count on all videos. |
| B | Classification | `classifier.py` | Evaluate CLIP on the labelled set; improve prompts and drop the staff/customer guess when confidence is low (`unknown`); fix label text ("a bag" → "bag"); try uniform-colour or linear-probe on CLIP features trained on our labels. | ≥80% accuracy on labelled set, or the field is shown as `unknown`. |
| C | Behaviour + forecast | `classifier.segment_shoppers`, `forecast.py` | Rule/threshold segments with real units (seconds, zone changes) instead of forced KMeans; keep KMeans as an exploration view. Forecast: real timestamps, hourly aggregation, compare LSTM vs naive/seasonal baseline, report MAE, refuse to show a forecast without data. | LSTM beats baseline on held-out hours, or the UI says "not enough data". |
| D | Local LLM | `llm_analyst.py` | Few-shot SQL examples for this schema; table/column allow-list; time-boxed queries; answer from rows only; test 30 questions (SQL correct?). Optional fine-tune with LoRA on question-to-SQL pairs. Time first-request latency; preload at startup. | ≥80% of the 30 questions give a correct query. |
| E | Backend + UI + CI | `main.py`, `pipeline.py`, `static/`, tests | Background job queue for uploads (job id, progress endpoint); upload size/type limit; real video timestamps (`recorded_at` field); dashboard shows data-quality notes; GitHub Actions running tests. | A 10-minute video upload does not block or time out. |

### Phase 3: Integration (Week 4)
- Merge branches; re-run all accuracy checks on the whole pipeline.
- Add the regression tests: unique-count tolerance, classification accuracy, SQL correctness,
  forecast beats baseline, API contract tests.
- Demo script: upload video, view dashboard, ask three questions, see the forecast.

### Phase 4: Stretch (only if time remains)
- Live RTSP/webcam stream; heat-map overlay of zones; configurable store layout zones
  (named areas instead of a 3x3 grid); entry/exit line counting; Dockerfile.

## 4. Shared rules
- Database schema in `db.py` is the contract; schema changes need a team note.
- Everyone tests with the same labelled videos.
- A metric is only reported if it has a number from the labelled set.
- One PR per workstream; the owner of the file reviews.

## 5. Risks
| Risk | Mitigation |
|---|---|
| No real store footage | Use public pedestrian datasets (MOT17/MOT20 clips); record a small in-person video. |
| Small local LLM writes wrong SQL | Few-shot examples, allow-list, show the SQL, fall back to canned queries. |
| Not enough time-series data for LSTM | Simulate clearly marked data for demos; otherwise show baseline and say so. |
| Laptops without GPU are slow | Stride 3, nano model, background jobs; run CLIP only on the best crop per track (already done). |

## 6. Phase 1 results (real store footage: CUHK Mall dataset, 2000 CCTV frames, 640x480, <2 fps, 13-53 people per frame, mean 31)

Per-frame count accuracy of person detection vs ground truth (200 evenly spaced frames, Apple MPS):

| Model | imgsz | conf | MAE (people) | Bias | MAPE |
|---|---|---|---|---|---|
| yolov8n (current pipeline default) | 640 | 0.25 | 13.2 | -13.2 | 42.2% |
| yolov8n | 640 | 0.15 | 9.5 | -9.5 | 30.1% |
| yolov8n | 1280 | 0.25 | 9.0 | -9.0 | 29.0% |
| yolov8n | 1280 | 0.15 | 4.4 | -4.1 | 14.3% |
| yolov8s | 640 | 0.25 | 12.0 | -12.0 | 37.9% |
| yolov8s | 1280 | 0.15 | **4.1** | -3.2 | **12.9%** |

Findings
- The pipeline as shipped (n, 640, conf 0.25, stride 3) misses ~42% of people on crowded CCTV.
  Higher input size and lower confidence are the biggest wins; 8s over 8n adds little. Still a
  small undercount (-3 people) from occlusion, which is expected.
- Tracking on a 400-frame (200 s) clip: 389 raw track IDs at stride 1 and 109 at stride 3, median
  track length 6 points, 56 IDs under 3 points, while ~26 people are in view at any time. IDs
  fragment heavily; "unique visitors" from this tracker is unreliable and stride 3 on a 2 fps feed
  discards two thirds of the data. Mean per-second count is 11.4 vs ground truth 25.9.
- Ground truth gives counts per frame only, not identities, so unique-visitor accuracy still needs
  a hand-count on a short clip (see next steps).

Actions this triggers
1. Tracker: expose `imgsz`, `conf`, `stride` as parameters; default to 1280 / 0.15 / stride 1 for
   low-fps feeds (stride relative to fps, not fixed 3). (Workstream A)
2. Add `evaluation/count_eval.py` to CI as a regression check: MAE must stay under 6.
3. Hand-count unique people on a 60-second clip and label ~50 people (staff/customer, carrying)
   from the Mall frames to evaluate CLIP (Workstream B). Mall frames are low resolution, so
   expect CLIP attributes to be weak; be prepared to report "unknown" often.

## 7. Phase 1 follow-up: tracker and classifier changes (done)
- `tracker.py`: defaults now `yolov8s`, imgsz 1280, conf 0.15, stride from fps (~5 processed
  frames/s, 1 for low-fps CCTV), ByteTrack config `retailvision/bytetrack_retail.yaml`
  (buffer 60, match 0.9), plus `merge_fragments` (gap ≤4 s, distance ≤0.12 of frame).
- Mall clip (200 s, ~26 people in view): raw IDs 389 -> 352 (config) -> 267 (merged); tracks under
  3 points 56 -> 14; median track length 6 -> 13; mean people/sec 11.4 -> 17.6 (ground truth ~25.9).
  Unique-visitor count is still an estimate, not verified, since the dataset has no identities.
- CLIP on the 60 largest Mall person crops (~90x45 px): 58/60 "employee" at 84% mean confidence,
  i.e. confidently wrong. Classifier now returns `unknown` for crops under 128 px tall.
  Staff/customer needs either higher-resolution footage or a trained classifier (Workstream B).
- Labelling sheet for the team: `data/labeling/crops/*.jpg` + `data/labeling/labels.csv` (60 rows).

## 8. Workstreams C and D results

### Forecast (`evaluation/forecast_eval.py`, Mall ground-truth counts, walk-forward, last 30% test)
| Series | Last value MAE | Window mean MAE | Train mean MAE | LSTM MAE |
|---|---|---|---|---|
| per frame, 1 step | 1.95 | 2.78 | 4.94 | 1.94 |
| 10-frame bins, 1 step | **3.69** | 4.43 | 4.38 | 5.33 |
| 10-frame bins, 5 steps | 5.05 | 5.15 | **4.34** | 6.31 |
| 25-frame bins, 1 step | **3.52** | 4.15 | 3.90 | 4.80 |

- The LSTM ties "predict the last value" at best and loses when data is small (<200 points).
  The Mall footage is only ~17 minutes with no daily cycle, so this is expected: an LSTM needs
  days of hourly history with a real pattern to be worth it.
- `forecast.py` now backtests (train 80%, test 20%) on every call and uses the LSTM only if it
  beats the last-value baseline; the response includes `method` and `backtest_mae`. On a
  synthetic 14-day hourly series with a daily cycle the LSTM wins (MAE 2.97 vs 5.37).
- Needed for a real result: several days of real hourly footfall from one store.

### Local LLM text-to-SQL (`evaluation/llm_eval.py`, Qwen2.5-1.5B, 30 questions, gold SQL, test DB from the Mall clip)
| Prompt | Strict | Lenient (extra columns ok) |
|---|---|---|
| schema only | 13/30 (43%) | - |
| + rules + 9 few-shot examples (none from the test set) | 23/30 (77%) | **27/30 (90%)** |

- Remaining real errors: "how many people in total" used `customer_traffic`; "busiest moment"
  returned MAX(timestamp); one invented a zone filter for "passer-by".
- Latency: ~1-2 s per question after the model is loaded; first load downloads ~3 GB and took
  several minutes. Preload the model at server start.
- Target (≥80% correct) is met on the lenient measure; keep adding failing questions as examples
  and re-run before every change to the prompt.

## 9. Workstream E results (done)
- `/analyze_video/?background=true` + `/jobs/{id}` (single-worker queue, `retailvision/jobs.py`);
  dashboard now polls instead of holding one long request.
- Upload validation: extension allow-list (400) and streaming size limit `MAX_UPLOAD_MB` (413).
- YOLO model cached per process; ByteTrack state reset per video (regression test).
- Models preloaded at startup in a background thread (`PRELOAD_MODELS`).
- `test_api.py`: 6 tests (validation, limit, sync + background analysis, 404, tracker reset);
  with `test_retailvision.py` 10 tests pass locally.
- CI: `.github/workflows/ci.yml` runs both test files. Not yet run on GitHub; requirements were
  pinned from a Python 3.14 venv, so the workflow may need a different Python version.
- Remaining: job state is in memory (lost on restart; fine for a single-process demo).

## 10. Workstream B results: classification on low-resolution CCTV

Data: the 60 largest merged tracks of `data/mall_clip.mp4`, best crop per track (87-188 px tall,
median 130), hand-labelled by one annotator from contact sheets (`evaluation/bag_labels.py`).
54 usable labels: 20 carrying a bag, 34 not. Majority-class accuracy 0.63. Small and noisy;
treat as a rough check.

| Method (carrying a bag) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority class ("no bag") | 0.63 | - | 0 | 0 |
| CLIP zero-shot, current wording | 0.54 | 0.33 | 0.25 | 0.29 |
| CLIP zero-shot, ensemble prompts | 0.48 | 0.40 | 0.85 | 0.55 |
| CLIP linear probe (LOO-CV, 54 samples) | 0.52-0.59 | 0.40-0.45 | 0.45-0.60 | 0.45-0.49 |

- No CLIP variant beats the trivial baseline on accuracy; the size gate does not help
  (crops >=128 px: accuracy 0.48 vs 0.76 majority).
- Staff vs customer cannot be evaluated: there is no visible uniform in this footage, and the
  earlier zero-shot output (58 of 60 "employee" at 84% confidence) is clearly wrong.
- Decision: CLIP attributes are OFF by default (`ENABLE_CLIP_ATTRIBUTES=1` to enable); the
  pipeline stores `unknown` rather than a wrong label. Revisit with >=256 px crops or a model
  trained on a person-attribute dataset (PA-100K / Market-1501; images are behind Google
  Drive/Baidu links, so not fetched here).

### Re-identification and identity fragmentation
- Same-person similarity check on 9 crops I believed were one person: CLIP image embeddings
  AUC 0.46 (chance); HSV colour histograms AUC 0.86.
- CORRECTION: those 9 tracks overlap in time over 100+ s at different positions, so they are not
  one person; my visual identification was wrong or they are several similarly dressed people.
  The AUC numbers above rest on that guess and are therefore unreliable.
- Track quality on the mall clip (200 s, ~26 people in view):

| Variant | Tracks | Low coverage (<0.5) | Spans > 60 s |
|---|---|---|---|
| Raw ByteTrack | 352 | 39% | 24 |
| Merge, no guard | 267 | 44% | 32 |
| Merge + coverage guard (default) | 296 | 43% | 25 |
| + appearance gate (`use_appearance=True`, off) | 291 | 44% | 25 |

- Coverage guard added to `merge_fragments`: refuses merges that would be observed in <50% of
  their span (chains of different people inflate dwell).
- Honest conclusion: the unique-visitor count is somewhere between ~230 and ~350 for this clip
  and cannot be pinned down without identity ground truth (e.g. MOT17/MOT20 annotations or a
  hand-tracked 30 s window). Prefer robust metrics for reporting: average occupancy (people
  present), peak occupancy, and line-crossing counts at an entrance.

## 11. Identity-free metrics: occupancy and line crossings (done)

Because unique visitors cannot be verified (section 10), the reportable metrics are now:

| Metric | How | Where |
|---|---|---|
| Occupancy (people visible) | raw detections per processed frame, averaged per second | `customer_traffic.people_count`, summary `occupancy.mean/peak` |
| Line crossings in/out | per-track sign change across a configurable segment, with a 0.02 margin against jitter, counted on **raw** tracks | `crossings` table, `GET /crossings/`, summary `crossings` |

Lines are set with `COUNT_LINES='{"door":[0.0,0.5,1.0,0.5]}'` (normalised x1,y1,x2,y2); "in" =
moving from the left-hand side of A->B to the right-hand side (image coords, y down).

Bug found and fixed on the way: `model.track()` returns only *confirmed* tracks, so at ~2 fps it
reported 13.7 people per frame where plain detection finds 23.9 (truth ~25.4). The tracker now
runs detection and ByteTrack association separately (one inference), and a fresh tracker is
created per video.

| Mall clip (200 s) | Before | After |
|---|---|---|
| Occupancy MAE (people, per second) | 12.2 (bias -12.2, 47%) | **3.8** (bias -3.0, 14%) |
| Mean occupancy (truth 25.9) | 13.7 | 22.8 |
| Peak occupancy (truth 49) | 26 | 42 |
| Raw / merged tracks | 352 / 296 | 352 / 296 (unchanged) |

- Crossing counts have no ground truth in this dataset; correctness is covered by 8 synthetic
  tests (direction, jitter, zig-zag, walking around a short line). On the mall clip a mid-frame
  horizontal line gave 111 in / 66 out; treat as unverified until counted by hand on a clip.
- Text-to-SQL re-check after adding the `crossings` table to the schema prompt: 28/30 lenient.
- Known limit: a person whose track ID breaks exactly at the line is missed (undercount).

## 12. Real store footage: two supplied videos (2026-10-14)

Two phone-forwarded CCTV clips, both H.264, run through the live app (YOLOv8s, 1280 px, conf 0.15).

| Clip | Scene | Length / fps / size | People tracked (est.) | Occupancy mean / peak | Crossings in / out |
|---|---|---|---|---|---|
| 13.48.04 | Pharmacy interior, high angle, counter + aisles | 110.9 s, 13.1, 634x360 | 39 | 12.0 / 16 | 9 / 9 (default mid-frame line, **not meaningful**: no door in view) |
| 13.52.34 | Store entrance, automatic doors, people walk toward the camera | 121.0 s, 6.0, 480x360 | 164 | 8.9 / 16 | 41 / 16 on a line at the doorway |

How it was checked (no hand-labelled ground truth exists for these clips):
- **Occupancy:** detections overlaid on 8 sampled frames and compared with what is visible. Boxes
  sit on the visible people, including partly hidden ones. Caveats: people seen *through the glass
  doors* and staff at the counter are counted; groups standing close together are sometimes one box.
- **Crossings:** every counted crossing was cropped and reviewed by eye (`scripts/review_crossings.py`).
  Directions matched what the footage shows (backs to the camera = out, faces = in).
- **Defect found:** 87 raw entrance crossings, but 18 of them were a track flipping across the line
  within 3 s (box jitter in the crowd at the door, merged group boxes, ID switches). One person
  could produce in-out-in within a second.
- **Fix:** cancel a same-track reversal within `REVERSAL_WINDOW` (default 3 s). Entrance count went
  87 -> 57 (41 in / 16 out), in line with ~50-57 distinct crossings seen in the crops. Variants tried
  offline on the same tracks: feet-point reference was worse (121 events), margin 0.05 gave 51.
  Visual review of the retained events: mostly distinct people; a few duplicates remain (the same
  person under two IDs), so treat 57 as an upper estimate (roughly 90% precision by eye).

What this shows and does not show
- Crossing counting needs a camera that actually looks at a door: the default mid-frame line on the
  pharmacy clip produced 9 in / 9 out that mean nothing. The analyst answered "50 entered, 25 left"
  by summing both clips, which is arithmetically right and practically misleading.
- 164 "people tracked" in 2 minutes at the entrance is an overcount from ID fragmentation in the
  crowd; the real number of distinct people is probably well under 100. It remains an estimate.
- Still unmeasured: precision/recall against a hand count of the whole clip.

## 13. Analyst fix: "SQL is not being executed" (user report, 2026-10-14)

Root causes found by reproducing against the live server and reading its log:
1. **Failing SQL shown as executed.** Both dashboard suggestion chips made the small model invent
   columns (`customer`; `zone` on `customer_traffic`), so SQLite returned "no such column". The
   chat still printed "Executed SQL: ...", whenever any SQL existed.
2. **Startup race.** The background model preload published the tokenizer before the model existed,
   so a question arriving mid-load skipped loading and crashed with `KeyError('model')` (HTTP 500,
   nothing shown). Reproduced: 5 of 6 concurrent first callers crashed under the old logic.
3. **Silently wrong answers.** "How many people were tracked in total?" used `SUM(people_count)`
   (1,336, which is people-seconds); "customers" produced `role = 'customer'`, which matches
   nothing because role labelling is off.

Changes
- `sql_with_repair`: on a SQL error, tell the model the error and which table owns the missing
  column, one retry; if a single-table query names a column that lives in exactly one other table,
  switch the table deterministically (`direction` -> `crossings`).
- `role` filters are dropped when attribute labelling is off (they can only return zero rows).
- Prompt rules: people totals are `COUNT(*) FROM tracks`, never `SUM(people_count)`; zone traffic
  means rows per zone in `tracks`; no `customer`/`traffic` columns.
- Model loading is lock-guarded and publishes `ready` only when complete; `/health` reports
  `analyst_ready`.
- API returns `executed` and `attempts`; the chat shows "Executed SQL" only when it ran (with the
  result rows), and "Not executed" with the query otherwise.

Benchmark (`evaluation/llm_eval.py`, now 36 questions; 6 were added from this report)
| | Strict | Lenient (extra columns ok) |
|---|---|---|
| Before | 23/36 (64%) | 27/36 (75%) |
| After | 31/36 (86%) | 34/36 (94%) |

Caveat: the 6 added questions drove some of the fixes (rules, role filter, table swap), so the
number is optimistic for unseen wording. Remaining misses: "traffic records" resolved to `tracks`
instead of `customer_traffic` (ambiguous wording) and a spurious zone filter on "passer-by" path
length. 10 new unit tests (scripted LLM) and 1 browser test cover the repair path, the honest
labels, and the concurrent-load race.

## 14. Local completion pass (2026-10-14)

Closed the items that did not need new footage.

- **No arbitrary default line.** A camera with no saved counting line now reports no crossings
  (summary shows "No line set"); occupancy, dwell and people are still reported. Reason: the old
  mid-frame default produced "9 in / 9 out" on the pharmacy interior, which is meaningless but
  looked real, and the analyst then summed it with the real doorway counts.
- **Analyst guards** (all deterministic, each with unit tests): drop role filters when labelling is
  off; drop zone literals the question never mentions; add the behaviour-group filter when the
  question names exactly one group and the query ignores it; drop predicates on impossible enum
  values (`segment = 'shoppers'`); switch a single-table query to the table that owns a missing
  column; one error-aware model retry.
- **Benchmark honesty.** Two sets in `evaluation/llm_eval.py`:

| Set | Questions | Strict | Lenient | Note |
|---|---|---|---|---|
| Tuned (prompt and guards were fitted to these) | 36 | 33 (92%) | 36 (100%) | optimistic |
| Held-out, first run, before the enum guard | 12 | 9 (75%) | 11 (92%) | the unbiased estimate |
| Held-out, after the enum guard | 12 | 10 (83%) | 12 (100%) | no longer fully held out |

  Expect roughly 90% lenient on new wording from a 1.5B model, not 100%. The remaining strict
  misses return an extra column (zone plus its count when only the zone was asked), which is a
  correct answer in a different shape.
- **Tests: 72**, in 8 modules, all passing locally in several module orders.

Known limits that remain (unchanged, need data or hardware): unique-visitor accuracy, crossing
recall against a full hand count, staff/customer and bag labels, forecast benefit, continuous
live streaming, `v1.0` tag.

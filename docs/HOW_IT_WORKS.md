# RetailVision: how it works, algorithm by algorithm

This document is written from the source code, not from memory. File names point to where each
idea lives. Numbers quoted are measured results from `PLAN.md` (sections 5-15) and are labelled
with how they were obtained.

## 1. What the project does

RetailVision turns store-camera video into numbers a manager can use:

- **How many people are in view** over time (occupancy).
- **How many people cross a line you draw** (for example a doorway), in and out.
- **How long people stay, where they stand, and how they move** (dwell time, zone heat map,
  behaviour groups).
- **A forecast of footfall**, once enough hours of history exist.
- **A plain-English question box** that answers from the stored numbers using a small language
  model running on the same machine.

Everything runs locally: no cloud AI service is called, and uploaded video is deleted after
analysis. The one exception is the browser page, which loads Chart.js and Google Fonts from the
internet (see section 9).

## 2. The whole pipeline in one picture

```
 video file / camera window
        |
        v
 [1] YOLOv8s detector ──► boxes of people, every processed frame
        |                       |
        |                       └─► [2] occupancy = number of boxes (per second)
        v
 [3] ByteTrack ──► one track (path) per person, over time
        |
        ├─► [4] line-crossing geometry ──► in / out events on your lines
        ├─► [5] fragment merging ──► fewer, longer tracks
        ├─► [6] trajectory features ──► dwell, speed, path, stillness
        │         └─► [7] KMeans ──► passer-by / browser / engaged-shopper
        └─► [8] CLIP (optional) ──► carrying a bag? (large people only)
        |
        v
 SQLite database (videos, tracks, crossings, customer_traffic, jobs, count_lines)
        |
        ├─► [9] LSTM / baselines ──► footfall forecast
        ├─► [10] local LLM (Qwen) + SQL guard ──► answers to questions
        └─► FastAPI endpoints ──► dashboard (HTML + JavaScript)
```

## 3. Each algorithm: what, where, how, why

### 3.1 YOLOv8s: person detection
- **Where:** `retailvision/tracker.py`, `track_video()`, the call `model.predict(frame, classes=[0],
  imgsz=1280, conf=0.15)`.
- **What it is:** a convolutional neural network that looks at one image and outputs rectangles
  with a confidence score for each object. Class 0 of its 80 classes is "person". It is
  single-stage and anchor-free: one forward pass produces all boxes, then non-maximum
  suppression removes duplicate boxes.
- **How we use it:** pretrained weights (trained on the COCO dataset) are used as they are. We did
  not fine-tune it. We chose the "s" (small) size, 1280-pixel input and a low confidence
  threshold of 0.15, because in the first experiment on a public mall dataset the default
  (nano model, 640 px, 0.25) missed about 4 in 10 people.
- **Why this one:** fast enough to run on a laptop GPU, accurate enough, one file to load, no
  training needed. **Measured:** error of 4.1 people per frame (13%) against 13.2 (42%) with the
  defaults, on 200 frames of the CUHK Mall dataset.
- **Limits:** people hidden behind others, tiny people far from the camera, and people seen
  through glass are still detected or missed unpredictably. It does not know identity.

### 3.2 ByteTrack: turning detections into tracks
- **Where:** `tracker.py`, `BYTETracker(...).update(det, frame)`; settings in
  `retailvision/bytetrack_retail.yaml`.
- **What it is:** a tracker that links the boxes of one frame to the tracks of previous frames.
  For each existing track a **Kalman filter** predicts where the box should be now (it assumes
  roughly constant velocity). Predicted boxes and new detections are compared by **IoU**
  (overlap), and the **Hungarian algorithm** finds the best one-to-one assignment.
- **What makes ByteTrack special:** matching happens in two rounds. First, confident detections
  (score above `track_high_thresh`) are matched to tracks. Then the *leftover tracks* are matched
  against the *low-confidence* detections (between `track_low_thresh` and the high threshold).
  Those weak boxes are often a person who is partly hidden, so this keeps tracks alive through
  occlusion instead of throwing the weak boxes away.
- **Our settings:** `track_buffer: 60` (a lost track survives 60 *processed* frames, which is about 10 to 12 seconds at the supplied clips' rates), `match_thresh: 0.9`
  (accept loose matches because people move far between frames at low fps),
  `new_track_thresh: 0.30` (a new ID needs a confident box).
- **Why this one:** it uses only box geometry, so it is fast and needs no extra model. DeepSORT
  and BoT-SORT add a re-identification network that looks at appearance; that helps in crowds but
  costs time and a second model, and our own test showed generic image embeddings were not good
  at telling these people apart (see 3.5).
- **Important implementation detail:** we run detection and association separately. Ultralytics'
  convenience call `model.track()` only returns *confirmed* tracks, which at about 2 fps hid
  roughly 40% of people. Calling `predict()` and feeding the boxes to `BYTETracker` ourselves
  made the occupancy count correct (error 12.2 down to 3.8 people).
- **Limits:** when two people cross paths or a group walks together, the ID can jump to the wrong
  person (an *ID switch*) or split into several IDs (*fragmentation*). This is the main reason
  "people tracked" is only an estimate.

### 3.3 Occupancy: how many people are visible
- **Where:** `tracker.py` stores `len(det)` for every processed frame; `pipeline.py` averages it
  per second and writes it to `customer_traffic.people_count`.
- **How:** it counts *detections*, not confirmed tracks, so a person who has not been tracked yet
  is still counted. People seen through the glass doors and staff at the counter are counted too.
- **Measured:** about 14% error against ground truth on the mall clip (3.8 people, mean 22.8 vs
  25.9 real).

### 3.4 Line crossing: counting entrances (custom algorithm)
- **Where:** `tracker.py`, `line_crossings()`; lines are saved by the user in the dashboard
  (table `count_lines`) and used by `pipeline.count_lines()`.
- **How it works, step by step:**
  1. A line is two points A and B in fractions of the frame (0 to 1).
  2. For each track point (x, y) compute the **signed distance** to the line using the cross
     product. Positive means the right-hand side of A to B; negative the left-hand side.
  3. A crossing is recorded only when the track goes from beyond a **margin** (0.02 of the frame)
     on one side to beyond the margin on the other. The margin stops a person standing on the
     line from being counted again and again.
  4. The crossing point must lie within the segment (a small tolerance at the ends), so walking
     *around* a short line does not count.
  5. **Reversal cancelling:** if the same track crosses one way and then back within 3 seconds,
     both events are dropped. A person does not really cross a line twice in 3 seconds; it is
     jitter or an ID switch.
- **Direction:** "in" means moving from the left side of A to B to the right side, in image
  coordinates (y points down). For a horizontal line drawn left to right, "in" is downward.
- **Why custom:** it is a few lines of geometry, easy to test, and every rule came from something
  that went wrong on real footage.
- **Measured (one 2-minute entrance clip, my hand labels of 102 candidates):** precision about
  70%, recall about 71%, and a reported total of 57 against about 52 real crossers.
- **Limits:** a group is one box, so individuals inside it are undercounted; duplicates remain
  when one person gets two IDs.

### 3.5 Fragment merging: stitching broken tracks (custom heuristic)
- **Where:** `tracker.py`, `merge_fragments()`; it decides `unique_people` and dwell time.
- **How:** tracks are sorted by start time. A new track is attached to the *nearest* earlier
  track end if it starts within 4 seconds and within 0.12 of the frame, with no time overlap.
  A **coverage guard** refuses a merge that would make the person visible for less than half of
  their time span (that pattern means different people were glued together).
- **Optional appearance rule (off):** an HSV colour histogram of torso and legs, compared with
  cosine similarity, allows longer gaps. We tested it: on a hand-picked example it separated the
  same person from others better than generic CLIP image embeddings (AUC 0.86 vs 0.46), but the
  example was wrong in the end (see PLAN.md), so it stays off.
- **Why:** it is cheap and improves track length; honest limit: without identity ground truth we
  cannot prove it is correct, so the count stays labelled an estimate.

### 3.6 Trajectory features and zones
- **Where:** `tracker.py`, `track_features()` and `zone_name()`.
- **Features per person:** dwell seconds (last minus first time), path length (sum of step
  distances in frame units), average speed, and *stationary ratio* (share of steps slower than
  0.01 frame per second).
- **Zones:** the frame is split into a 3x3 grid named `r{row}c{col}`; the zone of a track is the
  one where it spent most points. **Limit:** these are *camera-view cells*, not named store areas.

### 3.7 KMeans: behaviour groups (unsupervised learning)
- **Where:** `classifier.py`, `segment_shoppers()`; called from `pipeline.analyze()`.
- **What it is:** KMeans clustering. It puts each person's four features (after
  standardising them) into k=3 groups by minimising distance to group centres. There are no
  labels; it only finds structure in the numbers. It is re-fitted for each video.
- **How groups get names:** the group with the highest average speed is called *passer-by*; of
  the rest, the one with the longest dwell is *engaged-shopper*; the last is *browser*.
- **Why:** it needs no labelled data, is instant, and gives a readable split.
- **Honest limit:** the names are *relative to that one video*. It always makes exactly three
  groups, even if everyone behaves alike, so a "browser" in one video is not comparable with a
  "browser" in another.

### 3.8 CLIP: reading attributes from a picture (zero-shot)
- **Where:** `classifier.py`, `carrying_bag()` and `classify_person()`.
- **What it is:** CLIP (ViT-B/32) is a neural network trained on image-text pairs. It maps an
  image and a sentence into the same vector space; similarity between them is a cosine score.
  *Zero-shot* means we give it sentences, not labelled training examples.
- **How we use it:** for the best (largest) crop of each person we compute the similarity to
  "a person carrying a bag" or "a shopping basket or cart" minus the similarity to "a person
  carrying nothing". Above zero means carrying.
- **Gate:** only for crops at least 180 pixels tall; smaller people are labelled `unknown`.
- **Measured:** 74% accuracy against a 51% baseline on 53 hand-labelled crops from the entrance
  clip (about plus or minus 12 points); on small mall crops (87 to 188 px) it was *worse than
  guessing*, which is why the gate exists.
- **Staff versus customer:** the same model can be asked, but it was never validated (there is no
  uniform in the test footage), so it is off.
- **Cost:** it loads a roughly 600 MB model, so it is opt-in per upload.

### 3.9 LSTM and baselines: footfall forecast
- **Where:** `retailvision/forecast.py`, `forecast()`; served at `/forecast/`.
- **Data:** per-second occupancy is averaged into **hourly buckets**.
- **The LSTM:** a recurrent neural network with one LSTM layer (32 hidden units) and a linear
  output. It looks at the last 6 hours (normalised) and predicts the next hour; multi-step
  forecasts feed each prediction back in. Trained with Adam (learning rate 0.01), mean-squared
  error, 200 epochs, on the store's own history, at request time.
- **Why also baselines:** an LSTM often does not beat simple rules on short data. So the code
  runs a **backtest**: train on the first 80%, predict the rest one step ahead, and compare the
  LSTM against two baselines: *last value* (tomorrow equals now) and *seasonal* (same hour 24
  hours earlier, looked up by real timestamp). The method with the lowest error is used and its
  name and errors are returned.
- **Measured:** on 17 minutes of mall data the LSTM only tied "last value", which is expected
  (no daily pattern in 17 minutes). On a clearly **synthetic** 14-day cycle it won, which shows
  the selection logic works, not that any real store benefits.
- **Fallback:** with fewer than 9 hourly points it returns the average with a small sine-shaped
  wiggle and labels it `baseline_extrapolation`. The wiggle is cosmetic and should be flat.

### 3.10 Local language model + SQL guard: the analyst
- **Where:** `retailvision/llm_analyst.py` and `retailvision/sqlguard.py`; endpoint `/ask/`.
- **Model:** Qwen2.5-1.5B-Instruct, a decoder-only transformer (about 1.5 billion parameters)
  run locally with greedy decoding.
- **Flow for a question:**
  1. If it contains words like "forecast" or "predict", call the forecaster directly.
  2. Otherwise build a prompt: table schema, rules, and 17 example question-to-SQL pairs;
     the model writes *one* SQL SELECT.
  3. **Deterministic fixes** are applied to the SQL (see below).
  4. The SQL runs through the **guard**. If SQLite reports a missing column, the error and the
     list of tables that own that column are given back to the model for one retry; a single-table
     query using another table's column is switched to the owning table.
  5. A second model call turns the returned rows into one to three sentences.
  6. The API returns the answer, the SQL, the rows, and `executed: true/false`.
- **The deterministic fixes (why they exist):** a 1.5B model makes characteristic mistakes, so
  the code removes `role = 'customer'` filters when role labelling is off (they can only return
  zero rows), drops zone literals the question never mentioned, adds a missing `segment` filter
  when the question names one behaviour group, drops comparisons against impossible values such
  as `segment = 'shoppers'`, and swaps to the right table for a missing column.
- **The guard:** SQLite's own *authorizer* allows reads only from four tables (so `jobs` and
  `sqlite_master` are unreachable) and denies everything else; the connection is read-only; a
  progress handler stops queries after 5 seconds; at most 50 rows return.
- **Measured:** on a 36-question benchmark tuned against, 100% lenient (92% strict); on 12
  questions written afterwards, 92% lenient before the last guard was added. Expect roughly 90%
  on new wording.
- **Why not a cloud model:** privacy and "no external API" were requirements. **Why this size:**
  it fits on a laptop; the price is the weaknesses above.

### 3.11 Job queue, monitor, auth, privacy (engineering, not ML)
- **Job queue** (`jobs.py`): analyses run one at a time on a single worker, stored in the SQLite
  `jobs` table, so a restart re-queues unfinished jobs and fails ones whose upload vanished. At
  most 5 may be queued.
- **Monitor** (`monitor.py`, `streams.py`): for a configured camera it records a short window and
  analyses it, over and over. URLs stay on the server; the browser only sees names.
- **Auth** (`auth.py`): an optional shared key (header or HttpOnly cookie, constant-time
  comparison) and a sliding-window rate limit on questions.
- **Privacy** (`privacy.py`): only derived numbers are stored; retention purge and a wipe-all
  with confirmation.

## 4. Algorithm comparison

| # | Algorithm | Kind of method | Learns from data? | Input | Output | Used for |
|---|---|---|---|---|---|---|
| 1 | YOLOv8s | CNN object detector | Pretrained (COCO), not retrained | one frame | boxes + scores | find people |
| 2 | ByteTrack | Kalman filter + Hungarian matching | No learning (classical) | boxes over time | tracks with IDs | follow people |
| 3 | Line crossing | Geometry (cross product) | No | track points + line | in/out events | entrance counts |
| 4 | Fragment merge | Greedy heuristic | No | tracks | fewer tracks | people estimate |
| 5 | KMeans | Unsupervised clustering | Fits on the video itself | 4 numbers per person | 3 groups | behaviour groups |
| 6 | CLIP | Vision-language network | Pretrained, zero-shot | person crop + sentences | similarity score | carrying a bag |
| 7 | LSTM | Recurrent neural network | Trains on the store's own history | last 6 hours | next hour(s) | footfall forecast |
| 8 | Seasonal / last-value | Baselines | No | history | next hour(s) | forecast fallback |
| 9 | Qwen2.5-1.5B | Transformer language model | Pretrained | question + schema | SQL, then sentences | analyst |
| 10 | SQL guard | Rules in the database engine | No | SQL text | allow / reject / time out | safety |

**Differences that matter:**
- *Detection vs tracking:* YOLO answers "who is in this frame" with no memory; ByteTrack adds
  time, so the same person keeps the same ID. Detection errors are per frame; tracking errors
  are identity errors (switches, fragments).
- *Learned vs rule-based:* YOLO, CLIP and Qwen were trained by others on huge datasets and used
  unchanged; the LSTM and KMeans are the only models fitted to *your* data; tracking, crossing
  and merging are hand-written rules.
- *Supervised vs unsupervised vs zero-shot:* KMeans uses no labels; CLIP uses sentences instead
  of labels; the LSTM is supervised by the series' own next value; YOLO was supervised on COCO.
- *Prediction vs generation:* the LSTM predicts a number; Qwen generates text (SQL, then a
  sentence), so it can produce plausible but wrong SQL, which is why execution is guarded.

## 5. Why these choices (and what we gave up)

| Decision | Chosen | Alternative | Reason | Cost |
|---|---|---|---|---|
| Detector | YOLOv8s at 1280 px | nano model at 640 px; larger models | measured error 4.1 vs 13.2 people | slower than nano |
| Tracker | ByteTrack | DeepSORT, BoT-SORT | no extra model, handles weak boxes | more ID switches in crowds |
| Segments | KMeans | fixed thresholds in seconds | no labels needed | names are relative per video |
| Attributes | CLIP zero-shot | train a classifier | no labelled data; measured 74% on big crops | weak on small people |
| Forecast | LSTM chosen by backtest | only baselines | can learn daily shape once data exists | needs days of history |
| Analyst | local Qwen 1.5B | cloud LLM | privacy, no API | about 90% on new wording |
| Storage | SQLite | server database | single file, zero setup | single machine |

## 6. Backend: what actually happens when you upload a video

1. The browser posts the file to `POST /analyze_video/?background=true&camera=...`.
2. `main.py` checks the extension and streams the file to `uploads/` with a size limit, then
   writes a row to `jobs` and returns a `job_id` at once.
3. The single worker thread picks it up and runs `pipeline.analyze()`:
   `track_video` (steps 3.1 to 3.6), `segment_shoppers` (3.7), optional `classify_person` (3.8).
4. Results are written in one transaction to `videos`, `tracks`, `customer_traffic`, `crossings`.
5. The job row becomes `done` with the summary as JSON; the upload file is deleted.
6. The browser polls `GET /jobs/{id}` every 1.5 s until `done`, then refreshes the dashboard.

Other routes: `/overview/` (KPIs and zone counts), `/customer_traffic/`, `/segments/`,
`/forecast/`, `/crossings/`, `/lines/` (GET, PUT, DELETE), `/frame/` (one JPEG for drawing a
line), `/ask/`, `/streams/`, `/analyze_stream/`, `/monitor/*`, `/privacy/*`, `/jobs/`, `/health`.

**Database tables:** `videos`, `tracks` (one row per person), `crossings`, `customer_traffic`
(one row per second), `count_lines`, `jobs`.

## 7. Frontend: what you see and where it comes from

| What you see | Source | How it is made |
|---|---|---|
| KPI cards: people tracked (estimate), dwell, engaged ratio, occupancy mean/peak, crossings | `GET /overview/` | plain HTML text updated by `dashboard.js` |
| Upload box, camera field, "carried bags" toggle, progress, summary card | `POST /analyze_video/`, `GET /jobs/{id}` | form + polling |
| Counting Lines card | browser canvas + `/frame/`, `/lines/` | you click two points; the arrow shows "in" |
| Footfall and forecast chart | `/customer_traffic/`, `/forecast/` | Chart.js line chart, history and forecast tabs |
| Behaviour donut and segment rows | `/segments/` and `/overview/` | Chart.js doughnut |
| 3x3 zone heat map | `zones` in `/overview/` | CSS cells shaded by count |
| Analyst chat | `GET /ask/` | text, the SQL shown, result rows, "Executed" or "Not executed" |
| Camera streams, privacy card | `/streams/`, `/privacy/*` | small forms |

Honest notes: the four "pipeline step" badges (YOLO, count and cross, KMeans, persist) are a
timed animation, not live progress from the server. The first request after start can wait while
the language model loads.

## 8. What was measured, and how far to trust it

| Claim | Evidence | Trust |
|---|---|---|
| Occupancy error about 14% | public mall dataset with per-frame counts | good |
| Detection error 4.1 people per frame | same dataset, 200 frames | good |
| Crossings: 70% precision, 71% recall | one 2-minute clip, labelled by one person | indicative |
| Carried bag 74% vs 51% | 53 crops, one clip | indicative (about 12 points) |
| Analyst about 90% on new wording | 12 unseen questions | indicative |
| 30-minute soak: 135 windows, no errors, memory +4.4% | looping file, isolated server | good, memory noisy |

## 9. Known limitations (all of them)

- **People tracked** is inflated by ID switches in crowds; use occupancy and crossings.
- **Crossings** need a camera that faces a door and a line you draw; no line means no crossing
  numbers by design.
- **Staff versus customer** is not validated at any resolution.
- **The forecast** needs days of data; under 9 hours it shows a flat-ish baseline with a cosmetic
  wiggle.
- **KMeans names** are relative per video.
- **Zones** are grid cells of the camera view, not store areas.
- **Single machine, single worker:** jobs run one at a time.
- **The page needs the internet** for Chart.js and Google Fonts; everything else is local.
- **Shared key**, not per-user accounts; the cookie is not marked Secure without HTTPS.
- **The analyst** is a small model: expect an occasional wrong query; read the SQL it shows.

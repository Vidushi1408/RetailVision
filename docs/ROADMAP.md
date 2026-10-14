# RetailVision: plan to close the remaining gaps

Goal: turn "works on one public mall dataset" into "numbers a store owner can trust".
Every item has an exit criterion that is a measurement, not an opinion.

## Where we start
Working and tested: detection and occupancy (MAE 3.8 people), entrance crossings (logic tested),
local analyst (27/30), dashboard, background jobs, CI (18 tests).
Unverified or off: unique visitors, crossing accuracy on real video, staff/bag labels, forecast
benefit, behaviour-segment meaning. See `PLAN.md` sections 10 and 11.

## Phase A: get real data and ground truth (Week 1)  [blocks everything else]
| # | Task | Owner area | Exit criterion |
|---|---|---|---|
| A1 | Obtain 3 to 5 clips from a real store (2 to 5 min each, entrance + aisle, daylight and evening). Note camera height, resolution, fps, and get consent/permission. | Product | Clips in `data/real/`, a one-line description of each |
| A2 | Hand-count crossings on a 60 s window of each clip (two people count independently). | Tracking | CSV `data/real/crossings_truth.csv`; the two counters agree within 10% |
| A3 | Hand-track identities on a 30 s window (ID per person per frame, a free tool such as CVAT or a simple spreadsheet of frame/box/ID). | Tracking | `data/real/identities_30s.json` with at least 15 people |
| A4 | Re-run the existing evaluation scripts on the real clips (`count_eval`, line crossings). | Tracking | A table of "mall vs real store" errors in `PLAN.md` |

Fallback if no real footage in week 1: MOT17/MOT20 public clips for identity truth; keep the mall
clip for occupancy. State clearly that the store-specific claim is then unproven.

## Phase B: make the headline numbers measurable (Weeks 2-3)
| # | Task | Owner area | Exit criterion |
|---|---|---|---|
| B1 | **Crossings:** compare the counter to A2; tune margin, line placement, and merge handling; add a "double-count guard" if one person is counted twice by ID switch. | Tracking | Error of in/out totals ≤ 15% on every real clip, or the gap is documented |
| B2 | **Unique visitors:** score the tracker on A3 with standard metrics (IDF1, ID switches, fragmentation). Tune ByteTrack buffer, merge gap/distance and the coverage guard against that number, not by eye. Try a re-ID model only if IDF1 stays low. | Tracking | IDF1 reported; unique-visitor error ≤ 20% on the window, or the metric is dropped from the UI |
| B3 | **Occupancy:** test yolov8m and tiled inference on the densest frames; record accuracy vs speed. | Tracking | Keep the cheaper option unless MAE improves by ≥ 1 person |
| B4 | **Segments:** replace forced KMeans with rules in real units (for example passer-by < 8 s and fast; engaged > 60 s and mostly stationary); show the thresholds in the UI. | Behaviour | Same clip gives the same labels regardless of what else was uploaded |
| B5 | **Staff/bag labels:** only if real clips have crops ≥ 256 px; label 150 crops, re-run `bag_eval.py`, train a linear probe. Otherwise leave off. | Classification | ≥ 80% accuracy and above the majority baseline, or stay disabled |

## Phase C: forecasting that earns its place (Weeks 2-4, parallel with B)
| # | Task | Exit criterion |
|---|---|---|
| C1 | Collect or record at least 7 days of hourly occupancy from the real store (a scheduled job that analyses a clip per hour, or POS/door-counter data if available). | ≥ 168 hourly points in the DB |
| C2 | Re-run `forecast_eval.py` with a seasonal baseline (same hour yesterday / last week) in addition to "last value". | LSTM beats the best baseline on a held-out day, otherwise ship the seasonal baseline as the default |
| C3 | Show the method and its backtest error next to the forecast in the dashboard. | Visible on the chart |

## Phase D: product features a store owner needs (Weeks 2-4)
| # | Task | Exit criterion |
|---|---|---|
| D1 | **Entrance-line setup screen:** draw the line on a frame, save it (replaces the `COUNT_LINES` env var). **Done:** dashboard card, `/lines/` API, server-side frame fallback, 6 tests; click-tested end to end. Per-camera storage comes with D2. | A non-developer can set a line in under a minute |
| D2 | **Multiple cameras/lines** with names (door, till queue, aisle). | Crossings and occupancy reported per line |
| D3 | **Live stream input** (RTSP or webcam) with a rolling window instead of file upload. | Runs 30 minutes without growing memory |
| D4 | **Persist jobs** in SQLite so a restart does not lose queued work. | Restart test passes |
| D5 | **Auth and limits:** a simple API key or login on upload/ask; analyst table allow-list and a 5 s SQL timeout. | Unauthenticated calls return 401; a runaway query is cancelled |
| D6 | **Privacy controls:** do not keep the uploaded video after analysis (already deleted), optional face blur on exported frames, retention setting for the database, a short privacy note in the README. | Documented and tested |

## Phase E: quality gates and release (Week 4)
| # | Task | Exit criterion |
|---|---|---|
| E1 | Automated end-to-end test on a 20 s fixture clip (upload, job, summary, crossings, analyst answer). | Runs in CI |
| E2 | Add `count_eval` and the 30-question analyst benchmark as scheduled/optional CI jobs with thresholds (MAE < 6, SQL ≥ 80%). | Regressions fail the build |
| E3 | Browser test of the dashboard flow (Playwright). | Upload and results asserted |
| E4 | Repo tidy: delete merged branches, `git rm --cached yolov8n.pt`, pin Python version in CI, tag `v1.0`. | Clean tree, release tag |
| E5 | Rehearse the presentation with the real app; update the deck numbers from Phase B/C. | One timed run-through; deck matches `PLAN.md` |

## Suggested split for five people
| Person | Phases |
|---|---|
| 1 Tracking lead | A2-A4, B1, B2, B3 |
| 2 Classification/behaviour | B4, B5 |
| 3 Forecast | C1-C3 |
| 4 Analyst + backend | D4, D5, D6, E1, E2 |
| 5 Product/UI | A1, D1, D2, D3, E3, E5 |

## Order of work and dependencies
```
A1 footage ──► A2/A3 ground truth ──► B1/B2 tuning ──► update deck (E5)
A1 ──► C1 collect history ──► C2 forecast eval
D1, D4, D5, D6 and E1-E4 do not depend on footage and can start immediately.
```

## Risks
| Risk | Mitigation |
|---|---|
| No real footage arrives | Use MOT17/MOT20 for identity truth; label claims as "validated on public data only" |
| Real crops stay too small for attributes | Keep staff/bag off; sell occupancy, crossings, dwell, zones |
| Tracker cannot reach the identity target | Drop "unique visitors" from the UI; report occupancy and crossings only |
| Live stream too slow on a laptop | Lower resolution/stride, smaller model, or run the stream on a small GPU box |
| Privacy objections | Delete-after-analysis is already default; add blur and retention before any deployment |

## Definition of "done"
1. Occupancy, crossings and unique visitors each have a measured error on real store footage.
2. Anything unmeasured is either removed from the UI or labelled as an estimate.
3. Forecast ships with a backtest that beats a seasonal baseline, or ships as the baseline.
4. A non-developer can set up a line and upload a clip without touching the terminal.
5. CI guards accuracy, not just that the code runs.

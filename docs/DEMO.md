# Presenting RetailVision (about 10 minutes)

Slides: `docs/RetailVision_presentation.pptx` (speaker notes on every slide).

## Before you start (once, about 5 minutes)
```bash
.venv/bin/uvicorn main:app --port 8765     # models preload in the background; wait ~1 min
```
Have these ready: `data/mall_clip.mp4` (200 s clip) and `demo/annotated_demo.mp4`
(rebuild with `PYTHONPATH=. .venv/bin/python scripts/render_demo.py data/mall_clip.mp4 demo/annotated_demo.mp4`).
Use a clean database for the demo: stop the server, delete `store_analytics.db`, start again.

## Run of show
| Min | Slide | What to do |
|---|---|---|
| 0-1 | 1-2 | The problem: cameras exist, nobody turns them into numbers. We rebuilt a naive counter. |
| 1-3 | 3 | Walk the pipeline left to right. Everything is local. |
| 3-4 | 4 | Play `annotated_demo.mp4`. Point at the counters moving as people cross the line. |
| 4-6 | 6 | Live: open http://localhost:8765, upload `mall_clip.mp4`, untick CLIP, watch it finish (~30 s). |
| 6-7 | 5 | Accuracy against ground truth: error cut from 13 to 4 people per frame. |
| 7-8 | 7 | Ask the analyst 2-3 questions in the dashboard (see below). |
| 8-9 | 8 | Be candid: three things that did not work and what we did. |
| 9-10 | 9-10 | Next steps and close. |

## Questions that work well in the analyst box
- How many people crossed the line going in?
- How many people left?
- How many engaged shoppers are there?
- What is the average dwell time of engaged shoppers?
- Which zone has the most people?

## Known limits (say them before you are asked)
- "People tracked" is an **estimate**: IDs fragment in crowds and there is no identity ground truth.
- Crossing counts are **not validated** against a hand count.
- Staff/customer and bag labels are **off**: they were not accurate on this footage.
- The forecast needs days of history. With one clip it shows a baseline and says so.
- The analyst is a small local model: about 90% of benchmark questions are correct, so
  glance at the SQL it shows.

"""End-to-end video analysis: track -> classify -> segment -> persist."""
import datetime
import json
import os

from .classifier import attributes_enabled, classify_person, segment_shoppers
from .db import connect
from .tracker import track_features, track_video


# Counting lines as {"name": [ax, ay, bx, by]} in normalised frame coordinates (0-1).
# Override with the COUNT_LINES env var (JSON). Default: one horizontal line mid-frame.
def count_lines():
    raw = os.getenv("COUNT_LINES")
    return json.loads(raw) if raw else {"entrance": [0.0, 0.5, 1.0, 0.5]}


def analyze(path, filename, use_clip=True):
    meta, tracks, crops = track_video(path, lines=count_lines())
    ids = [i for i, t in tracks.items() if len(t["points"]) >= 3]  # drop flickers
    feats = [track_features(tracks[i]) for i in ids]
    segments = segment_shoppers(feats)

    conn = connect()
    now = datetime.datetime.now()
    vid = conn.execute("INSERT INTO videos (filename, fps, duration_sec, created) VALUES (?,?,?,?)",
                       (filename, meta["fps"], meta["duration_sec"], now.isoformat())).lastrowid
    for i, f, seg in zip(ids, feats, segments):
        role, carry = classify_person(crops.get(i)) if use_clip and attributes_enabled() else ("unknown", "unknown")
        conn.execute("""INSERT INTO tracks (video_id, track_id, first_sec, last_sec, dwell_sec,
                        path_length, avg_speed, stationary_ratio, zone, role, carrying, segment)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (vid, i, f["first_sec"], f["last_sec"], f["dwell_sec"], f["path_length"],
                      f["avg_speed"], f["stationary_ratio"], f["zone"], role, carry, seg))
    occ = meta["occupancy_per_second"]
    for sec, n in occ.items():  # occupancy = people visible, counting detections without a track id
        ts = (now + datetime.timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT INTO customer_traffic (timestamp, people_count) VALUES (?,?)", (ts, round(n)))
    crossing_summary = {}
    for name, events in meta["crossings"].items():
        for sec, direction in events:
            conn.execute("INSERT INTO crossings (video_id, line_name, direction, sec) VALUES (?,?,?,?)",
                         (vid, name, direction, sec))
        crossing_summary[name] = {"in": sum(d == "in" for _, d in events),
                                  "out": sum(d == "out" for _, d in events)}
    conn.commit()
    avg_dwell = round(sum(f["dwell_sec"] for f in feats) / max(len(feats), 1), 1) if feats else 0.0
    summary = {
        "video_id": vid,
        "unique_people": len(ids),
        "people_count": len(ids),  # Backward compatibility
        "duration_sec": round(meta["duration_sec"], 1),
        "avg_dwell_sec": avg_dwell,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "segments": {s: segments.count(s) for s in set(segments)},
        "occupancy": {"mean": round(sum(occ.values()) / max(len(occ), 1), 1),
                      "peak": round(max(occ.values(), default=0))},
        "crossings": crossing_summary
    }
    conn.close()
    return summary


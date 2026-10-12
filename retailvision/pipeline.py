"""End-to-end video analysis: track -> classify -> segment -> persist."""
import datetime

from .classifier import classify_person, segment_shoppers
from .db import connect
from .tracker import track_features, track_video


def analyze(path, filename, use_clip=True):
    meta, tracks, crops = track_video(path)
    ids = [i for i, t in tracks.items() if len(t["points"]) >= 3]  # drop flickers
    feats = [track_features(tracks[i]) for i in ids]
    segments = segment_shoppers(feats)

    conn = connect()
    now = datetime.datetime.now()
    vid = conn.execute("INSERT INTO videos (filename, fps, duration_sec, created) VALUES (?,?,?,?)",
                       (filename, meta["fps"], meta["duration_sec"], now.isoformat())).lastrowid
    for i, f, seg in zip(ids, feats, segments):
        role, carry = classify_person(crops.get(i)) if use_clip else ("unknown", "unknown")
        conn.execute("""INSERT INTO tracks (video_id, track_id, first_sec, last_sec, dwell_sec,
                        path_length, avg_speed, stationary_ratio, zone, role, carrying, segment)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (vid, i, f["first_sec"], f["last_sec"], f["dwell_sec"], f["path_length"],
                      f["avg_speed"], f["stationary_ratio"], f["zone"], role, carry, seg))
    for sec, n in meta["per_second_counts"].items():
        ts = (now + datetime.timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT INTO customer_traffic (timestamp, people_count) VALUES (?,?)", (ts, n))
    conn.commit()
    avg_dwell = round(sum(f["dwell_sec"] for f in feats) / max(len(feats), 1), 1) if feats else 0.0
    summary = {
        "video_id": vid,
        "unique_people": len(ids),
        "people_count": len(ids),  # Backward compatibility
        "duration_sec": round(meta["duration_sec"], 1),
        "avg_dwell_sec": avg_dwell,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "segments": {s: segments.count(s) for s in set(segments)}
    }
    conn.close()
    return summary


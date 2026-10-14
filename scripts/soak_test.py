"""Soak test for the rolling monitor: run it for N minutes and report memory growth.

Usage: python scripts/soak_test.py http://localhost:8768 STREAM_NAME 30 [window_sec=30]
The server must have STREAM_SOURCES set (a video file works as a looping fake camera) and ideally
its own RETAILVISION_DB / UPLOAD_DIR so it does not touch real data.
Pass criterion: RSS after warm-up grows by less than 10% between the middle and last thirds.
"""
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request


def call(base, path, method="GET"):
    req = urllib.request.Request(base + path, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main(base, name, minutes, window=30):
    q = urllib.parse.urlencode({"name": name, "window": window})
    call(base, "/monitor/start?" + q, "POST")
    t0, samples = time.time(), []
    print(f"monitoring '{name}' for {minutes} min, {window}s windows", flush=True)
    while time.time() - t0 < minutes * 60:
        time.sleep(30)
        st = call(base, "/monitor/")
        m = st["monitors"][name]
        samples.append((round(time.time() - t0), m["windows_done"], m["errors"], st["server_rss_mb"]))
        print(f"t={samples[-1][0]:5d}s windows={m['windows_done']:3d} errors={m['errors']} rss={st['server_rss_mb']} MB", flush=True)
    call(base, "/monitor/stop?" + urllib.parse.urlencode({"name": name}), "POST")
    rss = [s[3] for s in samples if s[3]]
    third = max(1, len(rss) // 3)
    mid, last = statistics.mean(rss[third:2 * third] or rss), statistics.mean(rss[-third:])
    growth = 100 * (last - mid) / mid
    print(json.dumps({"minutes": minutes, "samples": len(samples), "windows": samples[-1][1], "errors": samples[-1][2],
                      "rss_start_mb": rss[0], "rss_mid_mb": round(mid, 1), "rss_last_mb": round(last, 1),
                      "growth_mid_to_last_pct": round(growth, 1), "pass": growth < 10 and samples[-1][2] == 0}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 30)

"""Data retention and wipe. RetailVision stores derived analytics only: counts, track
statistics and crossing events. Uploaded video is deleted once its job ends, person crops are
held in memory during analysis and never written, and nothing here contains faces or images.
"""
import datetime
import time

from .db import connect

ANALYTICS_TABLES = ("tracks", "crossings", "videos", "customer_traffic", "jobs")


def purge_older_than(days):
    """Delete analytics older than `days`. Returns {table: rows_deleted}."""
    if days < 0:
        raise ValueError("days must be >= 0")
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    cutoff_iso, cutoff_ts = cutoff.isoformat(), cutoff.strftime("%Y-%m-%d %H:%M:%S")
    conn = connect()
    try:
        old = [r[0] for r in conn.execute("SELECT id FROM videos WHERE created < ?", (cutoff_iso,))]
        deleted = {}
        if old:
            marks = ",".join("?" * len(old))
            deleted["tracks"] = conn.execute(f"DELETE FROM tracks WHERE video_id IN ({marks})", old).rowcount
            deleted["crossings"] = conn.execute(f"DELETE FROM crossings WHERE video_id IN ({marks})", old).rowcount
            deleted["videos"] = conn.execute(f"DELETE FROM videos WHERE id IN ({marks})", old).rowcount
        else:
            deleted.update(tracks=0, crossings=0, videos=0)
        deleted["customer_traffic"] = conn.execute(
            "DELETE FROM customer_traffic WHERE timestamp < ?", (cutoff_ts,)).rowcount
        deleted["jobs"] = conn.execute(
            "DELETE FROM jobs WHERE status IN ('done','error') AND created < ?",
            (time.time() - days * 86400,)).rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def wipe_all():
    """Delete every analytics row (counting lines are configuration and are kept)."""
    conn = connect()
    try:
        deleted = {t: conn.execute(f"DELETE FROM {t}" + (
            " WHERE status IN ('done','error')" if t == "jobs" else "")).rowcount for t in ANALYTICS_TABLES}
        conn.commit()
        return deleted
    finally:
        conn.close()

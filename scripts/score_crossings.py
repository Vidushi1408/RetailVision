"""Score the crossing counter against hand labels made on gt_candidates.py contact sheets.

Usage: python scripts/score_crossings.py candidates.json labels.json
Labels per candidate number: TP (real first crossing, right direction), DUP (same person
again), WRONG (real crossing, wrong direction), NOT (not a crossing), UNSURE (excluded).
"""
import json
import sys


def score(cands, labels):
    rows = []
    for c in cands:
        lab = labels.get(str(c["n"]), "UNSURE")
        rows.append({**c, "label": lab})
    counted = [r for r in rows if r["counted"]]
    known = [r for r in counted if r["label"] != "UNSURE"]
    tp = sum(r["label"] == "TP" for r in known)
    bad = {k: sum(r["label"] == k for r in known) for k in ("DUP", "WRONG", "NOT")}
    # distinct people who really crossed: every TP anywhere in the pool, plus people whose only
    # event carries the wrong direction (their crossing exists, the counter just got it wrong)
    all_tp = [r for r in rows if r["label"] == "TP"]
    wrong_only = [r for r in rows if r["label"] == "WRONG"]
    gt = len(all_tp) + len(wrong_only) - sum(1 for r in wrong_only if r.get("has_tp"))
    out = {
        "candidates": len(rows), "counted_events": len(counted), "counted_unsure": len(counted) - len(known),
        "counted_true_positive": tp, "counted_duplicates": bad["DUP"], "counted_wrong_direction": bad["WRONG"],
        "counted_not_crossing": bad["NOT"],
        "precision": round(tp / max(len(known), 1), 3),
        "ground_truth_distinct_crossings": gt,
        "recall": round(tp / max(gt, 1), 3),
        "missed_by_production_but_real": sum(1 for r in all_tp if not r["counted"]),
    }
    p, r = out["precision"], out["recall"]
    out["f1"] = round(2 * p * r / (p + r), 3) if p + r else 0.0
    # same pool, before reversal cancelling: every production candidate counts
    prod = [r for r in rows if r["source"] == "production" and r["label"] != "UNSURE"]
    out["precision_without_cancelling"] = round(sum(r["label"] == "TP" for r in prod) / max(len(prod), 1), 3)
    return out


if __name__ == "__main__":
    print(json.dumps(score(json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))), indent=1))

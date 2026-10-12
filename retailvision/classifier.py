"""Data classification: CLIP zero-shot attributes + KMeans behaviour segments.

CLIP staff/customer and carrying attributes are OFF by default: on 640x480 mall CCTV
(~90-190 px person crops) they scored below the majority-class baseline (see PLAN.md
section 10). Set ENABLE_CLIP_ATTRIBUTES=1 to turn them on for higher-resolution footage.
"""
import os

import numpy as np


def attributes_enabled():
    return os.getenv("ENABLE_CLIP_ATTRIBUTES", "0") == "1"

_clip = {}

ROLE_LABELS = ["a photo of a store employee wearing a uniform",
               "a photo of a customer shopping"]
CARRY_LABELS = ["a person carrying a shopping basket or cart",
                "a person carrying a bag",
                "a person carrying nothing"]


from .device import get_device


def _load_clip():
    if not _clip:
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        dev = get_device()
        _clip["m"] = CLIPModel.from_pretrained(name).to(dev).eval()
        _clip["p"] = CLIPProcessor.from_pretrained(name)
        _clip["d"] = dev
    return _clip["m"], _clip["p"], _clip["d"]


def clip_zero_shot(crop_bgr, labels):
    """Classify one image crop against free-text labels. Returns (label, prob)."""
    import torch
    from PIL import Image
    model, proc, dev = _load_clip()
    img = Image.fromarray(crop_bgr[:, :, ::-1])
    inputs = proc(text=labels, images=img, return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    i = int(probs.argmax())
    return labels[i], float(probs[i])



MIN_CROP_HEIGHT = 128  # px; below this CLIP is confidently wrong (Phase 1: 58/60 "employee")


def classify_person(crop_bgr):
    """Return (role, carrying), or ("unknown", "unknown") when the crop is too small to judge."""
    if (not attributes_enabled() or crop_bgr is None or crop_bgr.size == 0
            or crop_bgr.shape[0] < MIN_CROP_HEIGHT):
        return "unknown", "unknown"
    role, _ = clip_zero_shot(crop_bgr, ROLE_LABELS)
    carry, _ = clip_zero_shot(crop_bgr, CARRY_LABELS)
    return ("employee" if "employee" in role else "customer",
            carry.replace("a person carrying ", "").replace("a ", "").replace("shopping basket or cart", "basket"))


FEATURES = ["dwell_sec", "path_length", "avg_speed", "stationary_ratio"]


def segment_shoppers(feats, k=3):
    """Unsupervised clustering of trajectories; clusters are named by their
    centroid: lowest speed+highest dwell -> 'browser', fastest -> 'passer-by'."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    if len(feats) < k:
        return ["browser"] * len(feats)
    X = StandardScaler().fit_transform([[f[c] for c in FEATURES] for f in feats])
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    cent = km.cluster_centers_
    dwell, speed = cent[:, 0], cent[:, 2]
    names = {}
    names[int(np.argmax(speed))] = "passer-by"
    rest = [i for i in range(k) if i not in names]
    names[max(rest, key=lambda i: dwell[i])] = "engaged-shopper"
    for i in range(k):
        names.setdefault(i, "browser")
    return [names[int(l)] for l in km.labels_]

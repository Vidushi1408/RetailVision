"""Data classification: CLIP zero-shot attributes + KMeans behaviour segments.

What is validated and what is not (details in PLAN.md sections 10 and 15):
- "carrying a bag": on person crops >= 180 px tall (the supplied entrance clip, 188-295 px)
  CLIP zero-shot scored 74% accuracy against a 51% majority baseline on 53 hand-labelled crops
  (about +/-12 points at that sample size). On 87-188 px mall crops it was below baseline, so
  below MIN_CARRY_HEIGHT the answer is "unknown". Set DISABLE_CARRYING=1 to turn it off.
- staff vs customer: NOT validated at any resolution (no uniform visible in the test footage;
  58/60 mall crops came back "employee"). Off unless ENABLE_CLIP_ATTRIBUTES=1.
"""
import os
import threading

import numpy as np

from .device import get_device


def attributes_enabled():
    """Staff/customer labelling (unvalidated). Carrying is controlled separately."""
    return os.getenv("ENABLE_CLIP_ATTRIBUTES", "0") == "1"


def carrying_enabled():
    return os.getenv("DISABLE_CARRYING", "0") != "1"


_clip = {}
_clip_lock = threading.Lock()
_text_cache = {}

ROLE_LABELS = ["a photo of a store employee wearing a uniform",
               "a photo of a customer shopping"]
CARRY_POS = ["a person carrying a bag", "a person carrying a shopping basket or cart"]
CARRY_NEG = ["a person carrying nothing"]
MIN_CROP_HEIGHT = 128   # px; role guesses below this were confidently wrong
MIN_CARRY_HEIGHT = 180  # px; smallest crop where the bag check was validated (188)


def _load_clip():
    """Thread-safe load-once; `ready` is set only after model and processor both exist."""
    if not _clip.get("ready"):
        with _clip_lock:
            if not _clip.get("ready"):
                from transformers import CLIPModel, CLIPProcessor
                name = "openai/clip-vit-base-patch32"
                dev = get_device()
                model = CLIPModel.from_pretrained(name).to(dev).eval()
                proc = CLIPProcessor.from_pretrained(name)
                _clip.update(m=model, p=proc, d=dev, ready=True)
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


def _feat(out):
    # transformers >= 5 returns an output object whose pooler_output is the projected feature
    import torch
    return out if torch.is_tensor(out) else out.pooler_output


def _embed_image(crop_bgr):
    import torch
    from PIL import Image
    model, proc, dev = _load_clip()
    inputs = proc(images=Image.fromarray(crop_bgr[:, :, ::-1]), return_tensors="pt")
    with torch.no_grad():
        f = _feat(model.get_image_features(**{k: v.to(dev) for k, v in inputs.items()}))
    return torch.nn.functional.normalize(f, dim=-1)[0]


def _embed_text_mean(prompts):
    import torch
    key = tuple(prompts)
    if key not in _text_cache:
        model, proc, dev = _load_clip()
        inputs = proc(text=list(prompts), return_tensors="pt", padding=True)
        with torch.no_grad():
            f = _feat(model.get_text_features(**{k: v.to(dev) for k, v in inputs.items()}))
        _text_cache[key] = torch.nn.functional.normalize(f, dim=-1).mean(0)
    return _text_cache[key]


def carrying_bag(crop_bgr):
    """("bag" | "nothing", score) for a person crop, or ("unknown", None) if it is too small.

    Same scoring as evaluation/bag_eval.py (variant A): cosine similarity to the "carrying a
    bag/basket" prompts minus the "carrying nothing" prompt; positive means carrying.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < MIN_CARRY_HEIGHT:
        return "unknown", None
    img = _embed_image(crop_bgr)
    score = float(img @ _embed_text_mean(CARRY_POS) - img @ _embed_text_mean(CARRY_NEG))
    return ("bag" if score > 0 else "nothing"), score


def classify_person(crop_bgr):
    """Return (role, carrying). Each is "unknown" when it cannot be judged reliably."""
    role = "unknown"
    if (attributes_enabled() and crop_bgr is not None and crop_bgr.size
            and crop_bgr.shape[0] >= MIN_CROP_HEIGHT):
        label, _ = clip_zero_shot(crop_bgr, ROLE_LABELS)
        role = "employee" if "employee" in label else "customer"
    carry = carrying_bag(crop_bgr)[0] if carrying_enabled() else "unknown"
    return role, carry


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

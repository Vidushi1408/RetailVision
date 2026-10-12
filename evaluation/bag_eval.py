"""CLIP zero-shot vs linear probe for "carrying a bag" on the hand-labelled mall crops.

Usage: PYTHONPATH=. python evaluation/bag_eval.py   (needs data/labeling/crops, see README)
"""
import csv

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from transformers import CLIPModel, CLIPProcessor

from evaluation.bag_labels import LABELS
from retailvision.device import get_device

NAME, dev = "openai/clip-vit-base-patch32", get_device()
model = CLIPModel.from_pretrained(NAME).to(dev).eval()
proc = CLIPProcessor.from_pretrained(NAME)


def _feat(out, _proj=None):
    # transformers >= 5 returns an output object whose pooler_output is the projected feature
    return out if torch.is_tensor(out) else out.pooler_output


def image_emb(imgs):
    with torch.no_grad():
        o = model.get_image_features(**{k: v.to(dev) for k, v in proc(images=imgs, return_tensors="pt").items()})
    return torch.nn.functional.normalize(_feat(o, model.visual_projection), dim=-1)


def text_emb(prompts):
    with torch.no_grad():
        o = model.get_text_features(**{k: v.to(dev) for k, v in proc(text=prompts, return_tensors="pt", padding=True).items()})
    return torch.nn.functional.normalize(_feat(o, model.text_projection), dim=-1)


def prf(pr, y):
    tp, fp, fn = ((pr == 1) & (y == 1)).sum(), ((pr == 1) & (y == 0)).sum(), ((pr == 0) & (y == 1)).sum()
    return (pr == y).mean(), tp / max(tp + fp, 1), tp / max(tp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)


if __name__ == "__main__":
    ids = [int(r[0]) for r in list(csv.reader(open("data/labeling/labels.csv")))[1:]]
    idx = sorted(LABELS)
    y = np.array([LABELS[i] for i in idx])
    emb = image_emb([Image.fromarray(cv2.imread(f"data/labeling/crops/{ids[i]}.jpg")[:, :, ::-1]) for i in idx])
    print(f"n={len(y)} bag={y.sum()} none={(1 - y).sum()}  majority-class accuracy={max(y.mean(), 1 - y.mean()):.2f}")
    prompts = {
        "A: current wording": (["a person carrying a bag", "a person carrying a shopping basket or cart"], ["a person carrying nothing"]),
        "B: binary": (["a photo of a person carrying a shopping bag or handbag"], ["a photo of a person with empty hands"]),
        "C: ensemble": (["a person holding a shopping bag", "a person carrying a handbag",
                         "a person with a backpack or shoulder bag", "a shopper carrying bags"],
                        ["a person walking with empty hands", "a person with no bags", "a person walking in a mall"]),
    }
    for name, (pos, neg) in prompts.items():
        s = (emb @ text_emb(pos).mean(0) - emb @ text_emb(neg).mean(0)).cpu().numpy()
        a, p, r, f = prf((s > 0).astype(int), y)
        print(f"{name:20s} zero-shot (thr=0): acc={a:.2f} prec={p:.2f} rec={r:.2f} F1={f:.2f}")
    X = emb.cpu().numpy()
    for C in (0.5, 2, 10):
        pr = cross_val_predict(LogisticRegression(C=C, max_iter=1000, class_weight="balanced"), X, y, cv=LeaveOneOut())
        a, p, r, f = prf(pr, y)
        print(f"linear probe C={C:<4} leave-one-out: acc={a:.2f} prec={p:.2f} rec={r:.2f} F1={f:.2f}")

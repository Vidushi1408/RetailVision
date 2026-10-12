"""Load heavy models once at server start so the first request is not 30 s slower."""
import os


def preload():
    wanted = os.getenv("PRELOAD_MODELS", "yolo,llm").split(",")
    if "yolo" in wanted:
        from .tracker import get_model
        get_model()
    if "clip" in wanted and os.getenv("ENABLE_CLIP_ATTRIBUTES", "0") == "1":
        from .classifier import _load_clip
        _load_clip()
    if "llm" in wanted:
        from .llm_analyst import load_llm
        load_llm()

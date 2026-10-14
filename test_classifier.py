"""Attribute gating logic. CLIP itself is replaced by a stub, so no weights are needed;
accuracy of the real model is measured by evaluation/bag_eval.py, not here."""
import os
import unittest
from unittest import mock

import numpy as np

from retailvision import classifier as C


def crop(h, w=80):
    return np.full((h, w, 3), 120, np.uint8)


class FakeVec:
    """Stand-in for a unit vector: `a @ b` returns a preset similarity."""
    def __init__(self, sims):
        self.sims = sims

    def __matmul__(self, other):
        return self.sims[other]


class TestCarrying(unittest.TestCase):
    def _patch_scores(self, pos_sim, neg_sim):
        pos, neg = "POS", "NEG"
        img = FakeVec({pos: pos_sim, neg: neg_sim})

        def text(prompts):
            return pos if tuple(prompts) == tuple(C.CARRY_POS) else neg

        return mock.patch.object(C, "_embed_image", lambda c: img), \
            mock.patch.object(C, "_embed_text_mean", text)

    def test_small_crops_are_unknown_and_never_touch_the_model(self):
        with mock.patch.object(C, "_embed_image", side_effect=AssertionError("model must not run")):
            self.assertEqual(C.carrying_bag(crop(C.MIN_CARRY_HEIGHT - 1)), ("unknown", None))
            self.assertEqual(C.carrying_bag(None), ("unknown", None))
            self.assertEqual(C.carrying_bag(np.zeros((0, 0, 3), np.uint8)), ("unknown", None))

    def test_positive_score_means_bag_negative_means_nothing(self):
        a, b = self._patch_scores(0.30, 0.25)
        with a, b:
            label, score = C.carrying_bag(crop(200))
        self.assertEqual(label, "bag")
        self.assertAlmostEqual(score, 0.05)
        a, b = self._patch_scores(0.20, 0.27)
        with a, b:
            self.assertEqual(C.carrying_bag(crop(200))[0], "nothing")

    def test_height_boundary(self):
        a, b = self._patch_scores(0.3, 0.2)
        with a, b:
            self.assertEqual(C.carrying_bag(crop(C.MIN_CARRY_HEIGHT))[0], "bag")

    def test_role_stays_unknown_without_the_flag_but_carrying_works(self):
        a, b = self._patch_scores(0.3, 0.2)
        with a, b, mock.patch.dict(os.environ, {"ENABLE_CLIP_ATTRIBUTES": "0", "DISABLE_CARRYING": "0"}), \
                mock.patch.object(C, "clip_zero_shot", side_effect=AssertionError("role model must not run")):
            self.assertEqual(C.classify_person(crop(220)), ("unknown", "bag"))

    def test_role_runs_only_when_enabled_and_big_enough(self):
        a, b = self._patch_scores(0.1, 0.2)
        with a, b, mock.patch.dict(os.environ, {"ENABLE_CLIP_ATTRIBUTES": "1"}), \
                mock.patch.object(C, "clip_zero_shot", return_value=("a photo of a customer shopping", 0.9)):
            self.assertEqual(C.classify_person(crop(220)), ("customer", "nothing"))
            self.assertEqual(C.classify_person(crop(100))[0], "unknown")  # below MIN_CROP_HEIGHT

    def test_carrying_can_be_disabled(self):
        with mock.patch.dict(os.environ, {"DISABLE_CARRYING": "1"}), \
                mock.patch.object(C, "_embed_image", side_effect=AssertionError("must not run")):
            self.assertEqual(C.classify_person(crop(250)), ("unknown", "unknown"))


if __name__ == "__main__":
    unittest.main()

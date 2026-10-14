"""Line-crossing logic on synthetic tracks (no models or video needed)."""
import unittest

from retailvision.tracker import line_crossings

H_LINE = (0.0, 0.5, 1.0, 0.5)  # horizontal line mid-frame; "in" = moving down the image


def path(*xy):
    return [(i * 0.5, x, y) for i, (x, y) in enumerate(xy)]


class TestLineCrossings(unittest.TestCase):
    def test_downward_crossing_is_in(self):
        ev = line_crossings(path((0.5, 0.2), (0.5, 0.4), (0.5, 0.6), (0.5, 0.8)), H_LINE)
        self.assertEqual([d for _, d in ev], ["in"])

    def test_upward_crossing_is_out(self):
        ev = line_crossings(path((0.5, 0.8), (0.5, 0.6), (0.5, 0.4)), H_LINE)
        self.assertEqual([d for _, d in ev], ["out"])

    def test_no_crossing_when_staying_on_one_side(self):
        self.assertEqual(line_crossings(path((0.1, 0.1), (0.5, 0.3), (0.9, 0.2)), H_LINE), [])

    def test_jitter_on_the_line_is_not_counted(self):
        # hovers within +-0.015 of the line (clearly inside the 0.02 margin) after arriving from above
        p = path((0.5, 0.3), (0.5, 0.51), (0.5, 0.49), (0.5, 0.515), (0.5, 0.485), (0.5, 0.505))
        self.assertEqual(line_crossings(p, H_LINE), [])

    def test_zigzag_across_line_counts_each_real_crossing(self):
        p = path((0.5, 0.3), (0.5, 0.7), (0.5, 0.3), (0.5, 0.7))
        self.assertEqual([d for _, d in line_crossings(p, H_LINE)], ["in", "out", "in"])

    def test_walking_around_the_end_of_a_short_line_is_not_counted(self):
        short = (0.4, 0.5, 0.6, 0.5)  # only the middle fifth of the frame
        p = path((0.1, 0.2), (0.1, 0.4), (0.1, 0.6), (0.1, 0.8))  # passes far to the left
        self.assertEqual(line_crossings(p, short), [])

    def test_vertical_line_direction(self):
        v = (0.5, 0.0, 0.5, 1.0)  # A->B points down; the right-hand side is image-left (x<0.5)
        ev = line_crossings(path((0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)), v)
        self.assertEqual(len(ev), 1)

    def test_event_time_is_the_sample_after_the_crossing(self):
        ev = line_crossings(path((0.5, 0.3), (0.5, 0.7)), H_LINE)
        self.assertEqual(ev, [(0.5, "in")])


class TestReversalCancellation(unittest.TestCase):
    """A track flipping across the line within the window is jitter, not two real crossings."""

    def test_quick_reversal_is_cancelled(self):
        p = path((0.5, 0.3), (0.5, 0.7), (0.5, 0.3))  # in at 0.5 s, out at 1.0 s
        self.assertEqual(len(line_crossings(p, H_LINE)), 2)
        self.assertEqual(line_crossings(p, H_LINE, reversal_window=3.0), [])

    def test_slow_reversal_is_kept(self):
        p = [(0.0, 0.5, 0.3), (0.5, 0.5, 0.7), (10.0, 0.5, 0.7), (10.5, 0.5, 0.3)]  # out 10 s later
        ev = line_crossings(p, H_LINE, reversal_window=3.0)
        self.assertEqual([d for _, d in ev], ["in", "out"])

    def test_flip_flop_leaves_the_final_direction(self):
        p = path((0.5, 0.3), (0.5, 0.7), (0.5, 0.3), (0.5, 0.7))  # in, out, in within 1 s
        self.assertEqual([d for _, d in line_crossings(p, H_LINE, reversal_window=3.0)], ["in"])

    def test_same_direction_twice_is_not_cancelled(self):
        p = [(0.0, 0.3, 0.3), (0.5, 0.3, 0.7), (1.0, 0.8, 0.3), (1.5, 0.8, 0.7)]  # two separate "in"s
        # second pass needs to come back up first; use two tracks' worth of motion in one path
        ev = line_crossings(p, H_LINE, reversal_window=0.0)
        self.assertEqual([d for _, d in ev], ["in", "out", "in"])
        self.assertEqual([d for _, d in line_crossings(p, H_LINE, reversal_window=0.4)], ["in", "out", "in"])


if __name__ == "__main__":
    unittest.main()

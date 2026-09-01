#!/usr/bin/env python3

import unittest

import numpy as np

from process import postprocess_champion_mask


class IterationZeroPostprocessTest(unittest.TestCase):
    def test_psma_drops_components_below_five_voxels(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[1, 1, 1:5] = 1
        mask[5, 5, 1:6] = 1

        filtered, policy = postprocess_champion_mask(mask, "psma")

        self.assertEqual(policy, "connectivity18_psma5")
        self.assertEqual(int(filtered.sum()), 5)
        self.assertFalse(filtered[1, 1, 1])
        self.assertTrue(filtered[5, 5, 1])

    def test_fdg_keeps_components_at_threshold(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[1, 1:5, 1:7] = 1
        mask[6, 1:6, 1:6] = 1

        filtered, policy = postprocess_champion_mask(mask, "fdg")

        self.assertEqual(policy, "connectivity18_fdg25")
        self.assertEqual(int(filtered.sum()), 25)
        self.assertFalse(filtered[1, 1, 1])
        self.assertTrue(filtered[6, 1, 1])

    def test_unknown_tracer_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported tracer route"):
            postprocess_champion_mask(np.zeros((2, 2, 2), dtype=np.uint8), "other")


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from gaussian_probability_click import _promote_majority_component_removals


class DetectionAwareRemovalTest(unittest.TestCase):
    def setUp(self):
        self.previous = np.zeros((9, 9, 9), dtype=bool)
        self.previous[2:7, 2:7, 2:7] = True
        self.negative = np.zeros_like(self.previous)
        self.negative[3, 3, 3] = True

    def test_promotes_strict_majority_to_whole_component(self):
        proposed = np.zeros_like(self.previous)
        proposed[2:6, 2:7, 2:7] = True

        promoted, count = _promote_majority_component_removals(
            self.previous, proposed, self.negative, connectivity=2
        )

        self.assertEqual(count, 1)
        np.testing.assert_array_equal(promoted, self.previous)

    def test_keeps_minority_edit_local(self):
        proposed = np.zeros_like(self.previous)
        proposed[2:4, 2:7, 2:7] = True

        promoted, count = _promote_majority_component_removals(
            self.previous, proposed, self.negative, connectivity=2
        )

        self.assertEqual(count, 0)
        np.testing.assert_array_equal(promoted, proposed)

    def test_does_not_promote_unprompted_component(self):
        previous = self.previous.copy()
        previous[0, 0, 0] = True
        proposed = np.zeros_like(previous)
        proposed[0, 0, 0] = True

        promoted, count = _promote_majority_component_removals(
            previous, proposed, self.negative, connectivity=2
        )

        self.assertEqual(count, 0)
        np.testing.assert_array_equal(promoted, proposed)


if __name__ == "__main__":
    unittest.main()

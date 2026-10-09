from __future__ import annotations

import unittest

import numpy as np

from pro_innovation.geometry import group_ranked_values, mask_area, mask_length
from pro_innovation.models import MaskInstance


class GeometryTests(unittest.TestCase):
    def test_measurements(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[3, 1:6] = True
        instance = MaskInstance(mask)
        self.assertEqual(mask_area(instance), 5.0)
        self.assertAlmostEqual(mask_length(instance), 5.0)

    def test_group_then_rank(self) -> None:
        groups = group_ranked_values([10.0, 9.5, 5.0, 4.8], relative_tolerance=0.10)
        self.assertEqual([group.count for group in groups], [2, 2])
        self.assertGreater(groups[0].representative, groups[1].representative)


if __name__ == "__main__":
    unittest.main()


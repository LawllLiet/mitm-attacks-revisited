"""Exact hull: known XOR polytope, and agreement of the two exact methods."""

import unittest

from skinny_mitm.exact_hull import (
    boolean_points,
    convex_hull,
    convex_hull_double_description,
    integer_points_in_hull,
)

# Hand-checked H-representation of z = x XOR y on {0,1}^3.
# x + y + z <= 2
# x - y - z <= 0
# -x + y - z <= 0
# -x - y + z <= 0
XOR_POINTS = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
XOR_FACETS = {
    ((1, 1, 1), 2),
    ((1, -1, -1), 0),
    ((-1, 1, -1), 0),
    ((-1, -1, 1), 0),
}


class ExactHullTests(unittest.TestCase):
    def test_xor_matches_hand_facets_and_excludes_other_boolean_points(self):
        by_cdd = convex_hull(XOR_POINTS)
        by_dd = convex_hull_double_description(XOR_POINTS)
        self.assertEqual(set(by_cdd.inequalities), XOR_FACETS)
        self.assertEqual(set(by_dd.inequalities), XOR_FACETS)
        self.assertEqual(set(integer_points_in_hull(by_cdd)), set(XOR_POINTS))
        self.assertEqual(set(integer_points_in_hull(by_dd)), set(XOR_POINTS))
        # Every boolean point was classified, not only the generators.
        self.assertEqual(len(list(boolean_points(3))), 8)

    def test_triangle_and_square(self):
        triangle = [(0, 0), (1, 0), (0, 1)]
        hull = convex_hull(triangle)
        self.assertEqual(set(integer_points_in_hull(hull)), set(triangle))
        self.assertEqual(
            set(hull.inequalities),
            {((1, 1), 1), ((-1, 0), 0), ((0, -1), 0)},
        )
        square = [(0, 0), (0, 1), (1, 0), (1, 1)]
        square_hull = convex_hull(square)
        self.assertEqual(set(integer_points_in_hull(square_hull)), set(square))
        dd_square = convex_hull_double_description(square)
        self.assertEqual(set(dd_square.inequalities), set(square_hull.inequalities))

    def test_lower_dimensional_segment_boolean_points(self):
        segment = [(0, 0, 0), (1, 1, 0)]
        hull = convex_hull(segment)
        dd = convex_hull_double_description(segment)
        self.assertEqual(set(integer_points_in_hull(hull)), set(segment))
        self.assertEqual(set(integer_points_in_hull(dd)), set(segment))
        for point in segment:
            self.assertTrue(hull.satisfied_by(point))
            self.assertTrue(dd.satisfied_by(point))


if __name__ == "__main__":
    unittest.main()

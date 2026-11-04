"""Paper XOR rules against the convex hull, over the whole boolean cube."""

import unittest

from skinny_mitm.colors import BLUE, GRAY, GREEN, RED, VALID_COLORS, WHITE
from skinny_mitm.exact_hull import boolean_points
from skinny_mitm.xor_rules import (
    concrete_multi_points,
    four_xor_hull,
    iter_multi_xor_points,
    iter_two_xor_points,
    multi_xor_allowed,
    multi_xor_hull,
    three_xor_hull,
    two_xor_allowed,
    two_xor_hull,
)


class XorRuleTests(unittest.TestCase):
    def test_two_xor_examples_for_each_bullet(self):
        # 1. white involved -> white, no consumption.
        self.assertTrue(two_xor_allowed(WHITE, BLUE, WHITE, 0, 0))
        self.assertTrue(two_xor_allowed(GREEN, WHITE, WHITE, 0, 0))
        self.assertFalse(two_xor_allowed(WHITE, BLUE, BLUE, 0, 0))
        self.assertFalse(two_xor_allowed(WHITE, GRAY, WHITE, 1, 0))
        # 2. only gray -> gray.
        self.assertTrue(two_xor_allowed(GRAY, GRAY, GRAY, 0, 0))
        self.assertFalse(two_xor_allowed(GRAY, GRAY, GRAY, 1, 0))
        # 3. only blue -> blue free, or gray with one forward DoF.
        self.assertTrue(two_xor_allowed(BLUE, BLUE, BLUE, 0, 0))
        self.assertTrue(two_xor_allowed(BLUE, GRAY, GRAY, 1, 0))
        self.assertFalse(two_xor_allowed(BLUE, BLUE, GRAY, 0, 0))
        self.assertFalse(two_xor_allowed(BLUE, BLUE, GRAY, 1, 1))
        self.assertFalse(two_xor_allowed(BLUE, BLUE, BLUE, 1, 0))
        # 4. only red -> red free, or gray with one backward DoF.
        self.assertTrue(two_xor_allowed(RED, GRAY, RED, 0, 0))
        self.assertTrue(two_xor_allowed(RED, RED, GRAY, 0, 1))
        self.assertFalse(two_xor_allowed(RED, RED, GRAY, 1, 1))
        # 5. mixture or green -> green, and the bullets do not list a consumption.
        self.assertTrue(two_xor_allowed(BLUE, RED, GREEN, 0, 0))
        self.assertTrue(two_xor_allowed(GREEN, GRAY, GREEN, 0, 0))
        self.assertTrue(two_xor_allowed(GREEN, GREEN, GREEN, 0, 0))
        self.assertFalse(two_xor_allowed(BLUE, RED, BLUE, 0, 1))
        self.assertFalse(two_xor_allowed(BLUE, RED, GRAY, 1, 1))
        self.assertFalse(two_xor_allowed(GREEN, BLUE, GREEN, 1, 0))

    def test_multi_xor_examples_including_four_mixed_outcomes(self):
        self.assertTrue(multi_xor_allowed(1, 1, 1, WHITE, 0, 0))
        self.assertFalse(multi_xor_allowed(0, 0, 1, WHITE, 1, 0))
        self.assertTrue(multi_xor_allowed(0, 0, 0, GRAY, 0, 0))
        self.assertFalse(multi_xor_allowed(0, 0, 0, BLUE, 0, 0))
        self.assertTrue(multi_xor_allowed(1, 0, 0, BLUE, 0, 0))
        self.assertTrue(multi_xor_allowed(1, 0, 0, GRAY, 1, 0))
        self.assertFalse(multi_xor_allowed(1, 0, 0, GRAY, 1, 1))
        self.assertTrue(multi_xor_allowed(0, 1, 0, RED, 0, 0))
        self.assertTrue(multi_xor_allowed(0, 1, 0, GRAY, 0, 1))
        # Both colors present: the four listed outcomes, and nothing else.
        self.assertTrue(multi_xor_allowed(1, 1, 0, GREEN, 0, 0))
        self.assertTrue(multi_xor_allowed(1, 1, 0, BLUE, 0, 1))
        self.assertTrue(multi_xor_allowed(1, 1, 0, RED, 1, 0))
        self.assertTrue(multi_xor_allowed(1, 1, 0, GRAY, 1, 1))
        self.assertFalse(multi_xor_allowed(1, 1, 0, GREEN, 1, 0))
        self.assertFalse(multi_xor_allowed(1, 1, 0, WHITE, 0, 0))
        self.assertFalse(multi_xor_allowed(1, 1, 0, BLUE, 0, 0))

    def test_two_xor_hull_accepts_exactly_the_valid_boolean_points(self):
        valid = set(iter_two_xor_points())
        hull = two_xor_hull()
        accepted = []
        for point in boolean_points(11):
            c1, c2, cout = point[0:3], point[3:6], point[6:9]
            cf, cb = point[9], point[10]
            allowed = two_xor_allowed(c1, c2, cout, cf, cb)
            self.assertEqual(hull.satisfied_by(point), allowed)
            if allowed:
                accepted.append(point)
        self.assertEqual(set(accepted), valid)
        # cddlib facet count for this truth table. Not a number taken from the paper.
        self.assertEqual(len(hull.inequalities), 19)
        self.assertEqual(len(hull.equalities), 0)
        self.assertEqual(len(list(boolean_points(11))), 2048)
        # Every valid color pair occurs, so the five rules were not dropped.
        pairs = {(point[0:3], point[3:6]) for point in valid}
        self.assertEqual(pairs, {(c1, c2) for c1 in VALID_COLORS for c2 in VALID_COLORS})

    def test_multi_xor_hull_for_3_and_4_and_all_boolean_points(self):
        valid = set(iter_multi_xor_points())
        self.assertEqual(concrete_multi_points(3), valid)
        self.assertEqual(concrete_multi_points(4), valid)
        hull3 = three_xor_hull()
        hull4 = four_xor_hull()
        self.assertIs(hull3, multi_xor_hull())
        self.assertIs(hull4, multi_xor_hull())
        self.assertEqual(hull3.inequalities, hull4.inequalities)
        self.assertEqual(hull3.equalities, hull4.equalities)
        self.assertEqual(len(hull3.inequalities), 10)
        self.assertEqual(len(hull3.equalities), 1)
        self.assertEqual(len(list(boolean_points(8))), 256)
        accepted = []
        for point in boolean_points(8):
            allowed = multi_xor_allowed(point[0], point[1], point[2], point[3:6], point[6], point[7])
            self.assertEqual(hull3.satisfied_by(point), allowed)
            if allowed:
                accepted.append(point)
        self.assertEqual(set(accepted), valid)

    def test_two_input_mixed_case_is_stricter_than_multi(self):
        # blue XOR red is green-only for two inputs, but the n-XOR rule can cancel.
        self.assertTrue(two_xor_allowed(BLUE, RED, GREEN, 0, 0))
        self.assertFalse(two_xor_allowed(BLUE, RED, GRAY, 1, 1))
        self.assertTrue(multi_xor_allowed(1, 1, 0, GRAY, 1, 1))


if __name__ == "__main__":
    unittest.main()

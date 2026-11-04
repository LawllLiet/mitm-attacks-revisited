"""The user script, the attached affine formulas, and the SB color rule."""

import unittest

from skinny_mitm.exact_hull import boolean_points
from skinny_mitm.sbox_lin import (
    APPROX_LIN_COST,
    EXACT_LIN_COST,
    IMAGE_FORWARD,
    IMAGE_INVERSE,
    INV_SBOX,
    SBOX,
    SLICES,
    apply_affine,
    forward_sbox,
    inverse_sbox,
    iter_sb_points,
    sb_allowed,
    sb_hull,
    x_set,
    y_set,
)


class SboxLinTests(unittest.TestCase):
    def test_user_script_exactly(self):
        # The block below is the user-supplied script, including its asserts.
        S = [0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB, 0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF]
        Inv = [S.index(i) for i in range(16)]
        ys = [{5, 7, 10, 11, 13, 14, 15}, {0, 2, 3, 6, 9}, {1, 4, 8, 12}]
        Cs = [(0, [2, 4, 8, 1]), (3, [14, 5, 7, 15]), (5, [1, 3, 9, 12])]
        printed = []
        for h, (c, cols) in enumerate(Cs):
            def f(y, c=c, cols=cols):
                return (
                    c
                    ^ (0 if not (y & 1) else cols[0])
                    ^ (0 if not (y & 2) else cols[1])
                    ^ (0 if not (y & 4) else cols[2])
                    ^ (0 if not (y & 8) else cols[3])
                )

            assert sorted(f(y) for y in range(16)) == list(range(16))
            assert all(f(y) == Inv[y] for y in ys[h])
            ifwd = [next(y for y in range(16) if f(y) == x) for x in range(16)]
            c2 = ifwd[0]
            cols2 = [ifwd[1 << i] ^ c2 for i in range(4)]
            xs = sorted(f(y) for y in ys[h])
            assert all(S[x] == ifwd[x] for x in xs)
            printed.append((h, tuple(sorted(ys[h])), tuple(xs), c, tuple(cols), c2, tuple(cols2)))
        self.assertEqual(Inv, INV_SBOX)
        self.assertEqual(S, SBOX)
        for h, y_vals, x_vals, c, cols, c2, cols2 in printed:
            rec = SLICES[h]
            self.assertEqual(y_vals, rec["Y"])
            self.assertEqual(x_vals, rec["X"])
            self.assertEqual((c, cols), (rec["inverse_const"], rec["inverse_cols"]))
            self.assertEqual((c2, cols2), (rec["forward_const"], rec["forward_cols"]))

    def test_forward_and_inverse_match_the_image(self):
        self.assertEqual(
            [(rec["forward_const"], rec["forward_cols"]) for rec in SLICES],
            list(IMAGE_FORWARD),
        )
        self.assertEqual(
            [(rec["inverse_const"], rec["inverse_cols"]) for rec in SLICES],
            list(IMAGE_INVERSE),
        )
        # h = 1 is the visible table row: X = 12368, Y = 02369.
        self.assertEqual(SLICES[1]["X"], (1, 2, 3, 6, 8))
        self.assertEqual(SLICES[1]["Y"], (0, 2, 3, 6, 9))

    def test_sets_partition_and_agree_with_sbox(self):
        self.assertEqual(set().union(*(x_set(h) for h in range(3))), set(range(16)))
        self.assertEqual(set().union(*(y_set(h) for h in range(3))), set(range(16)))
        for h in range(3):
            for other in range(h):
                self.assertTrue(x_set(h).isdisjoint(x_set(other)))
                self.assertTrue(y_set(h).isdisjoint(y_set(other)))
            for x in x_set(h):
                self.assertEqual(forward_sbox(h, x), SBOX[x])
            for y in y_set(h):
                self.assertEqual(inverse_sbox(h, y), INV_SBOX[y])
                self.assertEqual(forward_sbox(h, inverse_sbox(h, y)), y)
            # The affine extension is a permutation of the whole nibble space.
            self.assertEqual(sorted(forward_sbox(h, x) for x in range(16)), list(range(16)))
            self.assertEqual(
                apply_affine(0, SLICES[h]["forward_const"], SLICES[h]["forward_cols"]),
                SLICES[h]["forward_const"],
            )

    def test_lin_costs(self):
        self.assertEqual(APPROX_LIN_COST, 0.5)
        self.assertAlmostEqual(EXACT_LIN_COST, 0.25 * __import__("math").log2(3))
        self.assertNotEqual(APPROX_LIN_COST, EXACT_LIN_COST)

    def test_sb_hull_boolean_points(self):
        valid = set(iter_sb_points())
        hull = sb_hull()
        seen = []
        for point in boolean_points(7):
            color_in = point[0:3]
            color_out = point[3:6]
            lin = point[6]
            allowed = sb_allowed(color_in, color_out, lin)
            self.assertEqual(hull.satisfied_by(point), allowed)
            if allowed:
                seen.append(point)
        self.assertEqual(set(seen), valid)
        self.assertEqual(len(hull.inequalities), 6)
        self.assertEqual(len(hull.equalities), 2)
        self.assertEqual(len(list(boolean_points(7))), 128)
        # Non-green passes unchanged; green may stay green only if lin = 1.
        self.assertIn((1, 1, 0, 1, 1, 0, 1), valid)
        self.assertIn((1, 1, 0, 0, 0, 1, 0), valid)
        self.assertNotIn((1, 1, 0, 1, 1, 0, 0), valid)
        self.assertNotIn((1, 0, 0, 1, 0, 0, 1), valid)


if __name__ == "__main__":
    unittest.main()

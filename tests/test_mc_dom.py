"""SKINNY MixColumns rank-DoM, including the cases that disagree with equation (10)."""

import unittest

from skinny_mitm.mc_dom import (
    H_COLUMNS,
    MIXCOLUMNS,
    chen_equation10_dom,
    column_dom,
    dom_table,
    gf2_rank,
    masks_where_rank_exceeds_equation10,
    mixcolumns,
    state_dom,
)
from skinny_mitm.skinny_spec import (
    SPEC_RC_TABLE,
    implied_tk_top,
    lfsr_round_constants,
    lower_tk_indices,
    permute_tweakey,
    shiftrows,
    skinny64_ac_mask,
    upper_art_indices,
)


def _mat_mul(a, b):
    out = []
    for i in range(4):
        row = []
        for j in range(4):
            acc = 0
            for k in range(4):
                acc ^= a[i][k] & b[k][j]
            row.append(acc)
        out.append(tuple(row))
    return tuple(out)


class McDomTests(unittest.TestCase):
    def test_matrix_matches_local_decomposition_and_is_invertible(self):
        for value in range(16):
            column = ((value >> i) & 1 for i in range(4))
            column = tuple(column)
            y = mixcolumns(column)
            # skinny/CPMITM.py MC_forward, at bit level:
            # y3 = x0 xor x2; y0 = y3 xor x3; y1 = x0; y2 = x1 xor x2.
            y3 = column[0] ^ column[2]
            y0 = y3 ^ column[3]
            y1 = column[0]
            y2 = column[1] ^ column[2]
            self.assertEqual(y, (y0, y1, y2, y3))
        # H columns are the columns of M followed by the identity, so M x + y = 0.
        for j in range(4):
            image = tuple(MIXCOLUMNS[row][j] for row in range(4))
            self.assertEqual(H_COLUMNS[j], image)
        for row in range(4):
            self.assertEqual(H_COLUMNS[4 + row], tuple(1 if i == row else 0 for i in range(4)))
        for value in range(16):
            column = tuple((value >> i) & 1 for i in range(4))
            y = mixcolumns(column)
            residual = [0, 0, 0, 0]
            for j, bit in enumerate(column):
                if bit:
                    for row in range(4):
                        residual[row] ^= H_COLUMNS[j][row]
            for row, bit in enumerate(y):
                if bit:
                    for i in range(4):
                        residual[i] ^= H_COLUMNS[4 + row][i]
            self.assertEqual(residual, [0, 0, 0, 0])
        self.assertEqual(gf2_rank(H_COLUMNS), 4)
        # Invert M by solving M m_j = e_j.
        inverse_cols = []
        for j in range(4):
            target = tuple(1 if i == j else 0 for i in range(4))
            # Brute force the 16 possibilities; M is 4 x 4 over GF(2).
            found = None
            for value in range(16):
                column = tuple((value >> i) & 1 for i in range(4))
                if mixcolumns(column) == target:
                    found = column
                    break
            self.assertIsNotNone(found)
            inverse_cols.append(found)
        inverse = tuple(tuple(inverse_cols[j][i] for j in range(4)) for i in range(4))
        identity = tuple(tuple(1 if i == j else 0 for j in range(4)) for i in range(4))
        self.assertEqual(_mat_mul(MIXCOLUMNS, inverse), identity)
        self.assertEqual(_mat_mul(inverse, MIXCOLUMNS), identity)

    def test_known_masks_and_full_table(self):
        table = dom_table()
        self.assertEqual(len(table), 256)
        self.assertEqual(column_dom(0), 4)          # no white cell
        self.assertEqual(table[0], 4)
        self.assertEqual(column_dom(0xFF), 0)       # all white
        self.assertEqual(table[0xFF], 0)
        self.assertTrue(all(0 <= value <= 4 for value in table))
        # Adding a white cell cannot raise DoM.
        for mask in range(256):
            for bit in range(8):
                if (mask >> bit) & 1:
                    continue
                self.assertLessEqual(column_dom(mask | (1 << bit)), column_dom(mask))

    def test_rank_disagrees_with_equation_10_on_dependent_whites(self):
        # x1 and y2 have the same column, so two whites absorb only one equation.
        mask_x1_y2 = (1 << 1) | (1 << 6)
        self.assertEqual(column_dom(mask_x1_y2), 3)
        self.assertEqual(chen_equation10_dom(mask_x1_y2), 2)
        # Four dependent whites: x1, x3, y0, y2. Naive formula gives 0.
        mask_four = (1 << 1) | (1 << 3) | (1 << 4) | (1 << 6)
        self.assertEqual(column_dom(mask_four), 2)
        self.assertEqual(chen_equation10_dom(mask_four), 0)
        self.assertIn(mask_four, masks_where_rank_exceeds_equation10())
        # Rank-DoM is never smaller than equation (10).
        for mask in range(256):
            self.assertGreaterEqual(column_dom(mask), chen_equation10_dom(mask))

    def test_four_column_sum(self):
        self.assertEqual(state_dom((4, 0, 1, 2)), 7)
        self.assertEqual(state_dom((column_dom(0),) * 4), 16)
        self.assertEqual(state_dom((column_dom(0xFF),) * 4), 0)
        with self.assertRaises(ValueError):
            state_dom((1, 2, 3))


class SkinnyGeometryTests(unittest.TestCase):
    def test_shiftrows_and_tweakey_permutation_match_local_code(self):
        identity = list(range(16))
        self.assertEqual(
            shiftrows(identity),
            (0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12),
        )
        self.assertEqual(
            permute_tweakey(identity),
            (9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7),
        )
        self.assertEqual(upper_art_indices(), tuple(range(8)))
        self.assertEqual(lower_tk_indices(), tuple(range(8, 16)))

    def test_lfsr_matches_published_table_and_hypothesis_xor(self):
        self.assertEqual(lfsr_round_constants(), SPEC_RC_TABLE)
        self.assertEqual(len(SPEC_RC_TABLE), 62)
        first = skinny64_ac_mask(1)
        self.assertEqual(first[0], 0x1)
        self.assertEqual(first[4], 0x0)
        self.assertEqual(first[8], 0x2)
        self.assertTrue(all(first[i] == 0 for i in range(16) if i not in (0, 4, 8)))
        pre = [0] * 16
        post = [0] * 16
        pre[0], post[0] = 0xA, 0x3
        ac = [0] * 16
        ac[0] = 0x1
        self.assertEqual(implied_tk_top(pre, post, ac)[0], 0xA ^ 0x3 ^ 0x1)
        zeros = tuple(0 for _ in range(16))
        self.assertEqual(implied_tk_top(pre, post, zeros)[0], 0xA ^ 0x3)


if __name__ == "__main__":
    unittest.main()

"""SKINNY MixColumns and per-column degree of matching.

The matrix is the binary MixColumns matrix of SKINNY (Beierle et al.,
ePrint 2016/660), acting cell-wise over GF(2), not the AES MDS matrix.
It is the same matrix encoded by ``MC_forward`` in ``skinny/CPMITM.py``:

    y0 = x0 XOR x2 XOR x3
    y1 = x0
    y2 = x1 XOR x2
    y3 = x0 XOR x2

Chen et al., ePrint 2024/300, equation (10) define MC-matching by

    d_M^{MC} = sum_Delta max(0, t_Delta - N_ROW),

where t_Delta counts non-white cells in the input column and the output
column together. That expression equals ``4 - n_white`` whenever every set
of white columns has full rank min(4, n_white). SKINNY's matrix is not MDS,
so the rank can be smaller and the degree can be larger.

This module follows the user's white/non-white rule instead of equation (10).
The table is labeled as that rule. The disagreement is recorded in
QUESTIONS.md (Q-DOM).

Formula
-------
Let H be the 4 x 8 coefficient matrix of ``M x + y = 0`` over GF(2), with
columns ordered ``(x0, x1, x2, x3, y0, y1, y2, y3)``. A cell contributes to
matching only when it is non-white: its value is known to one chunk and can
appear in a checkable parity. A white cell is an unknown. Its column is
placed in ``H_white`` and can only absorb equations.

    Gamma(mask) = rank(H) - rank(H_white) = 4 - rank(H_white).

``rank(H) = 4`` because the four output columns are the identity.
The state degree is the sum of the four column contributions. There is no
other matching site in this skeleton.
"""

from __future__ import annotations

from functools import lru_cache

# Rows are outputs y0..y3, columns are inputs x0..x3.
MIXCOLUMNS = (
    (1, 0, 1, 1),
    (1, 0, 0, 0),
    (0, 1, 1, 0),
    (1, 0, 1, 0),
)

N_ROWS = 4
# Column index in H: 0..3 inputs, 4..7 outputs.
H_COLUMNS = (
    (1, 1, 0, 1),  # x0
    (0, 0, 1, 0),  # x1
    (1, 0, 1, 1),  # x2
    (1, 0, 0, 0),  # x3
    (1, 0, 0, 0),  # y0
    (0, 1, 0, 0),  # y1
    (0, 0, 1, 0),  # y2
    (0, 0, 0, 1),  # y3
)


def mixcolumns(column):
    """Apply M to one 4-cell column. Cells are integers; XOR is bitwise."""
    if len(column) != 4:
        raise ValueError("a SKINNY column has 4 cells")
    out = []
    for row in MIXCOLUMNS:
        acc = 0
        for bit, cell in zip(row, column):
            if bit:
                acc ^= cell
        out.append(acc)
    return tuple(out)


def gf2_rank(columns) -> int:
    """Rank over GF(2) of the matrix whose columns are the given 4-bit vectors."""
    cols = [list(col) for col in columns]
    if not cols:
        return 0
    nrows = len(cols[0])
    ncols = len(cols)
    # Row-major working matrix.
    matrix = [[cols[c][r] & 1 for c in range(ncols)] for r in range(nrows)]
    rank = 0
    pivot_row = 0
    for col in range(ncols):
        pivot = None
        for row in range(pivot_row, nrows):
            if matrix[row][col]:
                pivot = row
                break
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        for row in range(nrows):
            if row != pivot_row and matrix[row][col]:
                for k in range(ncols):
                    matrix[row][k] ^= matrix[pivot_row][k]
        pivot_row += 1
        rank += 1
        if pivot_row == nrows:
            break
    return rank


def column_dom(white_mask: int) -> int:
    """Degree of matching of one column.

    ``white_mask`` is an integer in ``0..255``. Bit ``i`` is 1 when cell ``i``
    is white, with ``i`` in the H column order (x0, x1, x2, x3, y0, y1, y2, y3).
    """
    if not 0 <= white_mask <= 255:
        raise ValueError("white_mask must be an 8-bit mask")
    white_cols = [H_COLUMNS[i] for i in range(8) if (white_mask >> i) & 1]
    return N_ROWS - gf2_rank(white_cols)


def chen_equation10_dom(white_mask: int) -> int:
    """Chen et al. equation (10) for one column: max(0, t - 4).

    Kept only so tests can show where it disagrees with ``column_dom``.
    It is not the value used by the model.
    """
    if not 0 <= white_mask <= 255:
        raise ValueError("white_mask must be an 8-bit mask")
    n_known = 8 - bin(white_mask).count("1")
    return max(0, n_known - N_ROWS)


@lru_cache(maxsize=1)
def dom_table() -> tuple:
    """All 256 white-masks. Index is the mask, value is ``column_dom``."""
    return tuple(column_dom(mask) for mask in range(256))


def state_dom(column_doms) -> int:
    """Sum of the four column contributions. This is the whole MC matching degree."""
    doms = tuple(column_doms)
    if len(doms) != 4:
        raise ValueError("SKINNY has 4 MixColumns columns")
    return sum(doms)


def masks_where_rank_exceeds_equation10():
    """Masks whose rank-DoM is strictly larger than equation (10)."""
    return tuple(
        mask for mask in range(256) if column_dom(mask) > chen_equation10_dom(mask)
    )

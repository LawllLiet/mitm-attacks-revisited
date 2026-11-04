"""SKINNY geometry used by the initial-structure skeleton.

Row-major indexing matches ``skinny/CPMITM.py``: cell ``4*row + col``, and
``column(A, j) = [A[j], A[j+4], A[j+8], A[j+12]]``. AddRoundTweakey touches
the first two rows (indices 0..7). The lower two rows are indices 8..15.
ShiftRows and the tweakey permutation PT are the maps in Beierle et al.,
ePrint 2016/660, and they match the local ``ShiftRow_Skinny`` and ``PT``.

Round-constant placement for 4-bit cells (spec, s = 4):

    c0 = rc3 || rc2 || rc1 || rc0   at cell (0, 0)
    c1 = 0 || 0 || rc5 || rc4       at cell (1, 0)
    c2 = 0x2                        at cell (2, 0)

The 6-bit LFSR is initialized to 0 and updated before each round:

    (rc5 || rc4 || rc3 || rc2 || rc1 || rc0)
        -> (rc4 || rc3 || rc2 || rc1 || rc0 || rc5 XOR rc4 XOR 1)

with rc0 the least significant bit. ``SPEC_RC_TABLE`` is the 62 constants
printed in that paper; the LFSR is checked against the table rather than
trusted alone.

The value identity ``TK_top = X XOR Y XOR AC`` is the user's working
hypothesis. It is implemented and flagged. If X is the state immediately
after AddConstants, the specification says ``Y = X XOR TK_top`` on the first
two rows, so AC must not be XORed again. See QUESTIONS.md Q-AC.
"""

from __future__ import annotations

N_ROWS = 4
N_COLS = 4
N_CELLS = 16

# new[i] = old[SHIFTROWS_P[i]]. Second/third/fourth rows rotate right by 1/2/3.
SHIFTROWS_P = (0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12)

# new[i] = old[TWEAKEY_P[i]].
TWEAKEY_P = (9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7)

# Beierle et al., full version, the 62 round-constant bytes (rc0 = LSB).
SPEC_RC_TABLE = (
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
    0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
    0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04,
    0x09, 0x13, 0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A, 0x15, 0x2A, 0x14, 0x28, 0x10, 0x20,
)


def cell_index(row: int, col: int) -> int:
    if not (0 <= row < N_ROWS and 0 <= col < N_COLS):
        raise ValueError("cell outside the 4x4 state")
    return row * N_COLS + col


def upper_art_indices():
    """First two rows: the cells ART actually XORs into."""
    return tuple(cell_index(row, col) for row in (0, 1) for col in range(N_COLS))


def lower_tk_indices():
    """下面两行. Rows 2 and 3 in the local row-major layout."""
    return tuple(cell_index(row, col) for row in (2, 3) for col in range(N_COLS))


def shiftrows(state):
    if len(state) != N_CELLS:
        raise ValueError("state must have 16 cells")
    return tuple(state[SHIFTROWS_P[i]] for i in range(N_CELLS))


def permute_tweakey(tweakey):
    if len(tweakey) != N_CELLS:
        raise ValueError("tweakey must have 16 cells")
    return tuple(tweakey[TWEAKEY_P[i]] for i in range(N_CELLS))


def lfsr_round_constants(n_rounds: int = 62):
    """Update-before-use LFSR. Returns ``n_rounds`` 6-bit integers."""
    rc = 0
    out = []
    for _ in range(n_rounds):
        new_bit = ((rc >> 5) & 1) ^ ((rc >> 4) & 1) ^ 1
        rc = ((rc & 0x1F) << 1) | new_bit
        out.append(rc)
    return tuple(out)


def skinny64_ac_mask(round_1based: int):
    """16-cell AddConstants mask for SKINNY-64. ``round_1based`` is the spec's round number."""
    constants = lfsr_round_constants(62)
    if len(constants) != len(SPEC_RC_TABLE) or constants != SPEC_RC_TABLE:
        raise RuntimeError("LFSR does not reproduce the published SKINNY constant table")
    if not 1 <= round_1based <= len(constants):
        raise ValueError("SKINNY publishes constants for rounds 1..62")
    rc = constants[round_1based - 1]
    mask = [0] * N_CELLS
    mask[cell_index(0, 0)] = rc & 0xF
    mask[cell_index(1, 0)] = (rc >> 4) & 0x3
    mask[cell_index(2, 0)] = 0x2
    return tuple(mask)


def implied_tk_top(pre_art, post_art, ac_mask):
    """User hypothesis, flagged: TK_top = X XOR Y XOR AC on the first two rows.

    ``pre_art``, ``post_art`` and ``ac_mask`` are 16-cell sequences.
    Only the eight cells of rows 0 and 1 are returned, in row-major order.
    This does not decide whether AC sits between X and Y; pass a zero mask
    if X is already the post-AC state. See QUESTIONS.md Q-AC.
    """
    if not (len(pre_art) == len(post_art) == len(ac_mask) == N_CELLS):
        raise ValueError("X, Y and AC must each have 16 cells")
    return tuple(
        (pre_art[i] ^ post_art[i] ^ ac_mask[i]) & 0xF
        for i in upper_art_indices()
    )

"""SKINNY-64 4-bit S-box and the three affine slices used by LIN.

The slices below are derived by the same procedure as the user-supplied
script (not by inventing further decompositions). Bit ``x0`` is the least
significant bit. A term ``8*x0`` means "XOR the constant 8 if bit 0 is set";
it is not multiplication in GF(16).

Chen et al., ePrint 2024/300, Section 4.2, Propagations through SubBytes:

"When the input byte is not green, the color of the output byte is identical
to that of the input byte. When the input byte is green, the output byte is
either white by default or green with one cost of linearization (dL
incremented by one)."

The per-S-box weight of that one cost is NOT the AES value. The user fixed
it at (1/4)*log2(3), with a coarse Gurobi charge of 1/2.
"""

from __future__ import annotations

import math
from functools import lru_cache

from skinny_mitm.colors import BLUE, GRAY, GREEN, RED, VALID_COLORS, WHITE
from skinny_mitm.exact_hull import HRep, convex_hull

# User-approved coarse binary charge. Not equal to the exact fraction below.
APPROX_LIN_COST = 0.5
# Exact cost of linearizing one SKINNY-64 S-box, in cell units: (1/4)*log2(3).
EXACT_LIN_COST = 0.25 * math.log2(3)

# SKINNY-64 4-bit S-box, as given by the user.
SBOX = [0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB, 0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF]
INV_SBOX = [SBOX.index(i) for i in range(16)]

# Inverse affine data from the user script: (constant, (col_bit0, ... col_bit3)).
_USER_INV = (
    (0, (2, 4, 8, 1)),
    (3, (14, 5, 7, 15)),
    (5, (1, 3, 9, 12)),
)
# Y_h from the user script. X_h is derived, not typed in from the cropped figure.
_USER_Y = (
    frozenset({5, 7, 10, 11, 13, 14, 15}),
    frozenset({0, 2, 3, 6, 9}),
    frozenset({1, 4, 8, 12}),
)


def apply_affine(value: int, constant: int, columns) -> int:
    """Evaluate ``constant XOR columns[i]`` for each set bit ``i`` (LSB = bit 0)."""
    out = constant
    for i, col in enumerate(columns):
        if (value >> i) & 1:
            out ^= col
    return out & 0xF


def derive_slices():
    """Run the user's derivation and return one record per h in {0, 1, 2}.

    Each record is a dict with Y, X, inverse (const, cols), forward (const, cols).
    """
    records = []
    for h, (constant, columns) in enumerate(_USER_INV):
        def inverse_map(y, constant=constant, columns=columns):
            return apply_affine(y, constant, columns)

        images = [inverse_map(y) for y in range(16)]
        if sorted(images) != list(range(16)):
            raise RuntimeError(f"inverse slice h={h} is not a permutation")
        if any(inverse_map(y) != INV_SBOX[y] for y in _USER_Y[h]):
            raise RuntimeError(f"inverse slice h={h} disagrees with Inv on Y_h")
        forward_table = [next(y for y in range(16) if inverse_map(y) == x) for x in range(16)]
        forward_const = forward_table[0]
        forward_cols = tuple(forward_table[1 << i] ^ forward_const for i in range(4))
        x_set = tuple(sorted(inverse_map(y) for y in _USER_Y[h]))
        if any(SBOX[x] != forward_table[x] for x in x_set):
            raise RuntimeError(f"forward slice h={h} disagrees with S on X_h")
        records.append(
            {
                "h": h,
                "Y": tuple(sorted(_USER_Y[h])),
                "X": x_set,
                "inverse_const": constant,
                "inverse_cols": tuple(columns),
                "forward_const": forward_const,
                "forward_cols": forward_cols,
                "forward_table": tuple(forward_table),
            }
        )
    return tuple(records)


SLICES = derive_slices()

# Image in the user attachment, h = 0, 1, 2. h = 0 has constant 0, written as 8*x0 ⊕ ...
IMAGE_FORWARD = (
    (0x0, (0x8, 0x1, 0x2, 0x4)),
    (0xF, (0x9, 0x6, 0xB, 0xC)),
    (0xC, (0x1, 0x3, 0xD, 0x5)),
)
IMAGE_INVERSE = (
    (0x0, (0x2, 0x4, 0x8, 0x1)),
    (0x3, (0xE, 0x5, 0x7, 0xF)),
    (0x5, (0x1, 0x3, 0x9, 0xC)),
)


def forward_sbox(h: int, x: int) -> int:
    rec = SLICES[h]
    return apply_affine(x, rec["forward_const"], rec["forward_cols"])


def inverse_sbox(h: int, y: int) -> int:
    rec = SLICES[h]
    return apply_affine(y, rec["inverse_const"], rec["inverse_cols"])


def x_set(h: int) -> frozenset:
    return frozenset(SLICES[h]["X"])


def y_set(h: int) -> frozenset:
    return frozenset(SLICES[h]["Y"])


# SubBytes color rule. Variable order:
# (b_in, r_in, w_in, b_out, r_out, w_out, lin)
SB_VAR_NAMES = ("b_in", "r_in", "w_in", "b_out", "r_out", "w_out", "lin")


def sb_allowed(color_in, color_out, lin: int) -> bool:
    """Paper SB-rule. ``lin`` is the 0/1 linearization indicator, not its weight.

    Non-green input: output color equals input color and lin = 0.
    Green input: output is white with lin = 0, or green with lin = 1.
    """
    if tuple(color_in) not in VALID_COLORS or tuple(color_out) not in VALID_COLORS:
        return False
    if lin not in (0, 1):
        return False
    if tuple(color_in) != GREEN:
        return tuple(color_out) == tuple(color_in) and lin == 0
    if lin == 0:
        return tuple(color_out) == WHITE
    return tuple(color_out) == GREEN


def iter_sb_points():
    for color_in in VALID_COLORS:
        for color_out in VALID_COLORS:
            for lin in (0, 1):
                if sb_allowed(color_in, color_out, lin):
                    yield tuple(color_in) + tuple(color_out) + (lin,)


@lru_cache(maxsize=1)
def sb_hull() -> HRep:
    return convex_hull(iter_sb_points())


def sb_rules_cover_five_colors() -> bool:
    """Sanity: each non-green color is a fixed point of the rule, green has two exits."""
    for color in (GRAY, BLUE, RED, WHITE):
        if not sb_allowed(color, color, 0):
            return False
        if sb_allowed(color, GREEN, 0) or sb_allowed(color, color, 1):
            return False
    return sb_allowed(GREEN, WHITE, 0) and sb_allowed(GREEN, GREEN, 1)

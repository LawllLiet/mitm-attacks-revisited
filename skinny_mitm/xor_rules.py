"""XOR color rules from Chen et al., ePrint 2024/300, Section 4.2.

Two-input XOR, five bullets (paper text):

- When the input involves a white byte, the output is white.
- When the input contains only gray bytes, the output is gray.
- When the input contains only blue bytes, the output is either blue with no
  consumption of DoF or gray consuming one DoF from the forward chunk.
- When the input contains only red bytes, the output is either red with no
  DoF consumption or gray consuming one DoF from the backward chunk.
- When the input is a mixture of red and blue bytes or involves green bytes,
  the output is green.

n-input XOR (paper text). The extracted ePrint text has five top-level
bullets; the last bullet has four outcomes. The user referred to "six rules".
No sixth top-level bullet was found, so none was invented. See QUESTIONS.md
Q-SIX.

- When the input involves a white byte, the output is white.
- When the input contains only gray bytes, the output is gray.
- When the input contains only blue bytes, the output is either blue with no
  DoF consumed or gray with 1 DoF consumed from the forward chunk.
- When the input contains only red bytes, the output is either red with no
  DoF consumed or gray with 1 DoF consumed from the backward chunk.
- When the input contains both red and blue bytes, the output is one of:
  green, with no consumption of DoF;
  blue, with 1 DoF consumed from the backward chunk;
  red, with 1 DoF consumed from the forward chunk;
  gray, with 1 DoF consumed from each forward and backward chunk.

The n-input constraints are on (P_b, P_r, P_w) with
P_b = max b, P_r = max r, P_w = max w (paper equation (8)), the output
color, and two DoF-consumption bits. The sourcing function is not modeled.

Reading used for bits the bullets do not mention (flagged in Q-AUX): a
consumption bit is 1 only when a bullet explicitly consumes that chunk.
White takes priority over every later bullet, because "involves a white
byte" does not require the other bytes to be non-white. Gray does not set
P_b or P_r, so "only blue" includes blue/gray mixtures and excludes green.
"""

from __future__ import annotations

from functools import lru_cache

from skinny_mitm.colors import BLUE, GRAY, GREEN, RED, VALID_COLORS, WHITE
from skinny_mitm.exact_hull import HRep, convex_hull

TWO_XOR_VAR_NAMES = (
    "b1", "r1", "w1",
    "b2", "r2", "w2",
    "b_out", "r_out", "w_out",
    "cf", "cb",
)
MULTI_XOR_VAR_NAMES = (
    "P_b", "P_r", "P_w",
    "b_out", "r_out", "w_out",
    "cf", "cb",
)


def _pack(color, cf: int, cb: int):
    return tuple(color) + (cf, cb)


def two_xor_allowed(color1, color2, color_out, cf: int, cb: int) -> bool:
    """Return whether the two-input rule lists this tuple."""
    if any(tuple(c) not in VALID_COLORS for c in (color1, color2, color_out)):
        return False
    if cf not in (0, 1) or cb not in (0, 1):
        return False
    c1, c2, cout = tuple(color1), tuple(color2), tuple(color_out)
    # Bullet 1. White wins over green / red / blue.
    if c1 == WHITE or c2 == WHITE:
        return cout == WHITE and cf == 0 and cb == 0
    # Bullet 2.
    if c1 == GRAY and c2 == GRAY:
        return cout == GRAY and cf == 0 and cb == 0
    only_blue = c1 in (BLUE, GRAY) and c2 in (BLUE, GRAY) and (c1 == BLUE or c2 == BLUE)
    only_red = c1 in (RED, GRAY) and c2 in (RED, GRAY) and (c1 == RED or c2 == RED)
    # Bullet 3.
    if only_blue:
        return (cout == BLUE and cf == 0 and cb == 0) or (cout == GRAY and cf == 1 and cb == 0)
    # Bullet 4.
    if only_red:
        return (cout == RED and cf == 0 and cb == 0) or (cout == GRAY and cf == 0 and cb == 1)
    # Bullet 5: mixture of red and blue, or any green.
    return cout == GREEN and cf == 0 and cb == 0


def iter_two_xor_points():
    for c1 in VALID_COLORS:
        for c2 in VALID_COLORS:
            for cout in VALID_COLORS:
                for cf in (0, 1):
                    for cb in (0, 1):
                        if two_xor_allowed(c1, c2, cout, cf, cb):
                            yield tuple(c1) + tuple(c2) + tuple(cout) + (cf, cb)


def multi_xor_allowed(p_b: int, p_r: int, p_w: int, color_out, cf: int, cb: int) -> bool:
    """n-XOR rule on the aggregate bits. The same predicate is used for n=3 and n=4."""
    if tuple(color_out) not in VALID_COLORS:
        return False
    if cf not in (0, 1) or cb not in (0, 1):
        return False
    if p_b not in (0, 1) or p_r not in (0, 1) or p_w not in (0, 1):
        return False
    cout = tuple(color_out)
    # Bullet 1.
    if p_w == 1:
        return cout == WHITE and cf == 0 and cb == 0
    # Bullet 2.
    if p_b == 0 and p_r == 0:
        return cout == GRAY and cf == 0 and cb == 0
    # Bullet 3.
    if p_b == 1 and p_r == 0:
        return (cout == BLUE and cf == 0 and cb == 0) or (cout == GRAY and cf == 1 and cb == 0)
    # Bullet 4.
    if p_b == 0 and p_r == 1:
        return (cout == RED and cf == 0 and cb == 0) or (cout == GRAY and cf == 0 and cb == 1)
    # Bullet 5, four outcomes. Green inputs set both P_b and P_r via max.
    return (
        (cout == GREEN and cf == 0 and cb == 0)
        or (cout == BLUE and cf == 0 and cb == 1)
        or (cout == RED and cf == 1 and cb == 0)
        or (cout == GRAY and cf == 1 and cb == 1)
    )


def aggregate_color_bits(colors) -> tuple[int, int, int]:
    """Paper equation (8): componentwise max of the (b, r, w) codes."""
    p_b = max(c[0] for c in colors)
    p_r = max(c[1] for c in colors)
    p_w = max(c[2] for c in colors)
    return p_b, p_r, p_w


def iter_multi_xor_points():
    for p_b in (0, 1):
        for p_r in (0, 1):
            for p_w in (0, 1):
                for cout in VALID_COLORS:
                    for cf in (0, 1):
                        for cb in (0, 1):
                            if multi_xor_allowed(p_b, p_r, p_w, cout, cf, cb):
                                yield (p_b, p_r, p_w) + tuple(cout) + (cf, cb)


def concrete_multi_points(n_inputs: int):
    """Points realized by some coloring of ``n_inputs`` cells.

    The output side is still the aggregate rule: every concrete coloring with
    a given (P_b, P_r, P_w) allows the same (output, cf, cb) pairs.
    """
    if n_inputs < 1:
        raise ValueError("n_inputs must be positive")
    found = set()

    def rec(i, acc):
        if i == n_inputs:
            p_b, p_r, p_w = aggregate_color_bits(acc)
            for cout in VALID_COLORS:
                for cf in (0, 1):
                    for cb in (0, 1):
                        if multi_xor_allowed(p_b, p_r, p_w, cout, cf, cb):
                            found.add((p_b, p_r, p_w) + tuple(cout) + (cf, cb))
            return
        for color in VALID_COLORS:
            acc.append(color)
            rec(i + 1, acc)
            acc.pop()

    rec(0, [])
    return found


@lru_cache(maxsize=1)
def two_xor_hull() -> HRep:
    return convex_hull(iter_two_xor_points())


@lru_cache(maxsize=1)
def multi_xor_hull() -> HRep:
    """H-representation shared by the 3-xor and 4-xor aggregate rules."""
    return convex_hull(iter_multi_xor_points())


def three_xor_hull() -> HRep:
    return multi_xor_hull()


def four_xor_hull() -> HRep:
    return multi_xor_hull()


def two_xor_valid_points() -> frozenset:
    return frozenset(iter_two_xor_points())


def multi_xor_valid_points() -> frozenset:
    return frozenset(iter_multi_xor_points())

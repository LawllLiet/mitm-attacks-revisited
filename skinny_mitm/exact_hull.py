"""Exact V-to-H convex hull for small 0-1 truth tables.

The double-description method is run with rational arithmetic
(:class:`fractions.Fraction`). Float convex hulls are not used.

A polytope that is not full-dimensional is reduced to its affine hull.
Equalities are returned separately from inequalities.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Iterable, Sequence


Vec = tuple[Fraction, ...]
IntVec = tuple[int, ...]


def _frac(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    return Fraction(v)


def rref(rows: Sequence[Sequence]) -> tuple[list[list[Fraction]], list[int]]:
    """Exact reduced row echelon form. Returns (matrix, pivot columns)."""
    A = [[_frac(x) for x in row] for row in rows]
    if not A:
        return [], []
    m = len(A)
    n = len(A[0])
    pivots: list[int] = []
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        scale = A[r][c]
        A[r] = [v / scale for v in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                factor = A[i][c]
                A[i] = [A[i][j] - factor * A[r][j] for j in range(n)]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots


def matrix_rank(rows: Sequence[Sequence]) -> int:
    if not rows:
        return 0
    _, pivots = rref(rows)
    return len(pivots)


def _mat_inverse(M: Sequence[Sequence]) -> list[list[Fraction]]:
    n = len(M)
    A = [
        [_frac(x) for x in row] + [Fraction(int(i == j)) for j in range(n)]
        for i, row in enumerate(M)
    ]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if A[r][col] != 0:
                piv = r
                break
        if piv is None:
            raise ValueError("singular matrix")
        A[col], A[piv] = A[piv], A[col]
        scale = A[col][col]
        A[col] = [v / scale for v in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                factor = A[r][col]
                A[r] = [A[r][j] - factor * A[col][j] for j in range(2 * n)]
    return [row[n:] for row in A]


def _dot(a: Sequence, b: Sequence) -> Fraction:
    return sum((_frac(x) * _frac(y) for x, y in zip(a, b)), Fraction(0))


def _primitive_int(nums: Sequence[Fraction], flip_if_first_negative: bool = False) -> IntVec:
    """Clear denominators and divide by the positive gcd.

    If ``flip_if_first_negative`` is set, multiply by -1 so the first nonzero
    entry is positive. Use that for equalities only.
    """
    fracs = [_frac(x) for x in nums]
    if all(x == 0 for x in fracs):
        return tuple(0 for _ in fracs)
    lcm = 1
    for f in fracs:
        den = f.denominator
        lcm = lcm * den // gcd(lcm, den)
    ints = [int(f * lcm) for f in fracs]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    ints = [v // g for v in ints]
    if flip_if_first_negative:
        for v in ints:
            if v < 0:
                ints = [-x for x in ints]
                break
            if v > 0:
                break
    return tuple(ints)


def _normalize_inequality(coeffs: Sequence, rhs) -> tuple[IntVec, int]:
    packed = _primitive_int([*_coeffs(coeffs), _frac(rhs)])
    return packed[:-1], packed[-1]


def _coeffs(coeffs: Sequence) -> list[Fraction]:
    return [_frac(x) for x in coeffs]


@dataclass(frozen=True)
class HRep:
    """H-representation of conv(points) in the original coordinates.

    ``inequalities`` are ``a · x <= rhs``.
    ``equalities`` are ``a · x == rhs``.
    """

    dimension: int
    affine_dimension: int
    equalities: tuple[tuple[IntVec, int], ...]
    inequalities: tuple[tuple[IntVec, int], ...]

    def satisfied_by(self, point: Sequence[int]) -> bool:
        for a, rhs in self.equalities:
            if sum(ai * xi for ai, xi in zip(a, point)) != rhs:
                return False
        for a, rhs in self.inequalities:
            if sum(ai * xi for ai, xi in zip(a, point)) > rhs:
                return False
        return True


def _dedup_points(points: Iterable[Sequence[int]]) -> list[tuple[int, ...]]:
    uniq = sorted({tuple(int(x) for x in p) for p in points})
    if not uniq:
        raise ValueError("convex hull of an empty set is undefined")
    dim = len(uniq[0])
    if any(len(p) != dim for p in uniq):
        raise ValueError("points have different dimensions")
    return uniq


def _affine_rank(points: Sequence[Sequence[int]]) -> int:
    if len(points) <= 1:
        return 0
    p0 = points[0]
    dirs = [[p[i] - p0[i] for i in range(len(p0))] for p in points[1:]]
    return matrix_rank(dirs)


def _keep_facets(points, inequalities, affine_dim):
    """Drop inequalities whose tight face is not a facet."""
    kept = []
    for a, rhs in inequalities:
        tight = [p for p in points if sum(ai * pi for ai, pi in zip(a, p)) == rhs]
        if _affine_rank(tight) == affine_dim - 1:
            kept.append((a, rhs))
    return kept


def _independent_directions(points: Sequence[tuple[int, ...]]):
    p0 = points[0]
    dim = len(p0)
    basis_cols: list[list[Fraction]] = []
    for p in points[1:]:
        direction = [Fraction(p[j] - p0[j]) for j in range(dim)]
        if matrix_rank(basis_cols + [direction]) > len(basis_cols):
            basis_cols.append(direction)
    return p0, basis_cols


def _coordinates(
    points: Sequence[tuple[int, ...]],
    p0: Sequence[int],
    basis_cols: Sequence[Sequence[Fraction]],
) -> list[tuple[Fraction, ...]]:
    """Express ``x = p0 + D t`` and return each ``t``."""
    k = len(basis_cols)
    if k == 0:
        return [tuple()] * len(points)
    # D is dim x k, columns are basis directions. Solve least squares which
    # is exact because every point lies in the affine hull.
    D = [[basis_cols[j][i] for j in range(k)] for i in range(len(p0))]
    # Gram matrix G = D^T D (k x k)
    G = [
        [sum(D[r][a] * D[r][b] for r in range(len(p0))) for b in range(k)]
        for a in range(k)
    ]
    Ginv = _mat_inverse(G)
    coords = []
    for p in points:
        diff = [Fraction(p[i] - p0[i]) for i in range(len(p0))]
        Dt_diff = [sum(D[r][a] * diff[r] for r in range(len(p0))) for a in range(k)]
        t = [
            sum(Ginv[a][b] * Dt_diff[b] for b in range(k))
            for a in range(k)
        ]
        coords.append(tuple(t))
    return coords


def _affine_equalities(p0, basis_cols) -> list[tuple[IntVec, int]]:
    """n · (x - p0) = 0 for n orthogonal to every direction."""
    dim = len(p0)
    if not basis_cols:
        eqs = []
        for i in range(dim):
            a = tuple(1 if j == i else 0 for j in range(dim))
            eqs.append((a, int(p0[i])))
        return eqs
    # Nullspace of the direction span: n such that every basis column is orthogonal to n.
    constraint = [[basis_cols[j][i] for i in range(dim)] for j in range(len(basis_cols))]
    A, pivots = rref(constraint)
    pivot_row = {c: i for i, c in enumerate(pivots)}
    free = [c for c in range(dim) if c not in pivot_row]
    eqs = []
    for f in free:
        vec = [Fraction(0)] * dim
        vec[f] = Fraction(1)
        for pcol, row_i in pivot_row.items():
            vec[pcol] = -A[row_i][f]
        packed = list(_primitive_int(vec + [sum(vec[i] * p0[i] for i in range(dim))], True))
        a, rhs = tuple(packed[:-1]), packed[-1]
        eqs.append((a, rhs))
    return eqs


def _dd_full_dimensional(rays: list[tuple[Fraction, ...]]) -> list[tuple[Fraction, ...]]:
    """H-representation of a full-dimensional pointed cone pos(rays).

    Each returned vector ``u`` satisfies ``u · r >= 0`` for every ray.
    """
    n = len(rays[0])
    if any(len(r) != n for r in rays):
        raise ValueError("ragged rays")
    # Initial simplex: n linearly independent rays.
    chosen: list[int] = []
    current: list[tuple[Fraction, ...]] = []
    for i, ray in enumerate(rays):
        if matrix_rank(current + [ray]) > len(current):
            current.append(ray)
            chosen.append(i)
            if len(current) == n:
                break
    if len(current) != n:
        raise ValueError("rays do not span a full-dimensional cone")
    # Columns are the basis rays.
    M = [[current[j][i] for j in range(n)] for i in range(n)]
    Minv = _mat_inverse(M)
    ineqs: list[tuple[Fraction, ...]] = [tuple(row) for row in Minv]
    ray_set: list[tuple[Fraction, ...]] = list(current)
    for i, ray in enumerate(rays):
        if i in chosen:
            continue
        ineqs, ray_set = _dd_add_ray(ineqs, ray_set, ray)
    return ineqs


def _dd_add_ray(ineqs, rays, new_ray):
    signs = []
    dots = []
    for u in ineqs:
        val = _dot(u, new_ray)
        dots.append(val)
        if val > 0:
            signs.append(1)
        elif val < 0:
            signs.append(-1)
        else:
            signs.append(0)
    if all(s >= 0 for s in signs):
        return ineqs, rays
    pos = [i for i, s in enumerate(signs) if s > 0]
    neg = [i for i, s in enumerate(signs) if s < 0]
    zero = [i for i, s in enumerate(signs) if s == 0]
    n = len(new_ray)
    new_ineqs = [ineqs[i] for i in zero]
    new_ineqs.extend(ineqs[i] for i in pos)
    for i in pos:
        for j in neg:
            if not _facets_adjacent(ineqs[i], ineqs[j], rays, n):
                continue
            # u = (-(u_neg·r)) u_pos + (u_pos·r) u_neg, so u·r = 0 and
            # u·s >= 0 for every old ray s (both facet dots are nonnegative).
            di = dots[i]
            dj = dots[j]
            u_new = tuple((-dj) * ineqs[i][t] + di * ineqs[j][t] for t in range(n))
            if all(c == 0 for c in u_new):
                continue
            prim = _primitive_int(u_new)
            new_ineqs.append(tuple(Fraction(c) for c in prim))
    # Dedup.
    uniq = []
    seen = set()
    for u in new_ineqs:
        key = _primitive_int(u)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tuple(Fraction(c) for c in key))
    return uniq, rays + [tuple(_frac(x) for x in new_ray)]


def _facets_adjacent(u, v, rays, n: int) -> bool:
    common = [r for r in rays if _dot(u, r) == 0 and _dot(v, r) == 0]
    return matrix_rank(common) == n - 2


def _lift_inequality(a_t, rhs_t, p0, basis_cols) -> tuple[IntVec, int]:
    """Lift ``a_t · t <= rhs_t`` through ``x = p0 + D t``."""
    k = len(basis_cols)
    dim = len(p0)
    if k == 0:
        raise ValueError("nothing to lift")
    D = [[basis_cols[j][i] for j in range(k)] for i in range(dim)]
    G = [
        [sum(D[r][a] * D[r][b] for r in range(dim)) for b in range(k)]
        for a in range(k)
    ]
    Ginv = _mat_inverse(G)
    # c = D G^{-1} a_t, so D^T c = a_t.
    Ginv_a = [sum(Ginv[a][b] * a_t[b] for b in range(k)) for a in range(k)]
    c = [sum(D[r][a] * Ginv_a[a] for a in range(k)) for r in range(dim)]
    rhs = _frac(rhs_t) + sum(c[i] * p0[i] for i in range(dim))
    return _normalize_inequality(c, rhs)


def convex_hull_double_description(points: Iterable[Sequence[int]]) -> HRep:
    """Pure-Python exact H-representation (double description, rational arithmetic).

    Correct on the tiny polytopes in the unit tests. Prefer :func:`convex_hull`,
    which calls ``cddexec_gmp``, for the XOR truth tables: the intermediate
    cones of this implementation can become very large.
    """
    pts = _dedup_points(points)
    dim = len(pts[0])
    p0, basis_cols = _independent_directions(pts)
    k = len(basis_cols)
    equalities = tuple(_affine_equalities(p0, basis_cols))
    if k == 0:
        return HRep(dim, 0, equalities, tuple())
    ts = _coordinates(pts, p0, basis_cols)
    # Homogenize t -> (1, t) in R^{k+1}.
    rays = [tuple([Fraction(1), *t]) for t in ts]
    cone_ineqs = _dd_full_dimensional(rays)
    inequalities = []
    seen = set()
    for u in cone_ineqs:
        # u0 + u_t · t >= 0  <=>  (-u_t) · t <= u0.
        u0 = u[0]
        a_t = tuple(-u[i + 1] for i in range(k))
        ineq = _lift_inequality(a_t, u0, p0, basis_cols)
        if ineq in seen:
            continue
        # Drop 0 <= positive and 0 <= 0.
        if all(c == 0 for c in ineq[0]):
            if ineq[1] < 0:
                raise RuntimeError("hull inequality 0 <= negative; DD failed")
            continue
        seen.add(ineq)
        inequalities.append(ineq)
    # Pair opposite inequalities into equalities.
    ineq_set = set(inequalities)
    consumed = set()
    extra_eq = []
    kept = []
    for a, rhs in inequalities:
        if (a, rhs) in consumed:
            continue
        neg = (tuple(-c for c in a), -rhs)
        if neg in ineq_set and neg != (a, rhs):
            consumed.add((a, rhs))
            consumed.add(neg)
            packed = _primitive_int([*a, rhs], True)
            extra_eq.append((packed[:-1], packed[-1]))
        else:
            kept.append((a, rhs))
    all_eq = tuple(dict.fromkeys(equalities + tuple(extra_eq)))
    facets = _keep_facets(pts, kept, k)
    hull = HRep(dim, k, all_eq, tuple(facets))
    for p in pts:
        if not hull.satisfied_by(p):
            raise RuntimeError(f"generator {p} is cut off by the computed hull")
    return hull


def _cdd_executable() -> str:
    exe = shutil.which("cddexec_gmp")
    if exe:
        return exe
    candidate = os.path.join(sys.prefix, "bin", "cddexec_gmp")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    raise RuntimeError(
        "cddexec_gmp was not found. The exact hull uses cddlib's GMP binary "
        "from the auto_sat environment."
    )


def _parse_cdd_hrep(text: str, dim: int) -> HRep:
    """Parse cddexec_gmp --rep output.

    A row ``b c1 c2 ... cd`` means ``b + c·x >= 0``, i.e. ``(-c)·x <= b``.
    Rows listed in the ``linearity`` directive are equalities.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("*")]
    linearity: set[int] = set()
    rows: list[list[Fraction]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("linearity"):
            parts = ln.split()
            # linearity <count> <1-based indices>...
            count = int(parts[1])
            idxs = [int(x) for x in parts[2:2 + count]]
            linearity.update(idxs)
            i += 1
            continue
        if ln == "begin":
            i += 1
            header = lines[i].split()
            nrows = int(header[0])
            ncols = int(header[1])
            if ncols != dim + 1:
                raise RuntimeError(f"cdd returned {ncols} columns, expected {dim + 1}")
            i += 1
            for _ in range(nrows):
                rows.append([Fraction(tok) for tok in lines[i].split()])
                i += 1
            if i >= len(lines) or lines[i] != "end":
                raise RuntimeError("cdd output is missing 'end'")
            break
        i += 1
    if not rows and dim >= 0:
        # A 0-dimensional... we always get rows. Empty would be a bug unless no points.
        raise RuntimeError("cdd output contained no H-representation")
    equalities = []
    inequalities = []
    for idx, row in enumerate(rows, start=1):
        b = row[0]
        c = row[1:]  # b + c·x >= 0
        a = tuple(-coef for coef in c)  # a·x <= b
        if idx in linearity:
            packed = _primitive_int([*a, b], True)
            equalities.append((packed[:-1], packed[-1]))
            continue
        if all(coef == 0 for coef in a):
            if b < 0:
                raise RuntimeError("cdd produced the contradictory inequality 0 <= negative")
            continue
        inequalities.append(_normalize_inequality(a, b))
    # Affine dimension = ambient dim - equality rank.
    if equalities:
        aff = dim - matrix_rank([list(a) for a, _ in equalities])
    else:
        aff = dim
    equalities = tuple(sorted(set(equalities)))
    inequalities = tuple(sorted(set(inequalities)))
    return HRep(dim, aff, equalities, inequalities)


def convex_hull(points: Iterable[Sequence[int]]) -> HRep:
    """Exact H-representation via cddlib (``cddexec_gmp``, rational arithmetic)."""
    pts = _dedup_points(points)
    dim = len(pts[0])
    lines = [
        "V-representation",
        "begin",
        f"{len(pts)} {dim + 1} integer",
    ]
    for p in pts:
        lines.append("1 " + " ".join(str(int(x)) for x in p))
    lines += ["end", ""]
    proc = subprocess.run(
        [_cdd_executable(), "--rep"],
        input="\n".join(lines),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cddexec_gmp failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    hull = _parse_cdd_hrep(proc.stdout, dim)
    for p in pts:
        if not hull.satisfied_by(p):
            raise RuntimeError(f"generator {p} is cut off by the cdd hull")
    return hull


def boolean_points(n: int):
    for mask in range(1 << n):
        yield tuple((mask >> i) & 1 for i in range(n))


def integer_points_in_hull(hull: HRep) -> list[tuple[int, ...]]:
    return [p for p in boolean_points(hull.dimension) if hull.satisfied_by(p)]

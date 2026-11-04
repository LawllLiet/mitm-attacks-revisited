"""Gurobi skeleton for one SKINNY-64/192 (z = 3) initial-structure position.

Locked by the user
------------------
- Five colors only: blue (1,0,0), red (0,1,0), white (0,0,1), gray (0,0,0),
  green (1,1,0). The codes (1,0,1), (0,1,1) and (1,1,1) are forbidden.
- Blue adds one forward degree. Red adds one backward degree. White, gray
  and green add zero. Gray is legal on an encryption cell. Green is legal
  on a combined-tweakey cell and illegal on an encryption cell.
- z = 3. On cells 0..7 the value equation is state XOR TK1 XOR TK2 XOR TK3.
  The color model keeps one combined tweakey. AddConstants is a public
  constant and is not a color variable: XOR with a constant preserves
  (b, r, w). Each top cell therefore obeys the two-input XOR rule
  post = pre XOR tk_top. Those eight top cells are linked, not key-DoF
  variables. The n-XOR hull is not used here, because the three lanes are
  not separate color variables; see QUESTIONS.md.
- Key DoF is exactly the eight combined cells 8..15. Not three lanes, and
  not the top two rows.
- Objective is the constant 0. DoM is the rank formula, summed over four
  columns. MC colors follow the SKINNY matrix cell by cell. No feed-forward
  match and no guess-and-determine variables.
- Remaining forward and backward degrees are integers and at least 1.
  dL is stored separately and is not subtracted in those inequalities.

Still open (QUESTIONS.md)
-------------------------
Whether both encryption states should be counted, how consumption bits are
split against the Dong21 cross-accounting, the unnamed consumption bits, and
whether a sixth n-XOR bullet exists. The piece index h0,h1,h2 is in the
model; 4-bit neutral-word variables are not (Q-LIN-BITS).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from skinny_mitm.exact_hull import HRep
from skinny_mitm.mc_dom import dom_table
from skinny_mitm.sbox_lin import APPROX_LIN_COST, sb_hull
from skinny_mitm.skinny_spec import N_CELLS, lower_tk_indices, upper_art_indices
from skinny_mitm.xor_rules import four_xor_hull, three_xor_hull, two_xor_hull

# Re-exported so callers can cite the coarse charge next to the model.
LIN_COST = APPROX_LIN_COST

# SKINNY-64/192. The combined tweakey is TK1 XOR TK2 XOR TK3. Not a parameter.
Z_TWEAK = 3


@dataclass
class Cell:
    name: str
    b: gp.Var
    r: gp.Var
    w: gp.Var
    is_blue: gp.Var
    is_red: gp.Var

    def bits(self):
        return (self.b, self.r, self.w)


@dataclass
class SBoxLin:
    """One S-box linearization: the cost bit and the one-hot affine piece."""

    lin: gp.Var
    h0: gp.Var
    h1: gp.Var
    h2: gp.Var


@dataclass
class InitialStructure:
    """ART-adjacent initial structure for combined TK1 XOR TK2 XOR TK3.

    ``tk_bottom`` is the eight key-DoF cells (state indices 8..15).
    ``tk_top`` is the eight combined cells on the ART rows. Their colors are
    tied to the encryption states and are not part of the key-DoF sum.
    """

    model: gp.Model
    round_index: int
    z: int
    pre_art: list
    post_art: list
    tk_top: list
    tk_bottom: list
    art_cf: list
    art_cb: list
    enc_forward: gp.LinExpr
    enc_backward: gp.LinExpr
    key_forward: gp.LinExpr
    key_backward: gp.LinExpr
    consumed_enc_forward: gp.Var
    consumed_enc_backward: gp.Var
    consumed_key_forward: gp.Var
    consumed_key_backward: gp.Var
    forward_dof: gp.Var
    backward_dof: gp.Var
    dL: gp.Var
    lin_vars: list = field(default_factory=list)
    dl_constr: object = None


def _add_hrep(model: gp.Model, hull: HRep, variables, prefix: str) -> None:
    for i, (coeffs, rhs) in enumerate(hull.inequalities):
        model.addConstr(
            gp.quicksum(coeffs[j] * variables[j] for j in range(len(coeffs))) <= rhs,
            name=f"{prefix}_le_{i}",
        )
    for i, (coeffs, rhs) in enumerate(hull.equalities):
        model.addConstr(
            gp.quicksum(coeffs[j] * variables[j] for j in range(len(coeffs))) == rhs,
            name=f"{prefix}_eq_{i}",
        )


def _add_cell(model: gp.Model, name: str, allow_green: bool) -> Cell:
    b = model.addVar(vtype=GRB.BINARY, name=f"b_{name}")
    r = model.addVar(vtype=GRB.BINARY, name=f"r_{name}")
    w = model.addVar(vtype=GRB.BINARY, name=f"w_{name}")
    # Legal codes: (0,0,0) gray, (1,0,0) blue, (0,1,0) red, (0,0,1) white,
    # and (1,1,0) green when allow_green is true.
    # Forbid (1,0,1), (0,1,1) and (1,1,1).
    model.addConstr(b + w <= 1, name=f"color_{name}_bw")
    model.addConstr(r + w <= 1, name=f"color_{name}_rw")
    if not allow_green:
        # Encryption initial cells: green (1, 1, 0) is forbidden.
        # Gray (0, 0, 0) stays feasible.
        model.addConstr(b + r <= 1, name=f"no_green_{name}")
    is_blue = model.addVar(vtype=GRB.BINARY, name=f"is_blue_{name}")
    is_red = model.addVar(vtype=GRB.BINARY, name=f"is_red_{name}")
    # is_blue = 1 iff (b, r) = (1, 0). With the constraints above, w is then 0,
    # so the cell is blue and not green.
    model.addConstr(is_blue <= b, name=f"blue_le_b_{name}")
    model.addConstr(is_blue <= 1 - r, name=f"blue_le_notr_{name}")
    model.addConstr(is_blue >= b - r, name=f"blue_ge_{name}")
    model.addConstr(is_red <= r, name=f"red_le_r_{name}")
    model.addConstr(is_red <= 1 - b, name=f"red_le_notb_{name}")
    model.addConstr(is_red >= r - b, name=f"red_ge_{name}")
    return Cell(name, b, r, w, is_blue, is_red)


def _sum_indicator(cells, attr: str) -> gp.LinExpr:
    if not cells:
        return gp.LinExpr(0.0)
    return gp.quicksum(getattr(cell, attr) for cell in cells)


def build_initial_structure_model(round_index: int) -> InitialStructure:
    """Build the z=3 color-and-DoF skeleton at ``round_index``.

    There is no tweakey-lane parameter. Key degrees come only from the eight
    combined bottom cells.
    """
    if not isinstance(round_index, int) or isinstance(round_index, bool):
        raise TypeError("round_index must be an int")
    if round_index < 0:
        raise ValueError("round_index must be >= 0; no maximum round is assumed")

    model = gp.Model(f"skinny_is_r{round_index}_z{Z_TWEAK}")
    model.Params.OutputFlag = 0
    # Constant objective. Rounds are not maximized.
    model.setObjective(0)

    pre = [_add_cell(model, f"pre_r{round_index}_{i}", allow_green=False) for i in range(N_CELLS)]
    post = [_add_cell(model, f"post_r{round_index}_{i}", allow_green=False) for i in range(N_CELLS)]
    # Combined TK1 XOR TK2 XOR TK3. Green is allowed on both halves.
    # Only the bottom half enters the key-DoF sum below.
    tk_top = [
        _add_cell(model, f"tk_hi_r{round_index}_{state_i}", allow_green=True)
        for state_i in upper_art_indices()
    ]
    tk_bottom = [
        _add_cell(model, f"tk_lo_r{round_index}_{state_i}", allow_green=True)
        for state_i in lower_tk_indices()
    ]
    # Value-level ART is state XOR TK1 XOR TK2 XOR TK3. After the lanes are
    # combined, and because a constant (AC) preserves color, each of cells
    # 0..7 is the two-input rule post = pre XOR tk_top. The top cells are
    # not given free colors outside this link, and they are not key DoF.
    art_cf = []
    art_cb = []
    for offset, state_i in enumerate(upper_art_indices()):
        cf, cb = add_two_xor(
            model,
            pre[state_i],
            tk_top[offset],
            post[state_i],
            f"art_r{round_index}_{state_i}",
        )
        art_cf.append(cf)
        art_cb.append(cb)

    enc_cells = pre + post
    # OPEN (Q-ENCDOF): both encryption states are summed. This may double-count
    # neutral words that ART ties together. Green is excluded by is_blue / is_red.
    enc_forward = _sum_indicator(enc_cells, "is_blue")
    enc_backward = _sum_indicator(enc_cells, "is_red")
    # Key DoF is these eight combined bottom cells only.
    key_forward = _sum_indicator(tk_bottom, "is_blue")
    key_backward = _sum_indicator(tk_bottom, "is_red")

    consumed_enc_forward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_enc_forward")
    consumed_enc_backward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_enc_backward")
    consumed_key_forward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_key_forward")
    consumed_key_backward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_key_backward")
    forward_dof = model.addVar(lb=0, vtype=GRB.INTEGER, name="forward_dof")
    backward_dof = model.addVar(lb=0, vtype=GRB.INTEGER, name="backward_dof")

    # Encryption-state initial DoF is fully consumed.
    model.addConstr(consumed_enc_forward == enc_forward, name="enc_forward_fully_consumed")
    model.addConstr(consumed_enc_backward == enc_backward, name="enc_backward_fully_consumed")
    # Remaining key DoF, as integers. "> 0" is encoded by ">= 1".
    # dL is not subtracted here.
    # art_cf / art_cb are not wired in: which pool they debit is Q-DOF-SPLIT.
    model.addConstr(forward_dof == key_forward - consumed_key_forward, name="forward_dof_remaining_key")
    model.addConstr(backward_dof == key_backward - consumed_key_backward, name="backward_dof_remaining_key")
    model.addConstr(forward_dof >= 1, name="forward_dof_positive")
    model.addConstr(backward_dof >= 1, name="backward_dof_positive")

    dL = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="dL")
    structure = InitialStructure(
        model=model,
        round_index=round_index,
        z=Z_TWEAK,
        pre_art=pre,
        post_art=post,
        tk_top=tk_top,
        tk_bottom=tk_bottom,
        art_cf=art_cf,
        art_cb=art_cb,
        enc_forward=enc_forward,
        enc_backward=enc_backward,
        key_forward=key_forward,
        key_backward=key_backward,
        consumed_enc_forward=consumed_enc_forward,
        consumed_enc_backward=consumed_enc_backward,
        consumed_key_forward=consumed_key_forward,
        consumed_key_backward=consumed_key_backward,
        forward_dof=forward_dof,
        backward_dof=backward_dof,
        dL=dL,
    )
    _refresh_dl(structure)
    model.update()
    return structure


def _refresh_dl(structure: InitialStructure) -> None:
    """dL = APPROX_LIN_COST * (number of linearized S-boxes).

    0.5 is the user-approved coarse binary charge. The exact weight
    ``0.25*log2(3)`` is stored as ``EXACT_LIN_COST`` and is not substituted
    into the Gurobi expression.
    """
    if structure.dl_constr is not None:
        structure.model.remove(structure.dl_constr)
    if structure.lin_vars:
        expr = LIN_COST * gp.quicksum(structure.lin_vars)
    else:
        expr = 0.0
    structure.dl_constr = structure.model.addConstr(structure.dL == expr, name="dL_def")


def link_consumption(
    structure: InitialStructure,
    *,
    enc_forward_bits=None,
    enc_backward_bits=None,
    key_forward_bits=None,
    key_backward_bits=None,
) -> None:
    """Tie consumption aggregates to explicit auxiliary bits.

    Which bits belong to the encryption state and which belong to the key is
    not determined (Q-DOF-SPLIT). The caller passes the lists.
    """
    model = structure.model
    pairs = (
        ("link_enc_forward", structure.consumed_enc_forward, enc_forward_bits),
        ("link_enc_backward", structure.consumed_enc_backward, enc_backward_bits),
        ("link_key_forward", structure.consumed_key_forward, key_forward_bits),
        ("link_key_backward", structure.consumed_key_backward, key_backward_bits),
    )
    for name, aggregate, bits in pairs:
        if bits is None:
            continue
        if len(bits) == 0:
            model.addConstr(aggregate == 0, name=name)
        else:
            model.addConstr(aggregate == gp.quicksum(bits), name=name)


def add_sbox(structure: InitialStructure, inp: Cell, out: Cell, name: str) -> SBoxLin:
    """Paper SB color rule, plus a one-hot affine-piece index.

    ``h0 + h1 + h2 = lin``. The coarse cost is ``APPROX_LIN_COST`` per ``lin``.
    No 4-bit neutral-word variables are created.
    """
    model = structure.model
    lin = model.addVar(vtype=GRB.BINARY, name=f"lin_{name}")
    h0 = model.addVar(vtype=GRB.BINARY, name=f"h0_{name}")
    h1 = model.addVar(vtype=GRB.BINARY, name=f"h1_{name}")
    h2 = model.addVar(vtype=GRB.BINARY, name=f"h2_{name}")
    model.addConstr(h0 + h1 + h2 == lin, name=f"piece_{name}")
    variables = (*inp.bits(), *out.bits(), lin)
    _add_hrep(model, sb_hull(), variables, f"sb_{name}")
    structure.lin_vars.append(lin)
    _refresh_dl(structure)
    return SBoxLin(lin, h0, h1, h2)


def add_two_xor(model: gp.Model, left: Cell, right: Cell, out: Cell, name: str):
    """Five-rule two-input XOR. Returns the two consumption bits ``(cf, cb)``."""
    cf = model.addVar(vtype=GRB.BINARY, name=f"cf_{name}")
    cb = model.addVar(vtype=GRB.BINARY, name=f"cb_{name}")
    variables = (*left.bits(), *right.bits(), *out.bits(), cf, cb)
    _add_hrep(model, two_xor_hull(), variables, f"xor2_{name}")
    return cf, cb


def _add_max_bit(model: gp.Model, target: gp.Var, bits, name: str) -> None:
    for i, bit in enumerate(bits):
        model.addConstr(target >= bit, name=f"{name}_ge_{i}")
    if bits:
        model.addConstr(target <= gp.quicksum(bits), name=f"{name}_le_sum")
    else:
        model.addConstr(target == 0, name=f"{name}_zero")


def add_multi_xor(model: gp.Model, inputs, out: Cell, name: str):
    """3-xor or 4-xor on (P_b, P_r, P_w), output color, and two consumption bits.

    3-input and 4-input use the same n-XOR rule, so they share one hull.
    The separate names ``three_xor_hull`` and ``four_xor_hull`` both return it.
    """
    n_inputs = len(inputs)
    if n_inputs == 3:
        hull = three_xor_hull()
    elif n_inputs == 4:
        hull = four_xor_hull()
    else:
        raise ValueError("add_multi_xor supports 3 or 4 inputs")
    p_b = model.addVar(vtype=GRB.BINARY, name=f"Pb_{name}")
    p_r = model.addVar(vtype=GRB.BINARY, name=f"Pr_{name}")
    p_w = model.addVar(vtype=GRB.BINARY, name=f"Pw_{name}")
    _add_max_bit(model, p_b, [cell.b for cell in inputs], f"Pb_{name}")
    _add_max_bit(model, p_r, [cell.r for cell in inputs], f"Pr_{name}")
    _add_max_bit(model, p_w, [cell.w for cell in inputs], f"Pw_{name}")
    cf = model.addVar(vtype=GRB.BINARY, name=f"cf_{name}")
    cb = model.addVar(vtype=GRB.BINARY, name=f"cb_{name}")
    variables = (p_b, p_r, p_w, *out.bits(), cf, cb)
    _add_hrep(model, hull, variables, f"xor{n_inputs}_{name}")
    return cf, cb


def add_copy(model: gp.Model, src: Cell, dst: Cell, name: str) -> None:
    """Wire equality. MC uses this for y1 = x0. No DoF is consumed."""
    model.addConstr(dst.b == src.b, name=f"copy_b_{name}")
    model.addConstr(dst.r == src.r, name=f"copy_r_{name}")
    model.addConstr(dst.w == src.w, name=f"copy_w_{name}")


def add_column_dom(model: gp.Model, white_bits, name: str) -> gp.Var:
    """Lookup ``dom_table()[mask]`` from eight white indicators.

    Bit ``i`` of the mask is ``white_bits[i]``. Under the five-color
    constraints, ``w = 1`` is exactly the white color.
    """
    if len(white_bits) != 8:
        raise ValueError("one SKINNY column has 4 input cells and 4 output cells")
    table = dom_table()
    selectors = [model.addVar(vtype=GRB.BINARY, name=f"{name}_mask_{mask}") for mask in range(256)]
    model.addConstr(gp.quicksum(selectors) == 1, name=f"{name}_one_mask")
    for bit in range(8):
        model.addConstr(
            white_bits[bit] == gp.quicksum(selectors[mask] for mask in range(256) if (mask >> bit) & 1),
            name=f"{name}_bit_{bit}",
        )
    dom = model.addVar(lb=0, ub=4, vtype=GRB.INTEGER, name=f"{name}_dom")
    model.addConstr(
        dom == gp.quicksum(int(table[mask]) * selectors[mask] for mask in range(256)),
        name=f"{name}_dom_def",
    )
    return dom


def add_mc_column(model: gp.Model, x_cells, y_cells, name: str) -> gp.Var:
    """Color rules for one MixColumns column, plus the white-mask DoM lookup.

    ``x_cells`` and ``y_cells`` are ordered (row 0, row 1, row 2, row 3).

        y0 = x0 XOR x2 XOR x3     # 3-xor rule
        y1 = x0                   # copy, no consumption
        y2 = x1 XOR x2            # 2-xor rule
        y3 = x0 XOR x2            # 2-xor rule
    """
    if len(x_cells) != 4 or len(y_cells) != 4:
        raise ValueError("MC column expects 4 input cells and 4 output cells")
    add_multi_xor(model, (x_cells[0], x_cells[2], x_cells[3]), y_cells[0], f"{name}_y0")
    add_copy(model, x_cells[0], y_cells[1], f"{name}_y1")
    add_two_xor(model, x_cells[1], x_cells[2], y_cells[2], f"{name}_y2")
    add_two_xor(model, x_cells[0], x_cells[2], y_cells[3], f"{name}_y3")
    white_bits = [cell.w for cell in list(x_cells) + list(y_cells)]
    return add_column_dom(model, white_bits, f"{name}_dom")


def add_state_dom(model: gp.Model, column_doms, name: str = "state_dom") -> gp.Var:
    """Total DoM is the sum of the four columns."""
    if len(column_doms) != 4:
        raise ValueError("state DoM sums exactly four columns")
    total = model.addVar(lb=0, vtype=GRB.INTEGER, name=name)
    model.addConstr(total == gp.quicksum(column_doms), name=f"{name}_sum")
    return total


def art_cell_indices():
    """Cells 0..7. ART writes these and no others. AC is not a color input."""
    return upper_art_indices()

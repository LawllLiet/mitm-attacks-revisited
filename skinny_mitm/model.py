"""Gurobi skeleton for one SKINNY initial-structure position.

The five colors are the five codes in Chen et al., Section 4.2. They are
unambiguous there, so this module does not raise ``NotImplementedError`` for
the color map. Open choices (tweakey lanes, AC placement, which cells are
key DoF, how consumption bits are partitioned, the objective) are recorded
in QUESTIONS.md and are not filled in with an attack search.

What is encoded
---------------
- Color variables ``(b, r, w)`` for the pre-ART state, the post-ART state,
  and the lower two rows of each tweakey lane.
- Encryption cells cannot be green. Key cells may be green.
- One-way DoF counts a cell only when it is blue, or only when it is red.
  Green is not counted. White and gray are not counted.
- Encryption-state DoF is set equal to a consumption aggregate (fully
  consumed). Forward/backward attack DoF are the remaining key-state DoF
  and are constrained to be at least 1.
- LIN cost uses ``APPROX_LIN_COST`` (0.5) times a binary indicator.
- MC degree of matching is a lookup of the precomputed white-mask table.

What is deliberately not encoded
--------------------------------
No round objective, no hash mode, no sourcing function, no guess-and-determine
encoders, and no automatic XOR link from X and Y onto tweakey lanes. Those
depend on questions Q-Z, Q-AC, Q-DOF-SPLIT and Q-OBJ.
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
class InitialStructure:
    """One ART-adjacent initial structure. ``n_tk`` is chosen by the caller."""

    model: gp.Model
    round_index: int
    n_tk: int
    pre_art: list
    post_art: list
    tk_lower: list
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
    # Five listed codes: forbid w together with b or r.
    model.addConstr(b + w <= 1, name=f"color_{name}_bw")
    model.addConstr(r + w <= 1, name=f"color_{name}_rw")
    if not allow_green:
        # Encryption initial cells: green (1, 1, 0) is forbidden.
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


def build_initial_structure_model(round_index: int, n_tk: int) -> InitialStructure:
    """Build the color-and-DoF skeleton at ``round_index``.

    ``n_tk`` is required and must be 1, 2 or 3. There is no default: the
    SKINNY variant was not specified (Q-Z). For ``n_tk > 1`` the lower two
    rows of every lane are included. That reading is flagged in Q-KEYDOF.
    """
    if not isinstance(round_index, int) or isinstance(round_index, bool):
        raise TypeError("round_index must be an int")
    if round_index < 0:
        raise ValueError("round_index must be >= 0; no maximum round is assumed")
    if n_tk not in (1, 2, 3):
        raise ValueError("n_tk must be 1, 2 or 3; refusing to pick a SKINNY variant")

    model = gp.Model(f"skinny_is_r{round_index}_z{n_tk}")
    model.Params.OutputFlag = 0
    # No round-maximizing objective. The Chen objective is not installed (Q-OBJ).
    model.setObjective(0)

    pre = [_add_cell(model, f"pre_r{round_index}_{i}", allow_green=False) for i in range(N_CELLS)]
    post = [_add_cell(model, f"post_r{round_index}_{i}", allow_green=False) for i in range(N_CELLS)]
    tk_lower = []
    for lane in range(n_tk):
        lane_cells = []
        for state_i in lower_tk_indices():
            lane_cells.append(
                _add_cell(
                    model,
                    f"tk{lane}_lo_r{round_index}_{state_i}",
                    allow_green=True,
                )
            )
        tk_lower.append(lane_cells)

    enc_cells = pre + post
    key_cells = [cell for lane in tk_lower for cell in lane]
    # FLAGGED (Q-ENCDOF): both encryption states are summed. This may double-count
    # neutral words that ART ties together. Green is excluded by is_blue / is_red.
    enc_forward = _sum_indicator(enc_cells, "is_blue")
    enc_backward = _sum_indicator(enc_cells, "is_red")
    # FLAGGED (Q-KEYDOF): key DoF is the blue/red count on the included lower rows.
    key_forward = _sum_indicator(key_cells, "is_blue")
    key_backward = _sum_indicator(key_cells, "is_red")

    consumed_enc_forward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_enc_forward")
    consumed_enc_backward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_enc_backward")
    consumed_key_forward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_key_forward")
    consumed_key_backward = model.addVar(lb=0, vtype=GRB.INTEGER, name="consumed_key_backward")
    forward_dof = model.addVar(lb=0, vtype=GRB.INTEGER, name="forward_dof")
    backward_dof = model.addVar(lb=0, vtype=GRB.INTEGER, name="backward_dof")

    # Encryption-state initial DoF is fully consumed.
    model.addConstr(consumed_enc_forward == enc_forward, name="enc_forward_fully_consumed")
    model.addConstr(consumed_enc_backward == enc_backward, name="enc_backward_fully_consumed")
    # Attack DoF is the remaining key-state DoF. For an integer this is the MILP
    # form of "> 0". dL is NOT subtracted here (Q-POS).
    model.addConstr(forward_dof == key_forward - consumed_key_forward, name="forward_dof_remaining_key")
    model.addConstr(backward_dof == key_backward - consumed_key_backward, name="backward_dof_remaining_key")
    model.addConstr(forward_dof >= 1, name="forward_dof_positive")
    model.addConstr(backward_dof >= 1, name="backward_dof_positive")

    dL = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="dL")
    structure = InitialStructure(
        model=model,
        round_index=round_index,
        n_tk=n_tk,
        pre_art=pre,
        post_art=post,
        tk_lower=tk_lower,
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


def add_sbox(structure: InitialStructure, inp: Cell, out: Cell, name: str) -> gp.Var:
    """Paper SB-rule plus one binary whose coarse cost is ``APPROX_LIN_COST``."""
    lin = structure.model.addVar(vtype=GRB.BINARY, name=f"lin_{name}")
    variables = (*inp.bits(), *out.bits(), lin)
    _add_hrep(structure.model, sb_hull(), variables, f"sb_{name}")
    structure.lin_vars.append(lin)
    _refresh_dl(structure)
    return lin


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
    """Document the cells ART writes. Not a guess about AC."""
    return upper_art_indices()

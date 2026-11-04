"""Gurobi skeleton: colors, DoF inequalities, LIN charge, MC DoM lookup."""

import unittest

import gurobipy as gp
from gurobipy import GRB

from skinny_mitm.colors import BLUE, EXCLUDED_CODES, GRAY, GREEN, RED, VALID_COLORS, WHITE
from skinny_mitm.mc_dom import column_dom, state_dom
from skinny_mitm.model import (
    _add_cell,
    add_column_dom,
    add_copy,
    add_mc_column,
    add_multi_xor,
    add_sbox,
    add_state_dom,
    add_two_xor,
    build_initial_structure_model,
    link_consumption,
)
from skinny_mitm.sbox_lin import APPROX_LIN_COST

ALL_CODES = tuple((b, r, w) for b in (0, 1) for r in (0, 1) for w in (0, 1))


def _fix(model, cell, color):
    b, r, w = color
    model.addConstr(cell.b == b)
    model.addConstr(cell.r == r)
    model.addConstr(cell.w == w)


def _paint(structure, enc_color=GRAY, key_color=GRAY, overrides=None):
    overrides = overrides or {}
    model = structure.model
    for cell in structure.pre_art + structure.post_art:
        _fix(model, cell, overrides.get(cell.name, enc_color))
    for cell in structure.tk_bottom:
        _fix(model, cell, overrides.get(cell.name, key_color))
    for cell in structure.tk_top:
        if cell.name in overrides:
            _fix(model, cell, overrides[cell.name])


def _expect_indicators(code):
    return (1 if code == BLUE else 0, 1 if code == RED else 0)


def _bare(name="bare"):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    return model


class InitialStructureTests(unittest.TestCase):
    def test_z3_has_no_lane_parameter(self):
        with self.assertRaises(TypeError):
            build_initial_structure_model()
        with self.assertRaises(TypeError):
            build_initial_structure_model(0, 3)
        with self.assertRaises(ValueError):
            build_initial_structure_model(-1)
        with self.assertRaises(TypeError):
            build_initial_structure_model(True)
        structure = build_initial_structure_model(2)
        self.assertEqual(structure.z, 3)
        self.assertEqual(structure.round_index, 2)
        self.assertEqual(len(structure.pre_art), 16)
        self.assertEqual(len(structure.post_art), 16)
        self.assertEqual(len(structure.tk_bottom), 8)
        self.assertEqual(len(structure.tk_top), 8)
        self.assertEqual(len(structure.art_cf), 8)
        self.assertFalse(hasattr(structure, "n_tk"))
        self.assertFalse(hasattr(structure, "tk_lower"))
        self.assertEqual(structure.model.getObjective().size(), 0)

    def test_only_the_five_color_tuples_are_feasible(self):
        for allow_green in (True, False):
            legal = VALID_COLORS if allow_green else tuple(code for code in VALID_COLORS if code != GREEN)
            self.assertEqual(len(legal), 5 if allow_green else 4)
            for code in ALL_CODES:
                model = _bare(f"color_{allow_green}_{code}")
                cell = _add_cell(model, "c", allow_green)
                _fix(model, cell, code)
                model.optimize()
                if code in legal:
                    self.assertEqual(model.Status, GRB.OPTIMAL, code)
                    blue, red = _expect_indicators(code)
                    self.assertEqual(int(round(cell.is_blue.X)), blue)
                    self.assertEqual(int(round(cell.is_red.X)), red)
                else:
                    self.assertEqual(model.Status, GRB.INFEASIBLE, code)
        self.assertEqual(set(EXCLUDED_CODES), {(1, 0, 1), (0, 1, 1), (1, 1, 1)})
        self.assertEqual(
            set(VALID_COLORS),
            {BLUE, RED, WHITE, GRAY, GREEN},
        )
        self.assertEqual(BLUE, (1, 0, 0))
        self.assertEqual(RED, (0, 1, 0))
        self.assertEqual(WHITE, (0, 0, 1))
        self.assertEqual(GRAY, (0, 0, 0))
        self.assertEqual(GREEN, (1, 1, 0))

    def test_objective_stays_zero(self):
        structure = build_initial_structure_model(1)
        _paint(
            structure,
            overrides={
                structure.tk_bottom[0].name: BLUE,
                structure.tk_bottom[1].name: RED,
            },
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(structure.model.ObjVal, 0.0)

    def test_encryption_cell_cannot_be_green_and_key_bottom_can(self):
        blocked = build_initial_structure_model(0)
        _fix(blocked.model, blocked.pre_art[15], GREEN)
        blocked.model.optimize()
        self.assertEqual(blocked.model.Status, GRB.INFEASIBLE)

        allowed = build_initial_structure_model(0)
        bottom = allowed.tk_bottom
        _paint(
            allowed,
            GRAY,
            GRAY,
            {bottom[0].name: GREEN, bottom[1].name: BLUE, bottom[2].name: RED},
        )
        link_consumption(allowed, key_forward_bits=[], key_backward_bits=[])
        allowed.model.optimize()
        self.assertEqual(allowed.model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(bottom[0].b.X)), 1)
        self.assertEqual(int(round(bottom[0].r.X)), 1)
        self.assertEqual(int(round(bottom[0].w.X)), 0)
        self.assertEqual(int(round(bottom[0].is_blue.X)), 0)
        self.assertEqual(int(round(bottom[0].is_red.X)), 0)
        self.assertEqual(int(round(allowed.forward_dof.X)), 1)
        self.assertEqual(int(round(allowed.backward_dof.X)), 1)

    def test_gray_encryption_cell_is_feasible(self):
        structure = build_initial_structure_model(0)
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                structure.tk_bottom[0].name: BLUE,
                structure.tk_bottom[1].name: RED,
            },
        )
        link_consumption(
            structure,
            enc_forward_bits=[],
            enc_backward_bits=[],
            key_forward_bits=[],
            key_backward_bits=[],
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        cell = structure.pre_art[15]
        self.assertEqual((int(round(cell.b.X)), int(round(cell.r.X)), int(round(cell.w.X))), GRAY)
        self.assertEqual(int(round(structure.forward_dof.X)), 1)
        self.assertEqual(int(round(structure.backward_dof.X)), 1)
        self.assertEqual(int(round(structure.consumed_enc_forward.X)), 0)

    def test_key_dof_uses_exactly_eight_combined_bottom_cells(self):
        structure = build_initial_structure_model(0)
        overrides = {structure.tk_bottom[7].name: RED}
        for cell in structure.tk_bottom[:7]:
            overrides[cell.name] = BLUE
        # A blue top cell is linked by ART and must not add a key degree.
        overrides[structure.pre_art[0].name] = GRAY
        overrides[structure.post_art[0].name] = BLUE
        overrides[structure.tk_top[0].name] = BLUE
        _paint(structure, GRAY, GRAY, overrides)
        link_consumption(structure, key_forward_bits=[], key_backward_bits=[])
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(structure.tk_top[0].is_blue.X)), 1)
        self.assertEqual(int(round(structure.forward_dof.X)), 7)
        self.assertEqual(int(round(structure.backward_dof.X)), 1)
        self.assertEqual(sum(int(round(cell.is_blue.X)) for cell in structure.tk_bottom), 7)
        self.assertEqual(sum(int(round(cell.is_red.X)) for cell in structure.tk_bottom), 1)

    def test_art_top_link_is_two_input_xor_and_does_not_spend_key_dof(self):
        structure = build_initial_structure_model(0)
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                structure.pre_art[0].name: BLUE,
                structure.tk_top[0].name: BLUE,
                structure.post_art[0].name: GRAY,
                structure.tk_bottom[0].name: BLUE,
                structure.tk_bottom[1].name: RED,
            },
        )
        link_consumption(structure, key_forward_bits=[], key_backward_bits=[])
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(structure.art_cf[0].X)), 1)
        self.assertEqual(int(round(structure.art_cb[0].X)), 0)
        self.assertEqual(int(round(structure.forward_dof.X)), 1)

        rejected = build_initial_structure_model(1)
        _paint(
            rejected,
            GRAY,
            GRAY,
            {
                rejected.pre_art[0].name: WHITE,
                rejected.post_art[0].name: BLUE,
                rejected.tk_bottom[0].name: BLUE,
                rejected.tk_bottom[1].name: RED,
            },
        )
        rejected.model.optimize()
        self.assertEqual(rejected.model.Status, GRB.INFEASIBLE)

    def test_all_gray_key_is_infeasible(self):
        structure = build_initial_structure_model(0)
        _paint(structure, GRAY, GRAY)
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)

    def test_encryption_dof_must_be_consumed(self):
        structure = build_initial_structure_model(0)
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                structure.pre_art[0].name: BLUE,
                structure.post_art[0].name: GRAY,
                structure.tk_top[0].name: BLUE,
                structure.tk_bottom[0].name: BLUE,
                structure.tk_bottom[1].name: RED,
            },
        )
        link_consumption(
            structure,
            enc_forward_bits=[],
            enc_backward_bits=[],
            key_forward_bits=[],
            key_backward_bits=[],
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)

    def test_lin_one_hot_sums_to_lin_and_costs_one_half(self):
        structure = build_initial_structure_model(4)
        bottom = structure.tk_bottom
        kept = add_sbox(structure, bottom[0], bottom[1], "keep")
        dropped = add_sbox(structure, bottom[2], bottom[3], "drop")
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                bottom[0].name: GREEN,
                bottom[1].name: GREEN,
                bottom[2].name: GREEN,
                bottom[3].name: WHITE,
                bottom[4].name: BLUE,
                bottom[5].name: RED,
            },
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(APPROX_LIN_COST, 0.5)
        self.assertEqual(int(round(kept.lin.X)), 1)
        self.assertEqual(int(round(dropped.lin.X)), 0)
        for piece in (kept, dropped):
            bits = [int(round(piece.h0.X)), int(round(piece.h1.X)), int(round(piece.h2.X))]
            self.assertEqual(sum(bits), int(round(piece.lin.X)))
            self.assertTrue(all(bit in (0, 1) for bit in bits))
        self.assertEqual(
            [int(round(kept.h0.X)), int(round(kept.h1.X)), int(round(kept.h2.X))].count(1),
            1,
        )
        self.assertAlmostEqual(structure.dL.X, 0.5)
        structure.model.addConstr(kept.h0 + kept.h1 + kept.h2 == 0)
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)

    def test_non_green_sbox_cannot_pay_for_linearization(self):
        structure = build_initial_structure_model(0)
        piece = add_sbox(structure, structure.pre_art[0], structure.post_art[0], "pass")
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                structure.pre_art[0].name: BLUE,
                structure.post_art[0].name: BLUE,
                structure.tk_bottom[0].name: BLUE,
                structure.tk_bottom[1].name: RED,
            },
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(piece.lin.X)), 0)
        self.assertEqual(int(round(piece.h0.X + piece.h1.X + piece.h2.X)), 0)
        self.assertAlmostEqual(structure.dL.X, 0.0)
        structure.model.addConstr(piece.lin == 1)
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)


class PropagationTests(unittest.TestCase):
    def test_two_xor_accepts_a_paper_point_and_rejects_a_violation(self):
        model = _bare("xor2")
        left = _add_cell(model, "l", True)
        right = _add_cell(model, "r", True)
        out = _add_cell(model, "o", True)
        cf, cb = add_two_xor(model, left, right, out, "t")
        _fix(model, left, BLUE)
        _fix(model, right, GRAY)
        _fix(model, out, GRAY)
        model.addConstr(cf == 1)
        model.addConstr(cb == 0)
        model.optimize()
        self.assertEqual(model.Status, GRB.OPTIMAL)

        model.addConstr(cf == 0)
        model.optimize()
        self.assertEqual(model.Status, GRB.INFEASIBLE)

    def test_multi_xor_four_mixed_outcomes_and_bad_arity(self):
        model = _bare("xor3")
        inputs = [_add_cell(model, f"i{i}", True) for i in range(3)]
        out = _add_cell(model, "o", True)
        cf, cb = add_multi_xor(model, inputs, out, "m")
        _fix(model, inputs[0], BLUE)
        _fix(model, inputs[1], RED)
        _fix(model, inputs[2], GRAY)
        _fix(model, out, GRAY)
        model.addConstr(cf == 1)
        model.addConstr(cb == 1)
        model.optimize()
        self.assertEqual(model.Status, GRB.OPTIMAL)

        rejected = _bare("xor3bad")
        inputs = [_add_cell(rejected, f"i{i}", True) for i in range(3)]
        out = _add_cell(rejected, "o", True)
        cf, cb = add_multi_xor(rejected, inputs, out, "m")
        _fix(rejected, inputs[0], BLUE)
        _fix(rejected, inputs[1], RED)
        _fix(rejected, inputs[2], GRAY)
        _fix(rejected, out, GRAY)
        rejected.addConstr(cf == 0)
        rejected.addConstr(cb == 0)
        rejected.optimize()
        self.assertEqual(rejected.Status, GRB.INFEASIBLE)

        with self.assertRaises(ValueError):
            add_multi_xor(_bare("bad"), inputs[:2], out, "no")

    def test_copy_has_no_consumption_bit(self):
        model = _bare("copy")
        src = _add_cell(model, "s", True)
        dst = _add_cell(model, "d", True)
        add_copy(model, src, dst, "c")
        _fix(model, src, RED)
        _fix(model, dst, RED)
        model.optimize()
        self.assertEqual(model.Status, GRB.OPTIMAL)
        model.addConstr(dst.b == 1)
        model.optimize()
        self.assertEqual(model.Status, GRB.INFEASIBLE)

    def test_column_dom_lookup_matches_the_rank_table(self):
        masks = (
            0,
            0xFF,
            (1 << 1) | (1 << 6),
            (1 << 1) | (1 << 3) | (1 << 4) | (1 << 6),
        )
        for mask in masks:
            model = _bare(f"dom_{mask}")
            bits = [model.addVar(vtype=GRB.BINARY, name=f"w{i}") for i in range(8)]
            dom = add_column_dom(model, bits, "col")
            for i, bit in enumerate(bits):
                model.addConstr(bit == ((mask >> i) & 1))
            model.optimize()
            self.assertEqual(model.Status, GRB.OPTIMAL)
            self.assertEqual(int(round(dom.X)), column_dom(mask))

    def test_state_dom_sums_four_columns(self):
        model = _bare("state")
        values = (4, 0, 3, 2)
        doms = []
        for i, value in enumerate(values):
            var = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"d{i}")
            model.addConstr(var == value)
            doms.append(var)
        total = add_state_dom(model, doms, "sum")
        model.optimize()
        self.assertEqual(model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(total.X)), 9)
        self.assertEqual(state_dom(values), 9)
        with self.assertRaises(ValueError):
            add_state_dom(model, doms[:3])

    def test_mc_column_all_gray_has_dom_four(self):
        model = _bare("mc")
        xs = [_add_cell(model, f"x{i}", True) for i in range(4)]
        ys = [_add_cell(model, f"y{i}", True) for i in range(4)]
        dom = add_mc_column(model, xs, ys, "col")
        for cell in xs + ys:
            _fix(model, cell, GRAY)
        model.optimize()
        self.assertEqual(model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(dom.X)), 4)
        self.assertEqual(column_dom(0), 4)


if __name__ == "__main__":
    unittest.main()

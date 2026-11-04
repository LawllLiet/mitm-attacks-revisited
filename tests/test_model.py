"""Gurobi skeleton: colors, DoF inequalities, LIN charge, MC DoM lookup."""

import unittest

import gurobipy as gp
from gurobipy import GRB

from skinny_mitm.colors import BLUE, GRAY, GREEN, RED, WHITE
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
    for lane in structure.tk_lower:
        for cell in lane:
            _fix(model, cell, overrides.get(cell.name, key_color))


def _bare(name="bare"):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    return model


class InitialStructureTests(unittest.TestCase):
    def test_variant_and_round_are_not_defaulted(self):
        with self.assertRaises(TypeError):
            build_initial_structure_model(0)
        with self.assertRaises(ValueError):
            build_initial_structure_model(0, 4)
        with self.assertRaises(ValueError):
            build_initial_structure_model(-1, 1)
        with self.assertRaises(TypeError):
            build_initial_structure_model(True, 1)

    def test_cell_counts_for_one_and_three_lanes(self):
        one = build_initial_structure_model(0, 1)
        three = build_initial_structure_model(2, 3)
        self.assertEqual(len(one.pre_art), 16)
        self.assertEqual(len(one.post_art), 16)
        self.assertEqual(len(one.tk_lower), 1)
        self.assertEqual(len(one.tk_lower[0]), 8)
        self.assertEqual(len(three.tk_lower), 3)
        self.assertEqual(sum(len(lane) for lane in three.tk_lower), 24)
        self.assertEqual(one.model.getObjective().size(), 0)
        self.assertEqual(three.round_index, 2)
        self.assertEqual(three.n_tk, 3)

    def test_objective_stays_zero(self):
        structure = build_initial_structure_model(1, 1)
        _paint(
            structure,
            overrides={
                structure.tk_lower[0][0].name: BLUE,
                structure.tk_lower[0][1].name: RED,
            },
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(structure.model.ObjVal, 0.0)

    def test_encryption_cell_cannot_be_green(self):
        structure = build_initial_structure_model(0, 1)
        cell = structure.pre_art[3]
        _fix(structure.model, cell, GREEN)
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)

    def test_excluded_color_code_is_infeasible_on_a_key_cell(self):
        structure = build_initial_structure_model(0, 1)
        cell = structure.tk_lower[0][0]
        _fix(structure.model, cell, (1, 0, 1))
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)

    def test_all_gray_key_is_infeasible(self):
        structure = build_initial_structure_model(0, 1)
        _paint(structure, GRAY, GRAY)
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)

    def test_one_blue_and_one_red_key_cell_is_feasible(self):
        structure = build_initial_structure_model(0, 3)
        blue = structure.tk_lower[2][0]
        red = structure.tk_lower[0][7]
        _paint(structure, GRAY, GRAY, {blue.name: BLUE, red.name: RED})
        link_consumption(
            structure,
            enc_forward_bits=[],
            enc_backward_bits=[],
            key_forward_bits=[],
            key_backward_bits=[],
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(structure.forward_dof.X)), 1)
        self.assertEqual(int(round(structure.backward_dof.X)), 1)
        self.assertEqual(int(round(structure.consumed_key_forward.X)), 0)
        self.assertEqual(int(round(structure.consumed_key_backward.X)), 0)
        self.assertEqual(int(round(structure.consumed_enc_forward.X)), 0)
        self.assertEqual(int(round(structure.consumed_enc_backward.X)), 0)
        self.assertAlmostEqual(structure.dL.X, 0.0)

    def test_green_key_cell_adds_neither_forward_nor_backward_dof(self):
        structure = build_initial_structure_model(0, 1)
        lane = structure.tk_lower[0]
        _paint(
            structure,
            GRAY,
            GRAY,
            {lane[0].name: GREEN, lane[1].name: BLUE, lane[2].name: RED},
        )
        link_consumption(
            structure,
            key_forward_bits=[],
            key_backward_bits=[],
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(lane[0].b.X)), 1)
        self.assertEqual(int(round(lane[0].r.X)), 1)
        self.assertEqual(int(round(lane[0].w.X)), 0)
        self.assertEqual(int(round(lane[0].is_blue.X)), 0)
        self.assertEqual(int(round(lane[0].is_red.X)), 0)
        self.assertEqual(int(round(structure.forward_dof.X)), 1)
        self.assertEqual(int(round(structure.backward_dof.X)), 1)

    def test_encryption_dof_must_be_consumed(self):
        structure = build_initial_structure_model(0, 1)
        lane = structure.tk_lower[0]
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                structure.pre_art[0].name: BLUE,
                lane[0].name: BLUE,
                lane[1].name: RED,
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

    def test_lin_charge_is_half_per_indicator(self):
        structure = build_initial_structure_model(4, 1)
        lane = structure.tk_lower[0]
        lin_keep = add_sbox(structure, lane[0], lane[1], "keep")
        lin_drop = add_sbox(structure, lane[2], lane[3], "drop")
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                lane[0].name: GREEN,
                lane[1].name: GREEN,
                lane[2].name: GREEN,
                lane[3].name: WHITE,
                lane[4].name: BLUE,
                lane[5].name: RED,
            },
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(APPROX_LIN_COST, 0.5)
        self.assertEqual(int(round(lin_keep.X)), 1)
        self.assertEqual(int(round(lin_drop.X)), 0)
        self.assertAlmostEqual(structure.dL.X, 0.5)
        # Forcing the kept-green branch to skip its indicator violates the SB-rule.
        structure.model.addConstr(lin_keep == 0)
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.INFEASIBLE)

    def test_non_green_sbox_cannot_pay_for_linearization(self):
        structure = build_initial_structure_model(0, 1)
        lin = add_sbox(structure, structure.pre_art[0], structure.post_art[0], "pass")
        lane = structure.tk_lower[0]
        _paint(
            structure,
            GRAY,
            GRAY,
            {
                structure.pre_art[0].name: BLUE,
                structure.post_art[0].name: BLUE,
                lane[0].name: BLUE,
                lane[1].name: RED,
            },
        )
        structure.model.optimize()
        self.assertEqual(structure.model.Status, GRB.OPTIMAL)
        self.assertEqual(int(round(lin.X)), 0)
        self.assertAlmostEqual(structure.dL.X, 0.0)
        structure.model.addConstr(lin == 1)
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

"""MILP pieces for a SKINNY meet-in-the-middle model using Chen et al. LIN/DIS.

Color propagation follows the (b, r, w) encoding in Chen et al., ePrint 2024/300,
Section 4.2. Open modeling choices are listed in ``QUESTIONS.md`` and are not
given default attack parameters.
"""

from skinny_mitm.sbox_lin import APPROX_LIN_COST, EXACT_LIN_COST

__all__ = ["APPROX_LIN_COST", "EXACT_LIN_COST"]

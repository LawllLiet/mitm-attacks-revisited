"""Five cell colors from Chen et al., ePrint 2024/300, Section 4.2.

Quoted from the paper (Color-encoding Scheme of Neutral Words):

- A blue cell denotes a forward neutral byte, encoded as (b, r, w) = (1, 0, 0).
- A red cell denotes a backward neutral byte, encoded as (b, r, w) = (0, 1, 0).
- A white cell denotes an arbitrary byte, encoded as (b, r, w) = (0, 0, 1).
- A gray cell denotes a constant byte, encoded as (b, r, w) = (0, 0, 0).
- A green cell denotes a superposition byte, encoded as (b, r, w) = (1, 1, 0).

The paper assigns no meaning to (1, 0, 1), (0, 1, 1) or (1, 1, 1). Together with
the requirement that each cell has five color choices, those three codes are
excluded. See QUESTIONS.md, item Q-COLOR, which records that this exclusion is
the reading of the five listed codes and not a sixth color.
"""

from __future__ import annotations

BLUE = (1, 0, 0)
RED = (0, 1, 0)
WHITE = (0, 0, 1)
GRAY = (0, 0, 0)
GREEN = (1, 1, 0)

COLOR_BY_NAME = {
    "blue": BLUE,
    "red": RED,
    "white": WHITE,
    "gray": GRAY,
    "green": GREEN,
}

NAME_BY_COLOR = {value: name for name, value in COLOR_BY_NAME.items()}

VALID_COLORS = (GRAY, BLUE, RED, WHITE, GREEN)

# Codes with no paper meaning. Forbidden so that a cell has exactly five choices.
EXCLUDED_CODES = ((1, 0, 1), (0, 1, 1), (1, 1, 1))


def is_valid_color(code) -> bool:
    return tuple(code) in NAME_BY_COLOR

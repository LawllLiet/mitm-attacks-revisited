# Rule text used by the SKINNY LIN/DIS skeleton

Source: Shiyao Chen, Jian Guo, Eik List, Danping Shi, Tianyu Zhang, “Diving Deep into the Preimage Security of AES-like Hashing”, ePrint 2024/300, Section 4.2. The PDF URL returned 403. The quotations below are the extracted sentences with line-break hyphens removed. They are the sentences the code follows. A rule that was not in the extract is listed at the end and is not implemented.

## Color encoding

> A blue cell denotes a forward neutral byte, encoded as (b, r, w) = (1, 0, 0).
> A red cell denotes a backward neutral byte, encoded as (b, r, w) = (0, 1, 0).
> A white cell denotes an arbitrary byte, encoded as (b, r, w) = (0, 0, 1).
> A gray cell denotes a constant byte, encoded as (b, r, w) = (0, 0, 0).
> A green cell denotes a superposition byte, encoded as (b, r, w) = (1, 1, 0).

> w = 1 uniquely identifies an arbitrary byte, whose value is unknown, against other byte types.

The same paragraph says the initial degrees of freedom can be obtained by summing the b and r encoders. The user forbade counting green toward either side, so the model uses the pure-blue and pure-red indicators instead of those raw sums. See QUESTIONS.md.

## SubBytes

> When the input byte is not green, the color of the output byte is identical to that of the input byte.
> When the input byte is green, the output byte is either white by default or green with one cost of linearization (dL incremented by one).

The weight of that one cost is the user's (1/4)·log2(3), charged as 0.5 in Gurobi. The AES examples in the paper use 0.5 byte and are not substituted here.

## XOR with two inputs

> When the input involves a white byte, the output is white.
> When the input contains only gray bytes, the output is gray.
> When the input contains only blue bytes, the output is either blue with no consumption of DoF or gray consuming one DoF from the forward chunk.
> When the input contains only red bytes, the output is either red with no DoF consumption or gray consuming one DoF from the backward chunk.
> When the input is a mixture of red and blue bytes or involves green bytes, the output is green.

## XOR with multiple inputs

> When the input involves a white byte, the output is white.
> When the input contains only gray bytes, the output is gray.
> When the input contains only blue bytes, the output is either blue with no DoF consumed or gray with 1 DoF consumed from the forward chunk.
> When the input contains only red bytes, the output is either red with no DoF consumed or gray with 1 DoF consumed from the backward chunk.
> When the input contains both red and blue bytes, the output is one of the following:
> green, with no consumption of DoF;
> blue, with 1 DoF consumed from the backward chunk;
> red, with 1 DoF consumed from the forward chunk;
> or gray, with 1 DoF consumed from each forward and backward chunk.

> P_b = max b_γ, P_r = max r_γ, and P_w = max w_γ. (8)

> The constraints for the n-input XOR rule can be obtained by using the convex hull method on P_b, P_r, P_w, the encoders of the intermediate byte, and two encoders for DoF costs.

The sourcing function described just before these bullets is not modeled.

No sixth top-level bullet appears in the extract. None was added. See QUESTIONS.md Q-SIX.

## Matching, as printed in the paper

Equation (9) is the feed-forward XOR-match. It is not in this skeleton: the user said MC is the only matching site.

Equation (10):

> d_M^{MC} = Σ_Δ max(0, t_Δ − N_ROW).

The implemented table does not use (10). It uses the user's rank formula on the SKINNY matrix. Where the two differ, the table stores the rank value and the tests record both numbers.

## Objective, not installed

Equation (11):

> min{ d_B − g_R − g_BR − d_L, d_R − g_B − g_BR − d_L, d_M − g_B − g_R − g_BR }.

The Gurobi objective is the constant 0. No round is maximized.

## Paper MC color rule, not installed as written

The extract gives a column-wise MC-rule (any white input byte makes the whole output column white, with consumptions counted per output byte) and a GnD-MC variant with encoders g_w, g_b, g_r, g_br. SKINNY's MixColumns does not mix every output byte with every input byte, and the user asked for the 3-xor on the actual matrix. Those paper constraints are quoted here so they are not silently replaced:

> When n_w > 0, the output contains only white bytes.
> When n_w = 0 and n_b + n_r + n_g = 0, the output contains only gray bytes.
> When n_w = n_r = 0 and n_b + n_g > 0, the output contains n'_b blue bytes and n'_c gray bytes, with n'_c consumed DoF from the forward chunk.
> When n_w = n_b = 0 and n_r + n_g > 0, the output contains n'_r red bytes and n'_c gray bytes, with n'_c consumed DoF from the backward chunk.
> Otherwise, the output is a mix of n'_b blue bytes, n'_r red bytes, n'_c gray bytes, and n'_g green bytes, with n'_b + n'_c consumed DoF from the backward chunk and n'_r + n'_c consumed DoF from the forward chunk.

The cell-wise SKINNY rules actually coded are in `model.add_mc_column` and QUESTIONS.md Q-MC-RULE.

## Rules that were not found

- A sixth top-level n-XOR bullet, distinct from the four outcomes already inside the last bullet.
- A SKINNY-specific initial-structure figure, hash mode, or round count in the extracted Chen et al. text.
- A statement that fixes which (c_f, c_b) values are legal when a bullet does not mention consumption.

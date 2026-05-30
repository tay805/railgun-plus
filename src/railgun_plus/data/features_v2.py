"""Step 2: feature engineering with coordination signals.

V1 (the original RAILGUN-style features in features.py) gives the network only
WHERE agents are and HOW FAR they are from their goals. It does NOT tell the
network anything about *congestion* or *what other agents are about to do* --
which is exactly the signal needed to learn coordination beyond go-toward-goal.

V2 adds three new channels designed to fix that:

  channel 6: local agent-density (5x5 sum of current occupancy). Direct
             "this corridor is crowded" signal.

  channel 7: predicted next-step occupancy. For each agent, take the action
             that decreases its cost-to-goal most (the greedy go-toward-goal
             choice). Mark the cell that agent would move into. Tells the
             network "this cell will be contested next step."

  channel 8: per-cell remaining cost-to-goal of the AGENT occupying the cell
             (zero elsewhere). Distinguishes "agent with 50 steps left" from
             "agent with 2 steps left" -- matters for who should yield.

Channels 0-5 are identical to V1, so the v1 trained model is dimensionally
incompatible with v2 but the v1 FEATURES are a strict subset of v2.

USAGE
-----
Drop-in replacement for features.instance_to_samples():
    from railgun_plus.data.features_v2 import instance_to_samples_v2
    samples = list(instance_to_samples_v2(inst))

NUM_FEATURE_CHANNELS_V2 = 9 (vs 6 for v1).
"""
from __future__ import annotations

import numpy as np

from .types import Instance, DELTA_TO_ACTION
from .grid_utils import bfs_distance_field
from .features import (pad_to_multiple, cost_gradient, action_label,
                       NUM_ACTIONS)

NUM_FEATURE_CHANNELS_V2 = 9  # v1's 6 + 3 new coordination channels


def local_density(cur: np.ndarray, k: int = 5) -> np.ndarray:
    """Sum of agent occupancy in a (k x k) window centered on each cell.

    cur:  (H, W) array, nonzero where an agent stands.
    k:    odd window size (default 5).
    Returns (H, W) float array of integer counts.

    Implementation: a correct integral-image / summed-area-table approach.
    The earlier in-place version had a fence-post bug that caused mismatches
    of up to ~5 on random inputs (caught by the naive-baseline unit check).
    This version computes the SAT properly with a zero-padded prefix row/col,
    so window sums are clean differences with no boundary special-casing.
    """
    assert k % 2 == 1, "k must be odd"
    H, W = cur.shape
    binc = (cur > 0).astype(np.float32)

    # Pad with `r` zero rows/cols on each side so windows near borders just
    # see zeros there (matches the naive boundary behaviour).
    r = k // 2
    padded = np.zeros((H + 2 * r, W + 2 * r), dtype=np.float32)
    padded[r:r + H, r:r + W] = binc

    # Summed-area table with a zero prefix row/col so SAT[i+1,j+1] = sum of
    # padded[:i+1, :j+1], and SAT[0,*] = SAT[*,0] = 0. This makes the window
    # sum at output cell (i,j) -- which corresponds to padded rows [i, i+k)
    # and cols [j, j+k) -- a clean four-corner difference.
    Hp, Wp = padded.shape
    sat = np.zeros((Hp + 1, Wp + 1), dtype=np.float32)
    sat[1:, 1:] = padded.cumsum(0).cumsum(1)

    # Output (i,j) = sum over padded[i:i+k, j:j+k]
    #              = SAT[i+k, j+k] - SAT[i, j+k] - SAT[i+k, j] + SAT[i, j]
    out = (sat[k:k + H, k:k + W]
           - sat[:H, k:k + W]
           - sat[k:k + H, :W]
           + sat[:H, :W])
    return out


def predicted_next_occupancy(grid: np.ndarray, cur_pos, dist_fields) -> np.ndarray:
    """Where each agent would move next under a GREEDY go-toward-goal step.

    For each agent, find the neighbor (including wait) that minimizes its
    cost-to-goal distance; mark that cell with +1. Cells targeted by multiple
    agents get higher counts -- a direct "this cell will be contested" signal.

    Returns (H, W) float array.
    """
    H, W = grid.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i, (r, c) in enumerate(cur_pos):
        d = dist_fields[i]
        best = (r, c)
        best_d = d[r][c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                if d[nr][nc] < best_d:
                    best_d = d[nr][nc]
                    best = (nr, nc)
        out[best[0]][best[1]] += 1.0
    return out


def remaining_cost_per_cell(cur_pos, dist_fields, shape) -> np.ndarray:
    """Per-cell: remaining cost-to-goal of the agent currently AT that cell."""
    H, W = shape
    out = np.zeros((H, W), dtype=np.float32)
    for i, (r, c) in enumerate(cur_pos):
        out[r][c] = float(dist_fields[i][r][c])
    return out


def instance_to_samples_v2(inst: Instance, pad_multiple: int = 16):
    """Yield (F_in, F_out, mask) tuples -- one per timestep transition.

    F_in:  float32 (9, Hp, Wp)   <-- 9 channels (v1's 6 + 3 new)
    F_out: int64   (Hp, Wp)
    mask:  float32 (Hp, Wp)
    """
    grid = inst.grid
    padded, valid, (H, W) = pad_to_multiple(grid, pad_multiple)
    Hp, Wp = padded.shape

    dist_fields = [bfs_distance_field(grid, g) for g in inst.goals]

    goal_chan = np.zeros((H, W), dtype=np.float32)
    for i, (gr, gc) in enumerate(inst.goals):
        goal_chan[gr][gc] = i + 1

    T = inst.horizon
    for t in range(T - 1):
        cur_pos = [inst.paths[i][t] for i in range(inst.num_agents)]

        cur = np.zeros((H, W), dtype=np.float32)
        ctg = np.zeros((H, W), dtype=np.float32)
        gdx = np.zeros((H, W), dtype=np.float32)
        gdy = np.zeros((H, W), dtype=np.float32)
        out = np.zeros((Hp, Wp), dtype=np.int64)
        mask = np.zeros((Hp, Wp), dtype=np.float32)

        any_moving = False
        for i, (r, c) in enumerate(cur_pos):
            cur[r][c] = i + 1
            ctg[r][c] = dist_fields[i][r][c]
            dxi, dyi = cost_gradient(dist_fields[i], grid)
            gdx[r][c] = dxi[r][c]
            gdy[r][c] = dyi[r][c]
            nxt = inst.paths[i][t + 1]
            out[r][c] = action_label((r, c), nxt)
            mask[r][c] = 1.0
            if inst.paths[i][t] != inst.paths[i][t + 1] or inst.paths[i][t] != inst.goals[i]:
                any_moving = True

        if not any_moving:
            continue

        # --- v2 new channels --------------------------------------------------
        density = local_density(cur, k=5)
        pred_occ = predicted_next_occupancy(grid, cur_pos, dist_fields)
        rem_cost = remaining_cost_per_cell(cur_pos, dist_fields, (H, W))

        F_in = np.zeros((NUM_FEATURE_CHANNELS_V2, Hp, Wp), dtype=np.float32)
        F_in[0, :, :] = padded
        F_in[1, :H, :W] = cur
        F_in[2, :H, :W] = goal_chan
        F_in[3, :H, :W] = ctg
        F_in[4, :H, :W] = gdx
        F_in[5, :H, :W] = gdy
        F_in[6, :H, :W] = density
        F_in[7, :H, :W] = pred_occ
        F_in[8, :H, :W] = rem_cost

        yield F_in, out, mask


def normalize_features_v2(F_in: np.ndarray) -> np.ndarray:
    """Light normalization for v2 channels.

    Mirrors v1's normalize_features for channels 0-5, plus log1p on the three
    new channels which can have wide dynamic range (density up to k*k=25,
    pred_occ up to num_agents, rem_cost up to H*W).
    """
    out = F_in.copy()
    out[1] = (F_in[1] > 0).astype(np.float32)
    out[2] = (F_in[2] > 0).astype(np.float32)
    out[3] = np.log1p(np.clip(F_in[3], 0, None))
    out[6] = np.log1p(np.clip(F_in[6], 0, None))
    out[7] = np.log1p(np.clip(F_in[7], 0, None))
    out[8] = np.log1p(np.clip(F_in[8], 0, None))
    return out

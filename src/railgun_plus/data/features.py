"""Stage 3: convert an `Instance` into RAILGUN-style training tensors.

This is the ROUTE-AGNOSTIC heart of the pipeline. It consumes only the
uniform `Instance` format, so it is identical whether the expert was PIBT
(Route C) or LaCAM (Route A).

For each timestep t of a solved instance we build:
  - F_in:  (k, H, W)  input feature stack
  - F_out: (H, W)     integer action label per cell (0..4), and
  - mask:  (H, W)     1 where an agent stands (loss is computed only here)

RAILGUN's five feature types (paper Sec. IV-C). We use channels-first
(C, H, W) because that's PyTorch's conv convention.

  channel 0: map            (1 = obstacle, 0 = free)
  channel 1: current locs   (agent index at its current cell, else 0)
  channel 2: goal locs      (agent index at its goal cell, else 0)
  channel 3: cost-to-goal   (BFS distance to goal for the agent at this cell)
  channels 4,5: cost-to-goal gradient (dx, dy) — the discrete direction that
                decreases cost-to-goal, per RAILGUN Sec IV-C.

NOTE on padding: a 5-level U-Net halves H,W four times, so both must be
divisible by 16. `pad_to_multiple` handles this and returns a validity mask
so padded cells never contribute to the loss.
"""
from __future__ import annotations

import numpy as np

from .types import (Instance, ACTION_DELTA, DELTA_TO_ACTION,
                    WAIT, UP, DOWN, LEFT, RIGHT)
from .grid_utils import bfs_distance_field

NUM_FEATURE_CHANNELS = 6  # k (see channel list above)
NUM_ACTIONS = 5


def pad_to_multiple(grid: np.ndarray, multiple: int = 16):
    """Pad (H,W) grid up to multiples of `multiple` with OBSTACLE cells.

    Returns (padded_grid, valid_mask, (H, W)). valid_mask is 1 on original
    cells, 0 on padding. Padding with obstacles (value 1) is safe: no agent
    is ever placed there, and the loss mask excludes them anyway.
    """
    H, W = grid.shape
    Hp = ((H + multiple - 1) // multiple) * multiple
    Wp = ((W + multiple - 1) // multiple) * multiple
    padded = np.ones((Hp, Wp), dtype=grid.dtype)  # 1 = obstacle everywhere
    padded[:H, :W] = grid
    valid = np.zeros((Hp, Wp), dtype=np.float32)
    valid[:H, :W] = 1.0
    return padded, valid, (H, W)


def cost_gradient(dist: np.ndarray, grid: np.ndarray):
    """Discrete gradient of the cost-to-goal field (RAILGUN Sec IV-C).

    For each free cell, dx in {-1,0,1} points toward the lower-cost horizontal
    neighbor; dy likewise vertically. Returns (dx, dy) float arrays.
    """
    H, W = grid.shape
    dx = np.zeros((H, W), dtype=np.float32)
    dy = np.zeros((H, W), dtype=np.float32)
    INF = dist.max()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                continue
            left  = dist[r][c - 1] if c > 0 else INF
            right = dist[r][c + 1] if c < W - 1 else INF
            up    = dist[r - 1][c] if r > 0 else INF
            down  = dist[r + 1][c] if r < H - 1 else INF
            d_l = left - dist[r][c]
            d_r = right - dist[r][c]
            d_u = up - dist[r][c]
            d_d = down - dist[r][c]
            # delta < 0 means that direction approaches the goal.
            if d_l >= 0 and d_r >= 0:
                dx[r][c] = 0
            elif d_l >= 0 and d_r < 0:
                dx[r][c] = 1
            elif d_l < 0 and d_r >= 0:
                dx[r][c] = -1
            else:
                dx[r][c] = 0  # both decrease: ambiguous, leave neutral
            if d_u >= 0 and d_d >= 0:
                dy[r][c] = 0
            elif d_u >= 0 and d_d < 0:
                dy[r][c] = 1
            elif d_u < 0 and d_d >= 0:
                dy[r][c] = -1
            else:
                dy[r][c] = 0
    return dx, dy


def action_label(prev_cell, next_cell) -> int:
    """The action index that moves prev_cell -> next_cell."""
    dr = next_cell[0] - prev_cell[0]
    dc = next_cell[1] - prev_cell[1]
    return DELTA_TO_ACTION[(dr, dc)]


def instance_to_samples(inst: Instance, pad_multiple: int = 16):
    """Yield (F_in, F_out, mask) tuples — one per timestep transition.

    F_in:  float32 (k, Hp, Wp)
    F_out: int64   (Hp, Wp)   action label at each occupied cell, else 0
    mask:  float32 (Hp, Wp)   1 at occupied cells (where loss applies)
    """
    grid = inst.grid
    padded, valid, (H, W) = pad_to_multiple(grid, pad_multiple)
    Hp, Wp = padded.shape

    # Per-agent distance fields (also the cost-to-goal feature source).
    dist_fields = [bfs_distance_field(grid, g) for g in inst.goals]

    # Goal-location channel is static across timesteps.
    goal_chan = np.zeros((H, W), dtype=np.float32)
    for i, (gr, gc) in enumerate(inst.goals):
        goal_chan[gr][gc] = i + 1  # 1-indexed agent id, 0 == empty

    T = inst.horizon
    for t in range(T - 1):  # transition t -> t+1
        cur = np.zeros((H, W), dtype=np.float32)
        ctg = np.zeros((H, W), dtype=np.float32)
        gdx = np.zeros((H, W), dtype=np.float32)
        gdy = np.zeros((H, W), dtype=np.float32)
        out = np.zeros((Hp, Wp), dtype=np.int64)
        mask = np.zeros((Hp, Wp), dtype=np.float32)

        any_moving = False
        for i, path in enumerate(inst.paths):
            r, c = path[t]
            cur[r][c] = i + 1
            ctg[r][c] = dist_fields[i][r][c]
            # gradient at this cell, for this agent's field
            dxi, dyi = cost_gradient(dist_fields[i], grid)
            gdx[r][c] = dxi[r][c]
            gdy[r][c] = dyi[r][c]
            # label = the action the expert took at this step
            nxt = path[t + 1]
            out[r][c] = action_label((r, c), nxt)
            mask[r][c] = 1.0
            if path[t] != path[t + 1] or path[t] != inst.goals[i]:
                any_moving = True

        # Skip fully-static frames (all agents resting on goals): no signal.
        if not any_moving:
            continue

        F_in = np.zeros((NUM_FEATURE_CHANNELS, Hp, Wp), dtype=np.float32)
        F_in[0, :, :] = padded                      # map
        F_in[1, :H, :W] = cur                        # current locations
        F_in[2, :H, :W] = goal_chan                  # goal locations
        F_in[3, :H, :W] = ctg                        # cost-to-goal
        F_in[4, :H, :W] = gdx                        # gradient dx
        F_in[5, :H, :W] = gdy                        # gradient dy

        yield F_in, out, mask


def normalize_features(F_in: np.ndarray) -> np.ndarray:
    """Light normalization so channels are on comparable scales.

    Agent-index channels can be large (up to #agents); cost-to-goal can be
    up to H*W. We scale the index channels to {0,1} occupancy + keep the id
    in a separate normalized form, and log-scale cost-to-goal. Kept simple
    and explicit so you can tweak it during ablations.
    """
    out = F_in.copy()
    # current/goal: convert to binary occupancy (presence), drop raw id scale.
    # (RAILGUN found raw indices workable; binary is a safe, stable default.
    #  You can ablate this later.)
    out[1] = (F_in[1] > 0).astype(np.float32)
    out[2] = (F_in[2] > 0).astype(np.float32)
    # cost-to-goal: log1p to compress the dynamic range.
    out[3] = np.log1p(np.clip(F_in[3], 0, None))
    return out

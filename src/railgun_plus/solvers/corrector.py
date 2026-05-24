"""Inference rollouts: the core experimental comparison of the project.

Two ways to turn a trained map-based policy into a full MAPF solution:

  greedy_rollout    -- baseline RAILGUN behaviour. Each step, sample/argmax a
                       per-cell action and apply it. Vertex/swap conflicts are
                       resolved by a naive "stay put on conflict" rule. This is
                       what produces the DEADLOCK COLLAPSE at high density.

  corrected_rollout -- THE CONTRIBUTION. Feed the network's per-cell action
                       probabilities into PIBT as its preference ordering.
                       PIBT's priority inheritance + backtracking returns a
                       guaranteed collision-free joint move, breaking deadlocks
                       the greedy rule cannot.

Both consume the SAME trained model, so any CSR/SoC gap between them is
attributable to the corrector alone — the clean ablation that isolates the
contribution from expert-data quality.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch

from ..data.types import Instance, ACTION_DELTA
from ..data.features import (pad_to_multiple, instance_to_samples,
                             normalize_features, NUM_FEATURE_CHANNELS)
from ..solvers.pibt import PIBT
from ..data.grid_utils import bfs_distance_field, neighbors

Cell = tuple[int, int]


def _build_input_tensor(grid, cur_pos, goals, dist_fields, device):
    """Construct the (1, k, Hp, Wp) network input for the CURRENT state.

    Mirrors data/features.py but for a live state (no expert label needed).
    """
    from ..data.features import cost_gradient
    padded, valid, (H, W) = pad_to_multiple(grid, 16)
    Hp, Wp = padded.shape

    cur = np.zeros((H, W), dtype=np.float32)
    ctg = np.zeros((H, W), dtype=np.float32)
    gdx = np.zeros((H, W), dtype=np.float32)
    gdy = np.zeros((H, W), dtype=np.float32)
    goal_chan = np.zeros((H, W), dtype=np.float32)
    for i, (gr, gc) in enumerate(goals):
        goal_chan[gr][gc] = i + 1

    grads = [cost_gradient(dist_fields[i], grid) for i in range(len(goals))]
    for i, (r, c) in enumerate(cur_pos):
        cur[r][c] = i + 1
        ctg[r][c] = dist_fields[i][r][c]
        gdx[r][c] = grads[i][0][r][c]
        gdy[r][c] = grads[i][1][r][c]

    F = np.zeros((NUM_FEATURE_CHANNELS, Hp, Wp), dtype=np.float32)
    F[0] = padded
    F[1, :H, :W] = cur
    F[2, :H, :W] = goal_chan
    F[3, :H, :W] = ctg
    F[4, :H, :W] = gdx
    F[5, :H, :W] = gdy
    F = normalize_features(F)
    return torch.from_numpy(F).unsqueeze(0).to(device), (H, W)


@torch.no_grad()
def _policy_probs(model, grid, cur_pos, goals, dist_fields, device):
    """Run the model; return per-agent action-probability vectors (len 5)."""
    x, (H, W) = _build_input_tensor(grid, cur_pos, goals, dist_fields, device)
    logits = model(x)[0]                       # (5, Hp, Wp)
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    per_agent = []
    for (r, c) in cur_pos:
        per_agent.append(probs[:, r, c])       # (5,) at the agent's cell
    return per_agent


def greedy_rollout(model, inst: Instance, device="cpu",
                   max_steps: Optional[int] = None, stochastic=False):
    """Baseline rollout. Returns (success: bool, paths)."""
    grid, goals = inst.grid, inst.goals
    n = inst.num_agents
    dist_fields = [bfs_distance_field(grid, g) for g in goals]
    cur = list(inst.starts)
    paths = [[s] for s in cur]
    T = max_steps or (inst.horizon * 3)
    rng = np.random.default_rng(0)

    for _ in range(T):
        if all(cur[i] == goals[i] for i in range(n)):
            break
        per_agent = _policy_probs(model, grid, cur, goals, dist_fields, device)
        # Each agent independently picks an action, then we resolve conflicts
        # with the naive rule: if a chosen next-cell collides, the agent waits.
        desired = []
        for i, p in enumerate(per_agent):
            a = int(np.argmax(p)) if not stochastic else int(rng.choice(5, p=p / p.sum()))
            dr, dc = ACTION_DELTA[a]
            nr, nc = cur[i][0] + dr, cur[i][1] + dc
            H, W = grid.shape
            if not (0 <= nr < H and 0 <= nc < W) or grid[nr][nc] != 0:
                nr, nc = cur[i]   # illegal -> wait
            desired.append((nr, nc))
        # Naive conflict resolution (the source of deadlocks):
        next_pos = list(cur)
        claimed: dict[Cell, int] = {}
        for i in range(n):
            d = desired[i]
            if d == cur[i]:
                next_pos[i] = d
                claimed.setdefault(d, i)
        for i in range(n):
            d = desired[i]
            if d == cur[i]:
                continue
            swap = any(desired[j] == cur[i] and cur[j] == d for j in range(n))
            if d not in claimed and not swap:
                claimed[d] = i
                next_pos[i] = d
            else:
                next_pos[i] = cur[i]  # blocked -> wait (can deadlock forever)
        cur = next_pos
        for i in range(n):
            paths[i].append(cur[i])

    success = all(cur[i] == goals[i] for i in range(n))
    return success, paths


def corrected_rollout(model, inst: Instance, device="cpu",
                      max_steps: Optional[int] = None):
    """PIBT-corrected rollout (the contribution). Returns (success, paths).

    The network's softmax becomes PIBT's preference ordering: for each agent,
    candidate next-cells are sorted by the model's probability of the action
    that reaches them (ties broken by distance-to-goal). PIBT then guarantees a
    collision-free joint move.
    """
    grid, goals = inst.grid, inst.goals
    n = inst.num_agents
    dist_fields = [bfs_distance_field(grid, g) for g in goals]
    pibt = PIBT(grid, goals, dist_fields=dist_fields)
    cur = list(inst.starts)
    paths = [[s] for s in cur]
    T = max_steps or (inst.horizon * 3)

    for _ in range(T):
        if all(cur[i] == goals[i] for i in range(n)):
            break
        per_agent = _policy_probs(model, grid, cur, goals, dist_fields, device)

        def order_fn(agent: int, cur_cell: Cell, cands: list[Cell]) -> list[Cell]:
            # Map each candidate cell to the action that reaches it, then sort
            # by the network's probability for that action (descending).
            p = per_agent[agent]
            def score(cell):
                dr, dc = cell[0] - cur_cell[0], cell[1] - cur_cell[1]
                from ..data.types import DELTA_TO_ACTION
                a = DELTA_TO_ACTION.get((dr, dc))
                prob = p[a] if a is not None else -1.0
                # tie-break toward goal
                return (-prob, dist_fields[agent][cell[0]][cell[1]])
            return sorted(cands, key=score)

        cur = pibt.step(cur, order_fn=order_fn)
        for i in range(n):
            paths[i].append(cur[i])

    success = all(cur[i] == goals[i] for i in range(n))
    return success, paths

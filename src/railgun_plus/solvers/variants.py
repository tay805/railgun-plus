"""Step-1 corrector variants — same trained model, different ways of combining
the network's softmax with PIBT.

All variants share the SAME rollout loop (`_run_variant`). Only the agent-priority
function (`prio_fn`) and the candidate-ordering function (`order_fn`) differ.
This guarantees that any performance difference between variants is attributable
to the priority/ordering choice and NOT to subtle implementation drift.

Variants implemented:

  v_softmax_primary  -- our existing approach: softmax IS the candidate
                        ordering, ties broken by distance-to-goal.
                        Risk: a noisy softmax (we're at 88.6% val acc) can
                        OVERRIDE PIBT's smart defaults on easy decisions.

  v_softmax_tiebreak -- distance-to-goal is the PRIMARY ordering (PIBT default).
                        Softmax only breaks ties between equally-good cells.
                        Safer: never overrides PIBT when PIBT is confident.

  v_confidence_gated -- use softmax as primary ONLY if its max probability
                        exceeds a threshold (default 0.7); otherwise fall back
                        to PIBT default. Stops noisy predictions from polluting
                        easy decisions.

  v_priority_by_conf -- PIBT processes agents in priority order. Default is
                        farthest-from-goal first. This variant uses the
                        network's max-softmax confidence to set agent priority
                        (confident agents go first), with distance as tie-break.

A FIFTH option (no model needed, for comparison): pure pibt_only = `v_pibt`.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from ..data.types import Instance, DELTA_TO_ACTION
from ..data.grid_utils import bfs_distance_field
from .pibt import PIBT
from .corrector import _policy_probs

Cell = tuple[int, int]


def _action_prob(cur_cell: Cell, cand_cell: Cell, p: np.ndarray) -> float:
    """Probability the network assigns to the action that moves cur -> cand."""
    dr, dc = cand_cell[0] - cur_cell[0], cand_cell[1] - cur_cell[1]
    a = DELTA_TO_ACTION.get((dr, dc))
    return float(p[a]) if a is not None else 0.0


def _run_variant(model, inst: Instance, device,
                 make_order_fn, prio_fn=None,
                 max_steps: Optional[int] = None,
                 use_model: bool = True):
    """Shared rollout. The ONLY difference between variants is what
    make_order_fn and prio_fn return.

    make_order_fn(per_agent, dist_fields) -> order_fn(agent, cur, cands) -> sorted cands
    prio_fn(per_agent, cur, dist_fields)  -> list[int] of agent indices in
        priority order (PIBT processes them in this order each step).
    use_model: if False, skip the network call (for pure-PIBT comparison).
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

        per_agent = (_policy_probs(model, grid, cur, goals, dist_fields, device)
                     if use_model else None)

        order_fn = make_order_fn(per_agent, dist_fields)

        # If a custom priority is provided, monkey-patch PIBT's step to use it.
        # Otherwise PIBT uses its default (farthest-from-goal first).
        if prio_fn is not None and per_agent is not None:
            priority = prio_fn(per_agent, cur, dist_fields)
            cur = _pibt_step_with_priority(pibt, cur, order_fn, priority)
        else:
            cur = pibt.step(cur, order_fn=order_fn)

        for i in range(n):
            paths[i].append(cur[i])

    success = all(cur[i] == goals[i] for i in range(n))
    return success, paths


def _pibt_step_with_priority(pibt: PIBT, cur_pos, order_fn, priority):
    """Run one PIBT step with a CUSTOM agent priority order."""
    occupied_now = {pos: i for i, pos in enumerate(cur_pos)}
    reserved_next: dict[Cell, int] = {}
    proposed: dict[int, Cell] = {}
    for a in priority:
        if a not in proposed:
            ok = pibt._pibt_recursive(a, occupied_now, reserved_next,
                                      proposed, order_fn, cur_pos)
            if not ok:
                proposed[a] = cur_pos[a]
                reserved_next.setdefault(cur_pos[a], a)
    return [proposed[i] for i in range(pibt.n)]


# --- variant 1: softmax IS the primary ordering (current behaviour) -------
def v_softmax_primary(model, inst, device="cpu", max_steps=None):
    """Existing corrector. Kept here for apples-to-apples comparison."""
    def make_order(per_agent, dist_fields):
        def order_fn(agent, cur_cell, cands):
            p = per_agent[agent]
            return sorted(cands,
                          key=lambda c: (-_action_prob(cur_cell, c, p),
                                         dist_fields[agent][c[0]][c[1]]))
        return order_fn
    return _run_variant(model, inst, device, make_order, max_steps=max_steps)


# --- variant 2: distance primary, softmax tie-break -----------------------
def v_softmax_tiebreak(model, inst, device="cpu", max_steps=None):
    """Distance-to-goal is primary; softmax only orders cells with the SAME
    distance. The network can never override PIBT's smart default; it only
    helps decide when PIBT would otherwise tie-break arbitrarily."""
    def make_order(per_agent, dist_fields):
        def order_fn(agent, cur_cell, cands):
            p = per_agent[agent]
            return sorted(cands,
                          key=lambda c: (dist_fields[agent][c[0]][c[1]],
                                         -_action_prob(cur_cell, c, p)))
        return order_fn
    return _run_variant(model, inst, device, make_order, max_steps=max_steps)


# --- variant 3: confidence-gated --------------------------------------------
def v_confidence_gated(model, inst, device="cpu", max_steps=None,
                       conf_threshold: float = 0.7):
    """Use softmax as primary ONLY when max prob >= threshold; otherwise fall
    back to PIBT's distance default. Filters out noisy network predictions."""
    def make_order(per_agent, dist_fields):
        def order_fn(agent, cur_cell, cands):
            p = per_agent[agent]
            if float(p.max()) >= conf_threshold:
                return sorted(cands,
                              key=lambda c: (-_action_prob(cur_cell, c, p),
                                             dist_fields[agent][c[0]][c[1]]))
            else:
                return sorted(cands,
                              key=lambda c: dist_fields[agent][c[0]][c[1]])
        return order_fn
    return _run_variant(model, inst, device, make_order, max_steps=max_steps)


# --- variant 4: agent priority informed by confidence ---------------------
def v_priority_by_conf(model, inst, device="cpu", max_steps=None):
    """Network confidence determines which AGENT goes first in PIBT each step;
    candidate ordering stays distance-based. Tests whether the network is more
    useful for agent priority than for candidate ordering."""
    def make_order(per_agent, dist_fields):
        def order_fn(agent, cur_cell, cands):
            return sorted(cands, key=lambda c: dist_fields[agent][c[0]][c[1]])
        return order_fn
    def prio_fn(per_agent, cur, dist_fields):
        # higher max-prob -> higher priority; ties broken by farthest-from-goal
        return sorted(range(len(cur)),
                      key=lambda i: (-float(per_agent[i].max()),
                                     -int(dist_fields[i][cur[i][0]][cur[i][1]])))
    return _run_variant(model, inst, device, make_order, prio_fn=prio_fn,
                        max_steps=max_steps)


# --- registry for use in the harness --------------------------------------
VARIANTS = {
    "v1_softmax_primary":  v_softmax_primary,
    "v2_softmax_tiebreak": v_softmax_tiebreak,
    "v3_conf_gated":       v_confidence_gated,
    "v4_prio_by_conf":     v_priority_by_conf,
}

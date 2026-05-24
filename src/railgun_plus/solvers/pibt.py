"""Pure-Python PIBT (Priority Inheritance with Backtracking).

Okumura et al., "Priority Inheritance with Backtracking for Iterative
Multi-Agent Path Finding", IJCAI 2019 / Artificial Intelligence 2022.

This single module is used in TWO places, which is the whole reason Route C
is efficient:

  1. As an EXPERT DATA GENERATOR (Stage 2, Route C): run PIBT to completion
     to produce full collision-free paths -> Instance objects.

  2. As the INFERENCE CORRECTOR (Stage 5, your contribution): run ONE PIBT
     timestep, but instead of PIBT's built-in distance-based preference, use
     the U-Net's per-cell action probabilities as the preference ordering.
     PIBT's priority-inheritance + backtracking then guarantees a
     collision-free joint move, breaking the deadlocks that a greedy
     argmax rollout cannot.

The key extensibility point is `priority_order_fn`: a callable that, given an
agent and its current cell, returns candidate next-cells in *preferred* order.
- For data generation we pass a distance-to-goal ordering (vanilla PIBT).
- For the corrector we pass an ordering derived from the network's softmax.

PIBT guarantees collision-freeness (no vertex/swap conflicts) on the move it
returns. Reachability/completeness holds on biconnected graphs; on general
grids it is a strong, fast heuristic — exactly what we want as a corrector.
"""
from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from ..data.types import Instance, ACTION_DELTA, pad_paths_to_equal_length
from ..data.grid_utils import neighbors, bfs_distance_field

Cell = tuple[int, int]


# A preference function: given (agent_index, current_cell, candidate_cells),
# return candidate_cells sorted from most- to least-preferred.
PriorityOrderFn = Callable[[int, Cell, list[Cell]], list[Cell]]


class PIBT:
    """One-step PIBT planner. Call `.step()` repeatedly to roll out a solution,
    or once per timestep to correct a learned policy."""

    def __init__(self, grid: np.ndarray, goals: list[Cell],
                 dist_fields: Optional[list[np.ndarray]] = None,
                 rng: Optional[np.random.Generator] = None):
        self.grid = grid
        self.goals = goals
        self.n = len(goals)
        self.rng = rng or np.random.default_rng(0)
        # Precompute a distance field per agent (used by the default ordering
        # and reused as the cost-to-goal feature later).
        if dist_fields is None:
            dist_fields = [bfs_distance_field(grid, g) for g in goals]
        self.dist = dist_fields

    # --- default preference: greedy toward goal, random tie-break ----------
    def _default_order(self, agent: int, cur: Cell,
                       cands: list[Cell]) -> list[Cell]:
        d = self.dist[agent]
        # Sort by distance-to-goal; jitter ties so agents don't deadlock in lockstep.
        keyed = sorted(cands, key=lambda x: (d[x[0]][x[1]],
                                             self.rng.random()))
        return keyed

    def _pibt_recursive(self, agent: int, occupied_now: dict[Cell, int],
                        reserved_next: dict[Cell, int],
                        proposed: dict[int, Cell],
                        order_fn: PriorityOrderFn,
                        cur_pos: list[Cell]) -> bool:
        """Try to assign agent a next cell. Returns True on success.

        occupied_now:  cell -> agent currently standing there (this timestep)
        reserved_next: cell -> agent that has claimed it for next timestep
        proposed:      agent -> its chosen next cell (filled in as we go)
        """
        cur = cur_pos[agent]
        cands = order_fn(agent, cur, neighbors(self.grid, cur))
        for nxt in cands:
            # Vertex conflict: someone already reserved this cell for next step.
            if nxt in reserved_next:
                continue
            # Swap conflict: the agent currently in `nxt` wants to move into `cur`.
            occ = occupied_now.get(nxt)
            if occ is not None and occ != agent and proposed.get(occ) == cur:
                continue
            # Tentatively claim nxt.
            reserved_next[nxt] = agent
            proposed[agent] = nxt
            # If another agent currently occupies nxt and hasn't moved yet,
            # it must vacate -> priority inheritance (recurse).
            if occ is not None and occ != agent and occ not in proposed:
                if not self._pibt_recursive(occ, occupied_now, reserved_next,
                                            proposed, order_fn, cur_pos):
                    # Backtrack: that agent couldn't move, undo our claim.
                    del reserved_next[nxt]
                    del proposed[agent]
                    continue
            return True
        return False

    def step(self, cur_pos: list[Cell],
             order_fn: Optional[PriorityOrderFn] = None) -> list[Cell]:
        """Compute the next joint position from `cur_pos`.

        order_fn: preference ordering. If None, uses greedy-to-goal (vanilla
        PIBT, for data generation). For the corrector, pass an order_fn built
        from the network's softmax (see solvers/corrector.py).
        """
        order_fn = order_fn or self._default_order
        occupied_now = {pos: i for i, pos in enumerate(cur_pos)}
        reserved_next: dict[Cell, int] = {}
        proposed: dict[int, Cell] = {}
        # Higher priority to agents farther from goal (classic PIBT heuristic).
        priority = sorted(range(self.n),
                          key=lambda a: -self.dist[a][cur_pos[a][0]][cur_pos[a][1]])
        for a in priority:
            if a not in proposed:
                ok = self._pibt_recursive(a, occupied_now, reserved_next,
                                          proposed, order_fn, cur_pos)
                if not ok:
                    proposed[a] = cur_pos[a]  # forced wait
                    reserved_next.setdefault(cur_pos[a], a)
        return [proposed[i] for i in range(self.n)]

    def solve(self, starts: list[Cell], max_steps: int = 512
              ) -> Optional[list[list[Cell]]]:
        """Roll out vanilla PIBT to completion. Returns per-agent paths, or
        None if not all agents reached their goals within max_steps."""
        cur = list(starts)
        paths = [[s] for s in starts]
        for _ in range(max_steps):
            if all(cur[i] == self.goals[i] for i in range(self.n)):
                return pad_paths_to_equal_length(paths)
            cur = self.step(cur)
            for i in range(self.n):
                paths[i].append(cur[i])
        if all(cur[i] == self.goals[i] for i in range(self.n)):
            return pad_paths_to_equal_length(paths)
        return None  # timed out (treat as failure for this instance)


def solve_instance(grid: np.ndarray, starts: list[Cell], goals: list[Cell],
                   max_steps: int = 512, seed: int = 0) -> Optional[Instance]:
    """Convenience wrapper: solve one instance with vanilla PIBT.

    Returns an Instance (uniform format) on success, or None on failure.
    This is the Route-C expert data generator.
    """
    rng = np.random.default_rng(seed)
    pibt = PIBT(grid, goals, rng=rng)
    paths = pibt.solve(starts, max_steps=max_steps)
    if paths is None:
        return None
    inst = Instance(grid=grid, starts=list(starts), goals=list(goals), paths=paths)
    inst.validate()
    return inst

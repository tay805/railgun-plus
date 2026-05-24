"""The uniform intermediate format that ALL data routes emit.

Both Route C (PIBT) and Route A (LaCAM download) must produce a list of
`Instance` objects. Everything downstream (feature conversion, training,
evaluation) consumes only this — so it never knows or cares which expert
solver produced the data. Swapping data routes = swapping ONLY the code
that builds these objects.

Coordinate convention (used everywhere in this project):
  - A cell is (row, col), i.e. (y, x). row 0 is the top.
  - The grid is a 2D array of shape (H, W).
  - grid[r][c] == 1 means OBSTACLE (non-traversable),
    grid[r][c] == 0 means FREE.  (Same convention as RAILGUN's map feature.)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# The five actions, in a FIXED order used by the model's output channels.
# Index matches the last dimension of F_out (n, m, 5).
WAIT, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
ACTIONS = [WAIT, UP, DOWN, LEFT, RIGHT]

# (d_row, d_col) movement for each action index above.
ACTION_DELTA = {
    WAIT:  (0, 0),
    UP:    (-1, 0),
    DOWN:  (1, 0),
    LEFT:  (0, -1),
    RIGHT: (0, 1),
}

# Reverse lookup: (d_row, d_col) -> action index. Used when converting an
# expert path into per-step action labels.
DELTA_TO_ACTION = {v: k for k, v in ACTION_DELTA.items()}


@dataclass
class Instance:
    """One solved MAPF instance.

    Attributes:
        grid:   np.ndarray (H, W) of {0,1}. 1 = obstacle.
        starts: list of (row, col), one per agent. starts[i] is agent i's start.
        goals:  list of (row, col), one per agent.
        paths:  list of paths. paths[i] is agent i's path: a list of (row, col)
                cells, one per timestep, starting at starts[i]. Agents that have
                reached their goal stay there (the path is padded by repeating
                the goal cell) so that every path has the same length T.
    """
    grid: np.ndarray
    starts: list[tuple[int, int]]
    goals: list[tuple[int, int]]
    paths: list[list[tuple[int, int]]]

    @property
    def num_agents(self) -> int:
        return len(self.starts)

    @property
    def horizon(self) -> int:
        """Number of timesteps T (all paths share this length)."""
        return max(len(p) for p in self.paths)

    def validate(self) -> None:
        """Cheap sanity checks. Call this on freshly generated data — it catches
        the off-by-one / transpose bugs that otherwise waste hours of training."""
        H, W = self.grid.shape
        assert self.num_agents == len(self.goals) == len(self.paths), \
            "starts, goals, paths must have equal length"
        for i, path in enumerate(self.paths):
            assert path[0] == self.starts[i], f"agent {i}: path[0] != start"
            # Final cell should be the goal (paths are padded to rest at goal).
            assert path[-1] == self.goals[i], f"agent {i}: path[-1] != goal"
            for t, (r, c) in enumerate(path):
                assert 0 <= r < H and 0 <= c < W, f"agent {i} t={t}: off grid"
                assert self.grid[r][c] == 0, f"agent {i} t={t}: on obstacle"
                if t > 0:
                    pr, pc = path[t - 1]
                    dr, dc = r - pr, c - pc
                    assert (dr, dc) in DELTA_TO_ACTION, \
                        f"agent {i} t={t}: illegal move {(dr, dc)}"


def pad_paths_to_equal_length(paths: list[list[tuple[int, int]]]
                              ) -> list[list[tuple[int, int]]]:
    """Pad every path to the max length by repeating its final (goal) cell.

    Expert solvers return paths of differing lengths (agents finish at
    different times). We pad so that timestep indexing is uniform; a finished
    agent simply 'waits' on its goal, which is exactly the MAPF convention.
    """
    T = max(len(p) for p in paths)
    out = []
    for p in paths:
        if len(p) < T:
            p = p + [p[-1]] * (T - len(p))
        out.append(p)
    return out

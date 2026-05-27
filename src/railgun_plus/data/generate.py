"""Stage 2 (Route C): generate expert trajectories with PIBT on POGEMA maps.

This produces a list of `Instance` objects — the uniform format. To switch to
Route A (LaCAM) later, write a function with the SAME return type that loads
LaCAM-solved instances; everything downstream is unchanged.

POGEMA provides the maps and start/goal scenario generation. We solve each
scenario with vanilla PIBT (solvers/pibt.py) to get expert paths.

Requires: pip install pogema
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from .types import Instance
from ..solvers.pibt import solve_instance


def _grid_from_pogema(obstacles: np.ndarray) -> np.ndarray:
    """POGEMA obstacle grid -> our convention (1=obstacle, 0=free)."""
    return (np.asarray(obstacles) != 0).astype(np.int8)


def generate_with_pogema(num_instances: int,
                         map_size: int = 32,
                         obstacle_density: float = 0.2,
                         num_agents: int = 32,
                         seed: int = 0,
                         max_pibt_steps: int = 512) -> list[Instance]:
    """Generate `num_instances` solved instances on random POGEMA maps.

    Uses POGEMA's random-grid generation for maps and start/goal placement,
    then solves with PIBT. Instances PIBT fails to solve within max steps are
    skipped (so you may get slightly fewer than requested — generate extra).

    For maze/warehouse maps, swap the GridConfig parameters or use the
    pogema-toolbox map generators; the rest is identical.
    """
    try:
        from pogema import GridConfig
        from pogema.grid import Grid
    except ImportError as e:
        raise ImportError(
            "pogema is required for Route C data generation. "
            "Run: pip install pogema") from e

    rng = np.random.default_rng(seed)
    instances: list[Instance] = []
    attempts = 0
    while len(instances) < num_instances and attempts < num_instances * 4:
        attempts += 1
        s = int(rng.integers(0, 2**31 - 1))
        gc = GridConfig(size=map_size, density=obstacle_density,
                        num_agents=num_agents, seed=s,
                        max_episode_steps=max_pibt_steps,
                        obs_radius=5)
        grid_obj = Grid(gc)
        # Pull the static obstacle map and agent start/goal positions.
        obstacles = _grid_from_pogema(grid_obj.get_obstacles())
        starts = [tuple(map(int, p)) for p in grid_obj.get_agents_xy()]
        goals = [tuple(map(int, p)) for p in grid_obj.get_targets_xy()]

        # Skip degenerate placements (agent on obstacle, unreachable goal).
        ok = all(obstacles[r][c] == 0 for r, c in starts + goals)
        if not ok:
            continue

        inst = solve_instance(obstacles, starts, goals,
                              max_steps=max_pibt_steps, seed=s)
        if inst is not None:
            instances.append(inst)
    return instances


# ---- Route A (LaCAM) ------------------------------------------------------
# Implemented in lacam_expert.py. Re-exported here for convenience so callers
# can do `from railgun_plus.data.generate import generate_with_lacam`.
def generate_with_lacam(*args, **kwargs):
    """Generate solved instances using POGEMA maps + the LaCAM C++ expert.
    See data/lacam_expert.generate_with_lacam for the full signature
    (requires lacam_bin=path-to-compiled-binary)."""
    from .lacam_expert import generate_with_lacam as _g
    return _g(*args, **kwargs)

"""Fast sanity tests. Run `pytest` locally before pushing.

These catch the silent feature-construction bugs (transpose, off-by-one,
illegal moves) that otherwise waste hours of GPU time on Colab.
"""
import numpy as np

from railgun_plus.solvers.pibt import solve_instance, bfs_distance_field
from railgun_plus.data.features import instance_to_samples, NUM_FEATURE_CHANNELS
from railgun_plus.data.types import Instance


def make_corridor():
    # 3x5 free grid, no obstacles.
    grid = np.zeros((3, 5), dtype=np.int8)
    return grid


def test_bfs_distance():
    grid = make_corridor()
    d = bfs_distance_field(grid, (0, 0))
    assert d[0][0] == 0
    assert d[0][1] == 1
    assert d[1][0] == 1
    assert d[2][4] == 6  # manhattan in open grid


def test_pibt_solves_simple_swap():
    # Two agents that must pass each other in a wider area.
    grid = make_corridor()
    starts = [(0, 0), (0, 4)]
    goals = [(0, 4), (0, 0)]
    inst = solve_instance(grid, starts, goals, max_steps=64, seed=1)
    assert inst is not None, "PIBT failed to solve a trivial 2-agent instance"
    inst.validate()  # raises on any illegal move / collision-with-obstacle
    assert inst.paths[0][-1] == (0, 4)
    assert inst.paths[1][-1] == (0, 0)


def test_feature_shapes_and_mask():
    grid = make_corridor()
    starts = [(0, 0), (2, 4)]
    goals = [(2, 4), (0, 0)]
    inst = solve_instance(grid, starts, goals, max_steps=64, seed=2)
    assert inst is not None
    samples = list(instance_to_samples(inst))
    assert len(samples) > 0
    F_in, F_out, mask = samples[0]
    # padded to multiple of 16
    assert F_in.shape[0] == NUM_FEATURE_CHANNELS
    assert F_in.shape[1] % 16 == 0 and F_in.shape[2] % 16 == 0
    assert F_out.shape == mask.shape == F_in.shape[1:]
    # mask should mark exactly the occupied cells (== num agents at that step)
    assert mask.sum() == inst.num_agents
    # labels only where mask is set
    assert (F_out[mask == 0] == 0).all()


def test_no_collisions_in_pibt_output():
    grid = make_corridor()
    starts = [(0, 0), (0, 1), (0, 2)]
    goals = [(2, 4), (2, 3), (2, 2)]
    inst = solve_instance(grid, starts, goals, max_steps=128, seed=3)
    assert inst is not None
    T = inst.horizon
    for t in range(T):
        occupied = [inst.paths[i][t] for i in range(inst.num_agents)]
        assert len(set(occupied)) == len(occupied), f"vertex collision at t={t}"

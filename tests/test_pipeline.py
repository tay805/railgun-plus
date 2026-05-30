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


def test_metrics_lower_bound_and_ratio():
    """SoC ratio should be >= 1.0 and equal to soc/lower_bound on solved."""
    from railgun_plus.eval.metrics import (evaluate_instance, soc_lower_bound,
                                           summarize)
    grid = np.zeros((5, 7), dtype=np.int8)
    inst = solve_instance(grid, [(0, 0), (4, 6)], [(4, 6), (0, 0)],
                          max_steps=128, seed=11)
    assert inst is not None
    m = evaluate_instance(inst.paths, inst)
    assert m["success"]
    assert m["soc_lb"] > 0
    assert m["soc_ratio"] >= 1.0 - 1e-9   # can't beat the lower bound
    s = summarize([m])
    assert s["csr"] == 1.0
    assert s["n_solved"] == 1


def test_harness_expert_runs_without_model():
    """expert / pibt_only methods must run with model=None (no torch needed)."""
    from railgun_plus.eval.harness import run_method
    from railgun_plus.eval.metrics import summarize
    grid = np.zeros((5, 7), dtype=np.int8)
    insts = []
    for seed in range(3):
        inst = solve_instance(grid, [(0, 0), (4, 6)], [(4, 6), (0, 0)],
                              max_steps=128, seed=seed)
        if inst:
            insts.append(inst)
    res = run_method("expert", None, insts)
    s = summarize(res)
    assert s["n"] == len(insts)
    assert 0.0 <= s["csr"] <= 1.0


def test_local_density_matches_naive():
    """Catch any future regression in the v2 local_density integral-image impl."""
    from railgun_plus.data.features_v2 import local_density
    rng = np.random.default_rng(42)
    for _ in range(5):
        H = int(rng.integers(8, 20))
        W = int(rng.integers(8, 20))
        cur = (rng.random((H, W)) > 0.6).astype(np.float32) * \
              np.arange(1, H*W+1).reshape(H, W)
        r = 2  # k=5
        binc = (cur > 0).astype(np.float32)
        slow = np.zeros_like(binc)
        for i in range(H):
            for j in range(W):
                i0, i1 = max(0, i-r), min(H, i+r+1)
                j0, j1 = max(0, j-r), min(W, j+r+1)
                slow[i][j] = binc[i0:i1, j0:j1].sum()
        fast = local_density(cur, k=5)
        assert np.allclose(fast, slow), f"density mismatch: max diff {np.abs(fast-slow).max()}"


def test_features_v2_shapes():
    """v2 features must produce 9-channel inputs with correct shape."""
    from railgun_plus.data.features_v2 import (instance_to_samples_v2,
                                               NUM_FEATURE_CHANNELS_V2)
    grid = np.zeros((6, 8), dtype=np.int8)
    inst = solve_instance(grid, [(0, 0), (5, 7)], [(5, 7), (0, 0)],
                          max_steps=128, seed=0)
    assert inst is not None
    samples = list(instance_to_samples_v2(inst))
    assert len(samples) > 0
    F_in, F_out, mask = samples[0]
    assert F_in.shape[0] == NUM_FEATURE_CHANNELS_V2 == 9
    assert F_in.shape[1] % 16 == 0 and F_in.shape[2] % 16 == 0
    assert F_out.shape == mask.shape == F_in.shape[1:]

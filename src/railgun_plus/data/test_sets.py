"""Test-set generation that adapts density to actually produce N_PER instances.

The fixed-density approach (everything at density=0.2) fails at high agent counts
because POGEMA can't place 96+ agents on a 32x32 grid with 20% obstacles -- not
enough free cells. The result was test sets shrinking to 28/12 instances at
96/128 agents, making those numbers too noisy.

This module either:
  - lowers density automatically per agent count (more free space for crowding), or
  - retries with progressively lower densities until the target N is reached.

The headline plots stay comparable because the model was trained across the
same agent counts, and at evaluation what matters is *agent count*, not density.
"""
from __future__ import annotations

import os
import pickle

from .generate import generate_with_pogema


# A density schedule chosen so 32x32 maps can actually fit the agent count.
# Rule of thumb: leave at least 30% free cells per (start + goal) per agent.
# For a 32x32 = 1024-cell map and N agents we need >= 2N free cells, i.e.
# density <= 1 - 2N/1024.
DEFAULT_DENSITY_SCHEDULE = {
    16:  0.20,
    32:  0.20,
    48:  0.15,
    64:  0.15,
    80:  0.10,
    96:  0.10,
    128: 0.05,
}


def build_test_sets(agent_counts, n_per=50, map_size=32, seed_base=1000,
                    density_schedule=None, save_path=None):
    """Generate exactly n_per instances per agent count, resumable on disk.

    density_schedule: {agent_count: density}. If None, uses DEFAULT_DENSITY_SCHEDULE
    and the density of the *closest available* agent count for any missing key.

    save_path: if given, the dict is pickled here after every completed
    agent-count (so a disconnect doesn't lose what's done).

    Returns: {agent_count: [Instance, ...]}, each list of length exactly n_per
    UNLESS POGEMA hits the cap even at very low density (very rare; reported).
    """
    sched = density_schedule or DEFAULT_DENSITY_SCHEDULE
    test_sets = {}
    if save_path and os.path.exists(save_path):
        test_sets = pickle.load(open(save_path, "rb"))

    def pick_density(k):
        if k in sched:
            return sched[k]
        # nearest defined key
        key = min(sched.keys(), key=lambda kk: abs(kk - k))
        return sched[key]

    for k in agent_counts:
        existing = test_sets.get(k, [])
        if len(existing) >= n_per:
            print(f"agents={k:4d}: have {len(existing)} (skip)")
            continue
        # try descending densities if the first doesn't produce enough
        d0 = pick_density(k)
        for density in [d0, max(d0 - 0.05, 0.0), max(d0 - 0.1, 0.0)]:
            need = n_per - len(existing)
            print(f"agents={k:4d} density={density:.2f}: generating {need}...")
            more = generate_with_pogema(need, map_size, density, k,
                                        seed=seed_base + k + int(density * 1000))
            existing.extend(more)
            print(f"   -> now have {len(existing)}/{n_per}")
            if len(existing) >= n_per:
                break
        test_sets[k] = existing[:n_per]
        if save_path:
            pickle.dump(test_sets, open(save_path, "wb"))
    return test_sets

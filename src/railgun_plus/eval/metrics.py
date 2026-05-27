"""MAPF evaluation metrics, matching the definitions RAILGUN reports.

  CSR (Completion Success Rate): fraction of instances where ALL agents reach
       their goals. Higher is better.
  SoC (Sum of Costs): sum over agents of each agent's arrival time. Lower better.
  Makespan: the latest arrival time across all agents. Lower is better.
  SoC lower bound: sum of per-agent shortest-path distances ignoring other
       agents. The theoretical best SoC; used to NORMALIZE SoC so values are
       comparable across instances and maps (this is what the RAILGUN/POGEMA
       'Performance' metric uses as SoC_best's spirit).
"""
from __future__ import annotations

from ..data.grid_utils import bfs_distance_field


def soc(paths, goals) -> int:
    """Sum of individual arrival times."""
    total = 0
    for path, goal in zip(paths, goals):
        arrive = 0
        for t, cell in enumerate(path):
            if cell != goal:
                arrive = t + 1
        total += arrive
    return total


def makespan(paths, goals) -> int:
    m = 0
    for path, goal in zip(paths, goals):
        arrive = 0
        for t, cell in enumerate(path):
            if cell != goal:
                arrive = t + 1
        m = max(m, arrive)
    return m


def all_reached(paths, goals) -> bool:
    return all(path[-1] == goal for path, goal in zip(paths, goals))


def soc_lower_bound(grid, starts, goals) -> int:
    """Sum of per-agent shortest-path distances (ignoring collisions).

    This is the unattainable-best SoC. Reporting SoC / soc_lower_bound gives a
    map- and instance-independent quality ratio (>= 1.0; closer to 1 is better),
    which is far more meaningful than raw SoC and mirrors the paper's
    normalized 'Performance' metric.
    """
    total = 0
    for s, g in zip(starts, goals):
        d = bfs_distance_field(grid, g)
        total += int(d[s[0]][s[1]])
    return total


def evaluate_instance(paths, inst) -> dict:
    """Full per-instance metric bundle. `inst` is an Instance (for grid/start/goal
    and the lower bound)."""
    success = all_reached(paths, inst.goals)
    s = soc(paths, inst.goals)
    lb = soc_lower_bound(inst.grid, inst.starts, inst.goals)
    return {
        "success": success,
        "soc": s,
        "makespan": makespan(paths, inst.goals),
        "soc_lb": lb,
        # normalized SoC ratio, only meaningful when solved
        "soc_ratio": (s / lb) if (success and lb > 0) else None,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-instance dicts into headline numbers.

    IMPORTANT: avg_soc and avg_soc_ratio are computed over SOLVED instances
    only. Always read them alongside CSR — a method that solves nothing will
    show avg_soc of 0, which is an artifact of having nothing to average, NOT
    good performance. n_solved makes this explicit.
    """
    n = len(results)
    solved = [r for r in results if r["success"]]
    csr = len(solved) / n if n else 0.0
    avg_soc = sum(r["soc"] for r in solved) / len(solved) if solved else 0.0
    avg_mk = sum(r["makespan"] for r in solved) / len(solved) if solved else 0.0
    ratios = [r["soc_ratio"] for r in solved if r["soc_ratio"] is not None]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    # deadlock rate = fraction of instances NOT solved (agents stuck / livelocked).
    # This is the headline efficiency metric: greedy deadlocks often, the
    # corrector should deadlock rarely.
    deadlock_rate = 1.0 - csr
    return {
        "csr": csr,
        "deadlock_rate": deadlock_rate,
        "n": n,
        "n_solved": len(solved),
        "avg_soc_solved": avg_soc,
        "avg_makespan_solved": avg_mk,
        "avg_soc_ratio_solved": avg_ratio,  # >=1.0; closer to 1 is better
    }

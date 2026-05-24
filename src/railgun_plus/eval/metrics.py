"""MAPF evaluation metrics, matching the definitions RAILGUN reports.

  CSR (Completion Success Rate): fraction of instances where ALL agents reach
       their goals. Higher is better. This is where RAILGUN's greedy rollout
       collapses at high density and where the corrector should hold up.
  SoC (Sum of Costs): sum over agents of the timestep at which each agent
       *last* arrives at its goal (and stays). Lower is better.
  Makespan: the latest arrival time across all agents. Lower is better.
"""
from __future__ import annotations


def soc(paths, goals) -> int:
    """Sum of individual arrival times. Each agent's cost = last timestep at
    which it departs from / before settling on its goal."""
    total = 0
    for path, goal in zip(paths, goals):
        # find last index where the agent is NOT yet at goal, +1; if it starts
        # at goal, cost 0.
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


def summarize(results: list[dict]) -> dict:
    """Aggregate a list of per-instance result dicts into headline numbers."""
    n = len(results)
    solved = [r for r in results if r["success"]]
    csr = len(solved) / n if n else 0.0
    avg_soc = sum(r["soc"] for r in solved) / len(solved) if solved else 0.0
    avg_mk = sum(r["makespan"] for r in solved) / len(solved) if solved else 0.0
    return {"csr": csr, "avg_soc_solved": avg_soc,
            "avg_makespan_solved": avg_mk, "n": n, "n_solved": len(solved)}

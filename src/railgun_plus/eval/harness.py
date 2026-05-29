"""Evaluation harness: run multiple methods over agent-count sweeps and produce
paper-style tables and plots.

METHODS COMPARED (all runnable in Colab, no external systems):
  - "expert"    : the data-generating solver (PIBT here) run to completion.
                  This is your UPPER reference / oracle for solvability and SoC.
  - "pibt_only" : pure PIBT at inference, network ignored. Tells you how much
                  the learned policy adds on top of raw PIBT.
  - "greedy"    : the trained network with naive conflict resolution (baseline
                  RAILGUN behaviour) -- exhibits the deadlock collapse.
  - "corrected" : the trained network + PIBT corrector (the contribution).

WHAT IS NOT HERE (and why): SCRIMP / DCC / MAMBA / MAPF-GPT are separate
trained systems. Reproducing them in Colab is its own multi-week effort. The
honest way to compare against them is to run THIS model through POGEMA's
official benchmark harness (pogema-toolbox), which already tabulates those
baselines' scores. See `pogema_benchmark_stub()` at the bottom.
"""
from __future__ import annotations

import csv
import os

from .metrics import evaluate_instance, summarize
from ..solvers.pibt import PIBT
from ..data.grid_utils import bfs_distance_field

# NOTE: greedy_rollout / corrected_rollout are imported lazily inside
# run_method, because they pull in torch. This lets the expert / pibt_only
# reference methods run in environments without torch.


_LACAM_BIN_ENV = "RAILGUN_LACAM_BIN"


def _expert_paths(inst, lacam_bin=None):
    """Run the EXPERT solver to completion on an instance.

    If a LaCAM binary path is given (explicitly or via the RAILGUN_LACAM_BIN
    environment variable), invoke LaCAM — the near-optimal expert. Otherwise
    fall back to PIBT. This makes `expert` a meaningful upper-reference line
    that is DISTINCT from `pibt_only` when LaCAM is available.
    """
    if lacam_bin is None:
        lacam_bin = os.environ.get(_LACAM_BIN_ENV)
    if lacam_bin and os.path.exists(lacam_bin):
        from ..data.lacam_expert import solve_with_lacam
        sol = solve_with_lacam(inst.grid, inst.starts, inst.goals,
                               lacam_bin=lacam_bin, time_limit_ms=10000)
        if sol is not None:
            return sol.paths
        # LaCAM failed on this instance -> fall through to PIBT
    pibt = PIBT(inst.grid, inst.goals)
    paths = pibt.solve(inst.starts, max_steps=inst.horizon * 3)
    if paths is None:
        return [[s] for s in inst.starts]
    return paths


def _pibt_only_paths(inst):
    """Pure PIBT at inference (no network). This is the BASELINE TO BEAT —
    distinct from `expert` once a LaCAM binary is set."""
    pibt = PIBT(inst.grid, inst.goals)
    paths = pibt.solve(inst.starts, max_steps=inst.horizon * 3)
    if paths is None:
        return [[s] for s in inst.starts]
    return paths


def run_method(method: str, model, instances, device="cpu", lacam_bin=None):
    """Run one method over a list of instances; return list of metric dicts."""
    results = []
    for inst in instances:
        if method == "expert":
            paths = _expert_paths(inst, lacam_bin=lacam_bin)
        elif method == "pibt_only":
            paths = _pibt_only_paths(inst)
        elif method == "greedy":
            from ..solvers.corrector import greedy_rollout
            _, paths = greedy_rollout(model, inst, device=device)
        elif method == "corrected":
            from ..solvers.corrector import corrected_rollout
            _, paths = corrected_rollout(model, inst, device=device)
        else:
            raise ValueError(f"unknown method {method}")
        results.append(evaluate_instance(paths, inst))
    return results


def run_sweep(model, test_sets: dict, methods=None, device="cpu",
              lacam_bin=None):
    """Run all methods across all agent counts.

    lacam_bin: path to compiled LaCAM binary. If given, the `expert` method
        uses LaCAM (the near-optimal upper reference). If None, expert == PIBT.
    """
    methods = methods or ["expert", "pibt_only", "greedy", "corrected"]
    out = {m: {} for m in methods}
    for k in sorted(test_sets):
        for m in methods:
            res = run_method(m, model, test_sets[k], device=device,
                             lacam_bin=lacam_bin)
            out[m][k] = summarize(res)
            s = out[m][k]
            print(f"  [{m:10s}] agents={k:4d}  CSR={s['csr']:.2f}  "
                  f"SoC_ratio={s['avg_soc_ratio_solved']:.2f}  "
                  f"(solved {s['n_solved']}/{s['n']})")
    return out


def sweep_to_table(sweep: dict, out_csv: str = None):
    """Flatten a sweep into rows; optionally write a CSV (paper-style table)."""
    rows = []
    methods = list(sweep.keys())
    agent_counts = sorted(next(iter(sweep.values())).keys())
    for k in agent_counts:
        for m in methods:
            s = sweep[m][k]
            rows.append({
                "method": m, "agents": k, "csr": round(s["csr"], 3),
                "deadlock_rate": round(s["deadlock_rate"], 3),
                "avg_soc_ratio": round(s["avg_soc_ratio_solved"], 3),
                "avg_makespan": round(s["avg_makespan_solved"], 1),
                "n_solved": s["n_solved"], "n": s["n"],
            })
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote table -> {out_csv}")
    return rows


def plot_sweep(sweep: dict, metric="csr", title=None, savepath=None):
    """Plot one metric vs agent count, one line per method. Requires matplotlib.

    metric: "csr" | "avg_soc_ratio_solved" | "avg_makespan_solved"
    """
    import matplotlib.pyplot as plt
    styles = {"expert": ("k:", "Expert (PIBT, oracle)"),
              "pibt_only": ("^-.", "Pure PIBT (no net)"),
              "greedy": ("o--", "Greedy (baseline RAILGUN)"),
              "corrected": ("s-", "PIBT-corrected (ours)")}
    plt.figure(figsize=(7, 5))
    for m, data in sweep.items():
        ks = sorted(data)
        ys = [data[k][metric] for k in ks]
        fmt, label = styles.get(m, ("x-", m))
        plt.plot(ks, ys, fmt, label=label)
    plt.xlabel("Number of agents")
    ylabels = {"csr": "CSR (success rate)",
               "deadlock_rate": "Deadlock rate (1 - CSR; lower=better)",
               "avg_soc_ratio_solved": "SoC / lower-bound (solved; 1.0=optimal)",
               "avg_makespan_solved": "Avg makespan (solved)"}
    plt.ylabel(ylabels.get(metric, metric))
    plt.title(title or f"{metric} vs agents")
    if metric in ("csr", "deadlock_rate"):
        plt.ylim(0, 1.05)
    plt.legend(); plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"saved figure -> {savepath}")
    plt.show()


# ---------------------------------------------------------------------------
def pogema_benchmark_stub():
    """STUB / ROADMAP for comparing against SCRIMP, DCC, MAMBA, MAPF-GPT.

    DO NOT try to train those models here. Instead, the standard, honest path:

    1. Wrap THIS model as a POGEMA-compatible agent (implement an `act(obs)`
       interface using the corrected rollout).
    2. Use pogema-toolbox's evaluation harness + the official POGEMA benchmark
       map/scenario configs (the same eval_configs the MAPF-GPT repo ships).
    3. pogema-toolbox already includes/reports the baselines' results, so your
       model's scores land on the SAME radar (Performance, Coordination,
       Scalability, Cooperation, OOD, Pathfinding) as in the RAILGUN paper.

    This keeps the comparison apples-to-apples and avoids re-running other
    people's systems. Implement when the scaled-up Route-C model is solid.
    """
    raise NotImplementedError(
        "POGEMA-harness baseline comparison not wired up yet. See docstring "
        "for the intended approach (wrap model as a pogema agent, run via "
        "pogema-toolbox against official configs).")

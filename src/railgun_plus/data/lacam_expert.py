"""Route A expert: LaCAM (Okumura, AAAI 2023 / LaCAM* IJCAI 2023).

LaCAM is a C++ solver. The workflow (Kaggle):
  1. Compile it ONCE (see scripts/build_lacam.sh) and save the binary as a
     Kaggle dataset so you never recompile.
  2. This module shells out to that binary: it writes each instance as MovingAI
     .map/.scen files, runs LaCAM, and parses the solution back into our
     uniform `Instance` format. Everything downstream is unchanged.

Why LaCAM and not PIBT as the teacher:
  PIBT is available for free at inference (it's our corrector), so a network
  that merely imitates PIBT cannot beat PIBT. LaCAM is near-SoC-optimal but too
  slow to run per-step at inference. Distilling LaCAM into the fast CNN, then
  using PIBT only as a safety corrector, lets "net + PIBT" exceed "PIBT alone"
  -- which is the positive result the pure-PIBT ablation is designed to test.

IMPORTANT (format fragility): different LaCAM forks emit slightly different
solution text. `parse_lacam_solution` targets the common
"<agent>:(x,y),(x,y),..." line format used by Okumura's reference repo. If your
cloned version differs, that one function is the only thing to adjust (same
kind of one-line fix as the pogema method names).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import numpy as np

from .types import Instance, pad_paths_to_equal_length
from .grid_utils import bfs_distance_field


# --- MovingAI format writers ----------------------------------------------
def write_map_file(grid: np.ndarray, path: str) -> None:
    """Write a MovingAI .map file. Our convention: grid==1 obstacle, 0 free.
    MovingAI uses '.' for free / passable and '@' (or 'T') for obstacle."""
    H, W = grid.shape
    with open(path, "w") as f:
        f.write("type octile\n")
        f.write(f"height {H}\n")
        f.write(f"width {W}\n")
        f.write("map\n")
        for r in range(H):
            row = "".join("@" if grid[r][c] == 1 else "." for c in range(W))
            f.write(row + "\n")


def write_scen_file(starts, goals, map_name, grid, path: str) -> None:
    """Write a MovingAI .scen file. Columns:
    bucket  map  width  height  sx sy gx gy  optimal_length
    NOTE MovingAI uses (col,row) = (x,y) ordering in scen files; our cells are
    (row,col), so we swap when writing."""
    H, W = grid.shape
    with open(path, "w") as f:
        f.write("version 1\n")
        for (sr, sc), (gr, gc) in zip(starts, goals):
            d = bfs_distance_field(grid, (gr, gc))[sr][sc]
            # bucket, map, w, h, sx, sy, gx, gy, dist   (x=col, y=row)
            f.write(f"0\t{map_name}\t{W}\t{H}\t{sc}\t{sr}\t{gc}\t{gr}\t"
                    f"{float(d):.4f}\n")


# --- solution parser -------------------------------------------------------
_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+)$")
_COORD_RE = re.compile(r"\((\d+),\s*(\d+)\)")


def parse_lacam_solution(text: str, num_agents: int):
    """Parse LaCAM stdout/solution file into per-agent path lists of (row,col).

    Expected per-agent line format (Okumura reference repo):
        <t>:(x,y),(x,y),...      OR      <agent>:(x,y),(x,y),...
    Most forks print one line per TIMESTEP listing all agents' positions; some
    print one line per AGENT. We auto-detect: if the number of coordinate lines
    equals num_agents we treat them as per-agent; otherwise per-timestep.
    """
    lines = [ln for ln in text.splitlines() if _LINE_RE.match(ln)]
    parsed = []
    for ln in lines:
        m = _LINE_RE.match(ln)
        coords = [(int(x), int(y)) for x, y in _COORD_RE.findall(m.group(2))]
        parsed.append(coords)  # coords are (x=col, y=row)

    if not parsed:
        return None

    if len(parsed) == num_agents:
        # per-agent lines: parsed[i] = agent i's path of (col,row)
        paths = [[(y, x) for (x, y) in agent] for agent in parsed]
    else:
        # per-timestep lines: parsed[t][i] = agent i's pos at time t
        T = len(parsed)
        paths = [[] for _ in range(num_agents)]
        for t in range(T):
            row = parsed[t]
            for i in range(num_agents):
                x, y = row[i]
                paths[i].append((y, x))  # -> (row,col)
    return pad_paths_to_equal_length(paths)


# --- main entry ------------------------------------------------------------
def solve_with_lacam(grid, starts, goals, lacam_bin: str,
                     time_limit_ms: int = 10000, extra_args=None):
    """Solve one instance with the LaCAM binary. Returns Instance or None.

    lacam_bin: path to the compiled LaCAM executable.
    time_limit_ms: LaCAM anytime budget (longer -> lower SoC).
    extra_args: list of extra CLI flags for your LaCAM build if needed.
    """
    with tempfile.TemporaryDirectory() as td:
        map_path = os.path.join(td, "inst.map")
        scen_path = os.path.join(td, "inst.scen")
        out_path = os.path.join(td, "out.txt")
        write_map_file(grid, map_path)
        write_scen_file(starts, goals, "inst.map", grid, scen_path)

        # CLI mirrors Okumura's reference lacam: -m map -i scen -N agents
        #   -o output -t time(ms). Adjust flags to your build if different.
        cmd = [lacam_bin, "-m", map_path, "-i", scen_path,
               "-N", str(len(starts)), "-o", out_path,
               "-t", str(time_limit_ms)]
        if extra_args:
            cmd += list(extra_args)

        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=time_limit_ms / 1000 + 30)
        except subprocess.TimeoutExpired:
            return None

        # solution may be in the output file or stdout depending on build
        text = ""
        if os.path.exists(out_path):
            with open(out_path) as f:
                text = f.read()
        if not text.strip():
            text = res.stdout

        paths = parse_lacam_solution(text, len(starts))
        if paths is None:
            return None
        # trim trailing all-rest steps and validate
        inst = Instance(grid=grid, starts=list(starts), goals=list(goals),
                        paths=paths)
        try:
            inst.validate()
        except AssertionError:
            return None
        return inst


def generate_with_lacam(num_instances, lacam_bin, map_size=32,
                        obstacle_density=0.2, num_agents=32, seed=0,
                        time_limit_ms=10000):
    """Generate solved instances using POGEMA maps + LaCAM expert.

    Mirrors generate_with_pogema but uses LaCAM instead of PIBT to solve.
    Same return type (list[Instance]) so the rest of the pipeline is unchanged.
    """
    try:
        from pogema import GridConfig
        from pogema.grid import Grid
    except ImportError as e:
        raise ImportError("pip install pogema") from e

    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < num_instances and attempts < num_instances * 4:
        attempts += 1
        s = int(rng.integers(0, 2**31 - 1))
        gc = GridConfig(size=map_size, density=obstacle_density,
                        num_agents=num_agents, seed=s, obs_radius=5)
        g = Grid(gc)
        obstacles = (np.asarray(g.get_obstacles()) != 0).astype(np.int8)
        starts = [tuple(map(int, p)) for p in g.get_agents_xy()]
        goals = [tuple(map(int, p)) for p in g.get_targets_xy()]
        if not all(obstacles[r][c] == 0 for r, c in starts + goals):
            continue
        inst = solve_with_lacam(obstacles, starts, goals, lacam_bin,
                                time_limit_ms=time_limit_ms)
        if inst is not None:
            out.append(inst)
    return out

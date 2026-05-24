"""Low-level grid utilities with NO dependency on Instance or any package
__init__. Both data/features.py and solvers/pibt.py import from here, which
keeps the import graph acyclic.
"""
from __future__ import annotations

from collections import deque
import numpy as np

Cell = tuple[int, int]


def neighbors(grid: np.ndarray, cell: Cell) -> list[Cell]:
    """Free 4-connected neighbors plus the wait-in-place self-loop."""
    H, W = grid.shape
    r, c = cell
    out = [cell]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
            out.append((nr, nc))
    return out


def bfs_distance_field(grid: np.ndarray, goal: Cell) -> np.ndarray:
    """Shortest-path distance from every free cell to `goal` (BFS, 4-connected).
    Obstacle/unreachable cells get a large sentinel value. Doubles as the
    cost-to-goal feature."""
    H, W = grid.shape
    INF = H * W + 1
    dist = np.full((H, W), INF, dtype=np.int32)
    gr, gc = goal
    if grid[gr][gc] != 0:
        return dist
    dist[gr][gc] = 0
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0 \
                    and dist[nr][nc] == INF:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist

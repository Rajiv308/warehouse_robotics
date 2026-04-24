"""
A* path planner for Husky delivery navigation.

Builds a 2D occupancy grid over the workspace from the known shelf positions
in the env config, inflates obstacles by the Husky's body half-width + margin
so the returned waypoints guarantee clearance, and runs A* from the current
base position to the dropoff approach point. The resulting cell path is then
smoothed to a short list of world-frame waypoints that the demo's
diff-drive controller can follow.

This replaces the naive straight-line delivery with real obstacle-aware
routing. It is classical planning (not learned), but it is composable with
the learned modules (nav, BC pickup) that make up the rest of the pipeline.
"""
import heapq
from typing import List, Optional, Tuple

import numpy as np


GRID_CELL = 0.2                       # meters per cell
GRID_X_MIN, GRID_X_MAX = -3.5, 3.5    # planning region x
GRID_Y_MIN, GRID_Y_MAX = -3.0, 3.0    # planning region y
HUSKY_CLEARANCE = 0.55                # Husky half-width + margin (wider so
                                      # pure-pursuit arcs don't graze shelves)

GRID_W = int(round((GRID_X_MAX - GRID_X_MIN) / GRID_CELL))
GRID_H = int(round((GRID_Y_MAX - GRID_Y_MIN) / GRID_CELL))


def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    i = int((x - GRID_X_MIN) / GRID_CELL)
    j = int((y - GRID_Y_MIN) / GRID_CELL)
    i = max(0, min(GRID_W - 1, i))
    j = max(0, min(GRID_H - 1, j))
    return i, j


def grid_to_world(i: int, j: int) -> Tuple[float, float]:
    x = GRID_X_MIN + (i + 0.5) * GRID_CELL
    y = GRID_Y_MIN + (j + 0.5) * GRID_CELL
    return x, y


def build_occupancy_grid(shelf_positions, shelf_half_x=0.6, shelf_half_y=0.3,
                          clearance: float = HUSKY_CLEARANCE) -> np.ndarray:
    """Return a boolean W x H grid where True cells are blocked."""
    grid = np.zeros((GRID_W, GRID_H), dtype=bool)
    for sp in shelf_positions:
        sx, sy = float(sp[0]), float(sp[1])
        for i in range(GRID_W):
            for j in range(GRID_H):
                wx, wy = grid_to_world(i, j)
                if (abs(wx - sx) < shelf_half_x + clearance and
                        abs(wy - sy) < shelf_half_y + clearance):
                    grid[i, j] = True
    return grid


_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def astar(start: Tuple[int, int], goal: Tuple[int, int],
          grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Standard A* on an 8-connected grid. Returns a list of (i, j) cells
    from start to goal inclusive, or None if unreachable."""
    if grid[goal[0], goal[1]]:
        # Goal is inside an obstacle (shouldn't happen with our dropoff
        # ranges, but guard anyway): find nearest free neighbor.
        goal = _nearest_free(goal, grid)
        if goal is None:
            return None

    open_heap: list = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        for di, dj in _NEIGHBORS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < GRID_W and 0 <= nj < GRID_H):
                continue
            if grid[ni, nj]:
                continue
            step = float(np.hypot(di, dj))
            new_g = g_score[cur] + step
            nb = (ni, nj)
            if nb not in g_score or new_g < g_score[nb]:
                g_score[nb] = new_g
                came_from[nb] = cur
                h = float(np.hypot(ni - goal[0], nj - goal[1]))
                heapq.heappush(open_heap, (new_g + h, nb))

    return None


def _nearest_free(cell: Tuple[int, int], grid: np.ndarray,
                   max_radius: int = 8) -> Optional[Tuple[int, int]]:
    ci, cj = cell
    for r in range(1, max_radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if abs(di) != r and abs(dj) != r:
                    continue
                ni, nj = ci + di, cj + dj
                if 0 <= ni < GRID_W and 0 <= nj < GRID_H and not grid[ni, nj]:
                    return (ni, nj)
    return None


def _line_clear(a: Tuple[int, int], b: Tuple[int, int], grid: np.ndarray) -> bool:
    """Check via Bresenham-like supercover traversal that every grid cell the
    line from `a` to `b` passes through is free."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if grid[x, y]:
        return False
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
            if grid[x, y]:
                return False
    else:
        err = dy / 2.0
        while y != y1:
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            if grid[x, y]:
                return False
    return True


def smooth_path(path_cells: List[Tuple[int, int]],
                 grid: np.ndarray) -> List[Tuple[float, float]]:
    """Line-of-sight smoothing. From each kept waypoint, jump to the farthest
    subsequent cell that is in line-of-sight. Typically reduces a long
    staircase path to 2-4 waypoints through the open aisle."""
    if not path_cells:
        return []
    if len(path_cells) == 1:
        return [grid_to_world(*path_cells[0])]

    keep = [path_cells[0]]
    i = 0
    while i < len(path_cells) - 1:
        j = len(path_cells) - 1
        while j > i + 1:
            if _line_clear(path_cells[i], path_cells[j], grid):
                break
            j -= 1
        keep.append(path_cells[j])
        i = j
    return [grid_to_world(*c) for c in keep]


def plan_delivery_path(start_xy: Tuple[float, float],
                        goal_xy: Tuple[float, float],
                        grid: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """Plan a path from start to goal in world coordinates. Returns a list of
    world-frame waypoints or None if unreachable."""
    s = world_to_grid(*start_xy)
    g = world_to_grid(*goal_xy)
    cells = astar(s, g, grid)
    if cells is None:
        return None
    return smooth_path(cells, grid)

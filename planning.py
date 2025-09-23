# picarx_nav/planning.py

import heapq
import numpy as np

# 4-connected motions: (dy, dx, step_cost)
MOTIONS = [
    ( 0,  1, 1.0),  # right
    ( 1,  0, 1.0),  # down (toward robot)
    ( 0, -1, 1.0),  # left
    (-1,  0, 1.0),  # up (forward)
]

def inflate_obstacles(grid, inflation_cells):
    """Binary dilation with a square (Chebyshev) radius of `inflation_cells`."""
    if inflation_cells <= 0:
        return grid
    h, w = grid.shape
    inflated = grid.copy()
    occ = np.argwhere(grid == 1)
    for y, x in occ:
        y0 = max(0, y - inflation_cells)
        y1 = min(h, y + inflation_cells + 1)
        x0 = max(0, x - inflation_cells)
        x1 = min(w, x + inflation_cells + 1)
        inflated[y0:y1, x0:x1] = 1
    return inflated

def manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)

def a_star(grid, start, goal):
    """
    grid: 2D np.array, 0=free, 1=occupied (already inflated)
    start, goal: (gy, gx)
    returns list[(gy,gx)] path inclusive, or None if no path
    """
    h, w = grid.shape
    inb     = lambda y,x: 0 <= y < h and 0 <= x < w
    blocked = lambda y,x: grid[y, x] == 1

    sy, sx = start
    gy, gx = goal
    if blocked(gy, gx):
        return None

    openpq = []
    g_cost = { (sy, sx): 0.0 }
    came   = {}

    f0 = manhattan(sx, sy, gx, gy)
    heapq.heappush(openpq, (f0, 0.0, (sy, sx)))

    visited = set()
    while openpq:
        f, gc, (y, x) = heapq.heappop(openpq)
        if (y, x) == (gy, gx):
            return _reconstruct_path(came, (y, x))

        if (y, x) in visited:
            continue
        visited.add((y, x))

        for dy, dx, step_cost in MOTIONS:
            ny, nx = y + dy, x + dx
            if not inb(ny, nx) or blocked(ny, nx):
                continue
            ng = gc + step_cost
            if ng < g_cost.get((ny, nx), float('inf')):
                g_cost[(ny, nx)] = ng
                came[(ny, nx)]   = (y, x)
                hf = manhattan(nx, ny, gx, gy)
                heapq.heappush(openpq, (ng + hf, ng, (ny, nx)))
    return None

def _reconstruct_path(came, node):
    path = [node]
    while node in came:
        node = came[node]
        path.append(node)
    path.reverse()
    return path

def destination_to_goal_cell(dest_x_cm, dest_y_cm, width, height, res_cm, world_to_grid_fn):
    gx, gy = world_to_grid_fn(dest_x_cm, dest_y_cm, height, width, res_cm)
    return (gy, gx)

def nearest_free_cell(costmap, gy, gx, max_radius=10):
    """
    If requested goal cell is blocked, search outward in Manhattan rings for nearest free cell.
    """
    h, w = costmap.shape
    if 0 <= gy < h and 0 <= gx < w and costmap[gy, gx] == 0:
        return (gy, gx)
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            dx = r - abs(dy)
            for sx, sy in ((gx + dx, gy + dy), (gx - dx, gy + dy)):
                if 0 <= sy < h and 0 <= sx < w and costmap[sy, sx] == 0:
                    return (sy, sx)
    return None

# -*- coding: utf-8 -*-
"""
A* grid planner + executor for Picar-X (single file, mapper excluded).

What this file does:
- Prompts for a destination GRID CELL (gx, gy)
- Scans via your existing `build_map(...)` (function NOT defined here)
- Inflates obstacles forward-biased, gates forward steps with a corridor check
- Adds a tiny turn penalty to reduce zig-zags
- Executes steps with:
    * forward cell moves
    * gentle lane changes (optional)
    * 3-point 90° turn: reverse ±30°, then forward ∓30°
- After EVERY move, prints current pose in cells & centimeters, cells remaining,
  and metric distance-to-go. Uses a cm-threshold "arrived" condition.

Intentionally NOT included here (expected to exist elsewhere in your project):
- build_map(px, width, height, res_cm)
- polar_to_xy(...)
- world_to_grid(...)

Author: you + ChatGPT
"""

import time
import math
import heapq
import numpy as np
import cv2

from picarx import Picarx  # SunFounder Picar-X API

# =========================
# Tunables / constants
# =========================
WIDTH        = 20        # grid columns
HEIGHT       = 20        # grid rows
RES_CM       = 5.0       # centimeters per grid cell
MAX_RANGE_CM = 100.0     # (used by your mapper; here for reference)
PAN_SETTLE_SEC = 0.10    # (used by your mapper; here for reference)

# Costmap inflation (forward-biased)
INFLATE_SIDE_CELLS = 1   # grow obstacles sideways (cells)
INFLATE_FWD_CELLS  = 3   # grow obstacles forward (cells)
INFLATE_BACK_CELLS = 1   # grow obstacles backward (cells)

# A* neighbor rules
LOOKAHEAD_ROWS = 3       # corridor depth required to allow a 'forward' expansion
SIDE_PAD_COLS  = 1       # half-width of corridor (columns) for that check
TURN_PENALTY   = 0.10    # tiny cost for changing direction (reduces zig-zag)

# Driving / steering timing — CALIBRATE on your floor
DRIVE_POWER     = 40         # px.forward()/backward() power (0..100)
TIME_PER_CM     = 0.04       # seconds to drive 1 cm at DRIVE_POWER when steering ~0°
STEER_SETTLE_S  = 0.08       # seconds to let steering servo settle after set
CELL_RUNUP_CM   = RES_CM     # forward distance for one "cell step"

# Lane-change S-curve (kept for gentle lateral shifts)
LANE_STEER_DEG  = 22         # ±deg for S-curve
LANE_SPLIT_1    = 0.55       # time fraction of first arc
LANE_SPLIT_2    = 0.45       # time fraction of second arc

# 3-point 90° maneuver (your request)
THREE_PT_STEER_DEG = 30      # ±30° while reversing/forwarding
THREE_PT_TIME_S    = 0.60    # seconds for each leg; tune to achieve ~90° heading change

# “Arrived” when within this many cm of the destination (set ~0.8 cell)
ARRIVE_CM = RES_CM * 0.8

# =========================
# Costmap inflation (forward-biased)
# =========================
def inflate_obstacles(grid, side_cells=INFLATE_SIDE_CELLS, fwd_cells=INFLATE_FWD_CELLS, back_cells=INFLATE_BACK_CELLS):
    """
    Forward-biased morphological dilation. Obstacles cast a 'shadow' ahead
    to discourage late forward moves near them.
    """
    k_h = fwd_cells + 1 + back_cells
    k_w = 2 * side_cells + 1
    kernel = np.ones((k_h, k_w), np.uint8)
    return cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)

# =========================
# A* on 4-connected grid (with forward gating + turn penalty)
# =========================
# Motions: (dy, dx, step_cost); note: "forward" in grid is (dy=-1, dx=0)
MOTIONS = [
    ( 0,  1, 1.0),  # right (E)
    ( 1,  0, 1.0),  # down  (toward robot, S)
    ( 0, -1, 1.0),  # left  (W)
    (-1,  0, 1.0),  # up    (forward, N)
]

def manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)

def forward_clear(grid, x, y):
    """
    Allow the forward neighbor (dy=-1, dx=0) only if a corridor ahead is free for
    LOOKAHEAD_ROWS and ±SIDE_PAD_COLS columns.
    """
    H, W = grid.shape
    y0 = max(0, y - LOOKAHEAD_ROWS)
    x0 = max(0, x - SIDE_PAD_COLS)
    x1 = min(W - 1, x + SIDE_PAD_COLS)
    return grid[y0:y, x0:x1+1].sum() == 0

def a_star(grid, start, goal):
    """
    grid: 2D np.array, 0=free, 1=occupied (already inflated)
    start, goal: (gy, gx) row/col indices
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
    last_mv = { (sy, sx): None }  # store last (dy,dx) used to reach a node

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

            # Gate forward motion with corridor check
            if dy == -1 and dx == 0 and not forward_clear(grid, x, y):
                continue

            # Add tiny penalty on direction change
            prev = last_mv[(y, x)]
            penalty = TURN_PENALTY if (prev is not None and prev != (dy, dx)) else 0.0
            ng = gc + step_cost + penalty

            if ng < g_cost.get((ny, nx), float('inf')):
                g_cost[(ny, nx)] = ng
                came[(ny, nx)]   = (y, x)
                last_mv[(ny, nx)] = (dy, dx)
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

# =========================
# Helpers: pose/units
# =========================
def cell_to_world_cm(cell_y, cell_x, start_y, start_x, res_cm):
    """
    Convert current cell (y,x) displacement from start cell into (x_cm, y_cm)
    in a robot-start-centered frame: x right (+), y forward (+).
    """
    dx_cells = (cell_x - start_x)           # +right
    dy_cells = (start_y - cell_y)           # moving "up" (toward smaller y index) is +forward
    x_cm = dx_cells * res_cm
    y_cm = dy_cells * res_cm
    return x_cm, y_cm

# =========================
# Execution primitives
# =========================
def _wait_set_steer(px, angle_deg):
    px.set_dir_servo_angle(angle_deg)
    time.sleep(STEER_SETTLE_S)

def drive_forward_cm(px, cm):
    _wait_set_steer(px, 0)
    px.forward(DRIVE_POWER)
    time.sleep(cm * TIME_PER_CM)
    px.stop()

def drive_forward_cell(px):
    drive_forward_cm(px, CELL_RUNUP_CM)

def lane_change(px, right=True):
    """
    Gentle lateral shift that ends straight (optional; kept for completeness).
    """
    a = +LANE_STEER_DEG if right else -LANE_STEER_DEG
    t1 = CELL_RUNUP_CM * TIME_PER_CM * LANE_SPLIT_1
    t2 = CELL_RUNUP_CM * TIME_PER_CM * LANE_SPLIT_2

    _wait_set_steer(px, a)
    px.forward(DRIVE_POWER); time.sleep(t1)
    _wait_set_steer(px, -a)
    px.forward(DRIVE_POWER); time.sleep(t2)
    px.stop()
    _wait_set_steer(px, 0)

def three_point_turn_90(px, left=True):
    """
    90° turn as a 3-point maneuver:
      1) reverse with steering a1 = ∓30°
      2) forward with steering a2 = ±30°
    Tune THREE_PT_TIME_S for your floor.
    """
    a1 = -THREE_PT_STEER_DEG if left else +THREE_PT_STEER_DEG
    a2 = +THREE_PT_STEER_DEG if left else -THREE_PT_STEER_DEG

    # Leg 1: reverse with steering a1
    _wait_set_steer(px, a1)
    px.backward(DRIVE_POWER)
    time.sleep(THREE_PT_TIME_S)
    px.stop()

    # Leg 2: forward with steering a2
    _wait_set_steer(px, a2)
    px.forward(DRIVE_POWER)
    time.sleep(THREE_PT_TIME_S)
    px.stop()

    _wait_set_steer(px, 0)

def execute_next_step(px, path, start_cell, last_step_vec):
    """
    Execute one grid step and return new last_step_vec.
    - Forward (dy=-1,dx=0): drive_forward_cell()
    - Lateral (dx!=0) or backward (dy=+1): if 90° from last move, do 3-point turn;
      else do gentle lane change.
    """
    if path is None or len(path) < 2:
        return last_step_vec

    (y0, x0) = start_cell
    (y1, x1) = path[1]
    dy, dx = (y1 - y0), (x1 - x0)

    # Forward step
    if dy == -1 and dx == 0:
        drive_forward_cell(px)
        return (-1, 0)

    # Backward step (rare in planner, but supported)
    if dy == 1 and dx == 0:
        _wait_set_steer(px, 0)
        px.backward(DRIVE_POWER)
        time.sleep(CELL_RUNUP_CM * TIME_PER_CM)
        px.stop()
        return (1, 0)

    # Lateral step (E/W)
    is_right = (dx == +1)
    is_left  = (dx == -1)

    def is_perpendicular(v1, v2):
        if v1 is None or v2 is None:
            return True
        (py, px_) = v1
        (qy, qx_) = v2
        return (px_ * qx_ + py * qy) == 0  # dot product == 0

    if is_perpendicular(last_step_vec, (dy, dx)):
        # Your requested 3-point 90° turn
        three_point_turn_90(px, left=is_left)
        # Roll forward one cell to occupy target column cleanly
        drive_forward_cell(px)
    else:
        lane_change(px, right=is_right)

    return (dy, dx)

# =========================
# Prompt utilities
# =========================
def prompt_destination_cell():
    """
    Ask the user for a destination GRID CELL, not centimeters, to avoid relying on world_to_grid.
    Returns (gy, gx) as (row, col).
    """
    print(f"Enter destination grid cell indices (gy, gx). Grid size is HEIGHT={HEIGHT} (rows), WIDTH={WIDTH} (cols).")
    print("  gy: 0 is top row, increases downward; gx: 0 is left column, increases rightward.")
    while True:
        try:
            gy = int(input("Destination gy (row, 0..HEIGHT-1): ").strip())
            gx = int(input("Destination gx (col, 0..WIDTH-1): ").strip())
            if 0 <= gy < HEIGHT and 0 <= gx < WIDTH:
                return (gy, gx)
            else:
                print("Out of bounds. Please enter values within the grid size.\n")
        except ValueError:
            print("Please enter integer values.\n")

# =========================
# Main loop
# =========================
def main():
    # Ask for destination cell (gy, gx)
    requested_goal = prompt_destination_cell()

    # Init car
    px = Picarx()
    np.set_printoptions(linewidth=140, threshold=np.inf)

    # Robot starts at bottom-center, facing "up" the grid
    start_cell_origin = (HEIGHT - 1, WIDTH // 2)
    start_cell        = tuple(start_cell_origin)
    last_step_vec     = None

    try:
        while True:
            # Ensure car is stationary during scan
            _wait_set_steer(px, 0)
            px.stop()
            time.sleep(0.05)

            # 1) Sense: build occupancy (expects your build_map elsewhere)
            # NOTE: This function is intentionally NOT included here per your request.
            occ = build_map(px, width=WIDTH, height=HEIGHT, res_cm=RES_CM)

            # 2) Inflate obstacles (forward-biased)
            costmap = inflate_obstacles(occ, INFLATE_SIDE_CELLS, INFLATE_FWD_CELLS, INFLATE_BACK_CELLS)

            # 3) Snap requested goal to nearest free cell
            gy_req, gx_req = requested_goal
            goal = (gy_req, gx_req)
            if costmap[gy_req, gx_req] == 1:
                # Search outward in Manhattan rings
                goal = None
                h, w = costmap.shape
                for r in range(1, 11):
                    found = False
                    for dy in range(-r, r + 1):
                        dx = r - abs(dy)
                        for sx, sy in ((gx_req + dx, gy_req + dy), (gx_req - dx, gy_req + dy)):
                            if 0 <= sy < h and 0 <= sx < w and costmap[sy, sx] == 0:
                                goal = (sy, sx); found = True; break
                        if found: break
                    if found: break
                if goal is None:
                    print("No reachable free goal cell vicinity; rescanning...")
                    time.sleep(0.3)
                    continue

            # 4) Plan with A*
            path = a_star(costmap, start_cell, goal)
            if path is None or len(path) < 2:
                px.stop()
                print(f"No path from {start_cell} to {goal}; rescanning...")
                time.sleep(0.3)
                continue

            # --- Status / progress print BEFORE moving ---
            cells_remaining = len(path) - 1
            # Compute pose in cm relative to start origin for visibility
            x_cm, y_cm = cell_to_world_cm(
                cell_y=start_cell[0],
                cell_x=start_cell[1],
                start_y=start_cell_origin[0],
                start_x=start_cell_origin[1],
                res_cm=RES_CM
            )
            # Distance to destination in cells→cm
            gy_goal, gx_goal = goal
            cell_dist = math.hypot(gx_goal - start_cell[1], gy_goal - start_cell[0])
            dist_to_dest = cell_dist * RES_CM

            print("\n--- STATUS (pre-move) ---")
            print(f"Current cell: {start_cell} | World approx: x={x_cm:.1f} cm, y={y_cm:.1f} cm")
            print(f"Goal cell:    {goal}")
            print(f"Cells remaining: {cells_remaining}")
            print(f"Distance to destination: {dist_to_dest:.1f} cm")

            # 5) Arrived check (metric)
            if dist_to_dest <= ARRIVE_CM or cells_remaining <= 1:
                px.stop()
                print(f"Arrived: within {ARRIVE_CM:.1f} cm (or one step) of destination.")
                break

            # 6) Execute exactly one step
            last_step_vec = execute_next_step(px, path, start_cell, last_step_vec)

            # 7) Advance internal pose by consuming the next path cell
            start_cell = path[1]

            # --- Status / progress print AFTER moving ---
            x_cm, y_cm = cell_to_world_cm(
                cell_y=start_cell[0],
                cell_x=start_cell[1],
                start_y=start_cell_origin[0],
                start_x=start_cell_origin[1],
                res_cm=RES_CM
            )
            gy_goal, gx_goal = goal
            cell_dist = math.hypot(gx_goal - start_cell[1], gy_goal - start_cell[0])
            dist_to_dest = cell_dist * RES_CM

            print("\n--- STATUS (post-move) ---")
            print(f"Current cell: {start_cell} | World approx: x={x_cm:.1f} cm, y={y_cm:.1f} cm")
            print(f"Cells remaining: {max(0, cells_remaining-1)}")
            print(f"Distance to destination: {dist_to_dest:.1f} cm")
            print("-----------------------------")

            # Loop: scan → plan → move → print → repeat

    finally:
        try:
            px.stop()
            _wait_set_steer(px, 0)
        except Exception:
            pass

if __name__ == "__main__":
    main()

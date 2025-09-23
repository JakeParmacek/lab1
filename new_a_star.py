
import time
import math
import heapq
import numpy as np
import cv2

from picarx import Picarx

# =========================
# Global constants (TUNE ME)
# =========================
WIDTH        = 20          # grid columns
HEIGHT       = 20          # grid rows
RES_CM       = 5.0         # cm per grid cell
MAX_RANGE_CM = 100.0       # ultrasonic cutoff
PAN_SETTLE_SEC = 0.10

INFLATE_SIDE_CELLS = 1     # grow obstacles sideways (cells)
INFLATE_FWD_CELLS  = 3     # grow obstacles forward (+y in world, -row in grid here)
INFLATE_BACK_CELLS = 1

LOOKAHEAD_ROWS = 3         # how many rows ahead must be clear to allow forward step
SIDE_PAD_COLS  = 1         # corridor half-width for lookahead

TURN_PENALTY   = 0.10      # tiny penalty on direction change in A*

# Driving / steering (TUNE ME)
DRIVE_POWER     = 40       # px.forward() power (0..100)
STEER_SETTLE_S  = 0.08     # wait after changing steering angle
TIME_PER_CM     = 0.04     # seconds to drive 1cm at DRIVE_POWER when angle ≈ 0°
CELL_RUNUP_CM   = RES_CM   # how far to drive for a "forward cell" action

# Lane change S-curve (still used for gentle lateral shifts, optional)
LANE_STEER_DEG  = 22       # ±deg for S-curve segments
LANE_SPLIT_1    = 0.55     # first arc fraction of cell distance time
LANE_SPLIT_2    = 0.45     # second arc fraction

# 3-point 90° turn (your requested behavior)
THREE_PT_STEER_DEG = 30    # ±30° steering while reversing/forwarding
THREE_PT_TIME_S    = 0.60  # seconds for each leg (back then forward). Tune on floor.

# Discrete steering labels (not strictly needed now but kept for clarity)
LEFT_DEG     = -LANE_STEER_DEG
RIGHT_DEG    = +LANE_STEER_DEG
STRAIGHT_DEG = 0





def inflate_obstacles(grid, side_cells=INFLATE_SIDE_CELLS, fwd_cells=INFLATE_FWD_CELLS, back_cells=INFLATE_BACK_CELLS):
    """
    Forward-biased dilation. We treat obstacles as "casting a shadow" in the forward
    direction (robot-forward = -row in our grid when moving toward smaller y).

    Implementation detail: we build a rectangular structuring element sized to the
    requested span and apply a dilation. Simpler and faster than manual loops.
    """
    # Rectangular kernel that spans sideways and fore/back
    k_h = fwd_cells + 1 + back_cells
    k_w = 2 * side_cells + 1
    kernel = np.ones((k_h, k_w), np.uint8)

    # Apply dilation
    inflated = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
    return inflated

# =========================
# A* on 4-connected grid (with forward gating + turn penalty)
# =========================
# 4-connected motions: (dy, dx, step_cost)
MOTIONS = [
    ( 0,  1, 1.0),  # right
    ( 1,  0, 1.0),  # down (toward robot)
    ( 0, -1, 1.0),  # left
    (-1,  0, 1.0),  # up (forward)
]

def manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)

def forward_clear(grid, x, y):
    """
    Allow the forward neighbor (dy=-1, dx=0) only if a small corridor
    ahead is free for LOOKAHEAD_ROWS and +/- SIDE_PAD_COLS columns.
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
    last_mv = { (sy, sx): None }  # track last motion (dy,dx)

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

            # Gate the "forward" step (dy=-1, dx=0)
            if dy == -1 and dx == 0 and not forward_clear(grid, x, y):
                continue

            # Add small penalty for turning (prefers straighter runs)
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
# Goal utilities
# =========================
def destination_to_goal_cell(dest_x_cm, dest_y_cm, width, height, res_cm):
    gx, gy = world_to_grid(dest_x_cm, dest_y_cm, height, width, res_cm)
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

# =========================
# Helpers: pose/units
# =========================
def cell_to_world_cm(cell_y, cell_x, start_y, start_x, res_cm):
    """
    Convert current cell (y,x) displacement from start cell into (x_cm, y_cm)
    in the robot's start-centered frame: x right (+), y forward (+).
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
    Gentle lateral shift that ends straight (optional; still useful for small E/W moves).
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
    Your requested 90° turn: reverse with +/−30°, then forward with the opposite +/−30°.
    This does NOT truly rotate in place; it approximates a tight heading change with
    minimal lateral drift. Tune THREE_PT_TIME_S on your floor.
    """
    a1 = -THREE_PT_STEER_DEG if left else +THREE_PT_STEER_DEG  # first leg steering while reversing
    a2 = +THREE_PT_STEER_DEG if left else -THREE_PT_STEER_DEG  # second leg steering while moving forward

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
    Execute one grid step. We look at the next cell relative to current cell and:
      - If moving straight 'forward' (dy=-1, dx=0): drive_forward_cell()
      - If moving lateral (dx != 0) or backward (dy=+1): 
            if it's a 90° change from last_step_vec, do a 3-point 90° turn
            otherwise, perform a gentle lane_change (ends straight)
    Returns the new last_step_vec used next iteration.
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

    # Backward step (rare in planner; but handle if occurs)
    if dy == 1 and dx == 0:
        # Use a 3-point to "about-face-ish" then move, or just reverse one cell:
        # Here we keep it simple: reverse straight one cell.
        _wait_set_steer(px, 0)
        px.backward(DRIVE_POWER)
        time.sleep(CELL_RUNUP_CM * TIME_PER_CM)
        px.stop()
        return (1, 0)

    # Lateral step (E/W). Decide between 3-point 90° vs gentle lane change.
    is_right = (dx == +1)
    is_left  = (dx == -1)

    # Detect 90° change from last motion: if last was forward/back and now lateral, or vice versa.
    def is_perpendicular(v1, v2):
        if v1 is None or v2 is None:
            return True
        (py, px_) = v1
        (qy, qx_) = v2
        return (px_ * qx_ + py * qy) == 0  # dot product == 0

    if is_perpendicular(last_step_vec, (dy, dx)):
        # Do your requested 3-point 90° turn: choose left/right
        three_point_turn_90(px, left=is_left)
        # After the maneuver, roll a short straight to "occupy" the next cell
        drive_forward_cell(px)
    else:
        # Small lateral without hard turn
        lane_change(px, right=is_right)

    return (dy, dx)

# =========================
# Simple prompt
# =========================
def prompt_destination_cm():
    print("Enter destination in ROBOT frame (centimeters).")
    print("  x: right is positive; left is negative")
    print("  y: forward is positive; behind is negative (keep positive for this demo)")
    while True:
        try:
            x = float(input("Destination X (cm): ").strip())
            y = float(input("Destination Y (cm): ").strip())
            return x, y
        except ValueError:
            print("Please enter numeric values (e.g., 0 or 80 or -20). Try again.\n")

# =========================
# Main
# =========================
def main():
    # Prompt for destination
    dest_x_cm, dest_y_cm = prompt_destination_cm()

    # Init car
    px = Picarx()
    np.set_printoptions(linewidth=140, threshold=np.inf)

    # Assume robot starts at bottom-center, facing "up" the grid
    start_cell_origin = (HEIGHT - 1, WIDTH // 2)
    start_cell        = tuple(start_cell_origin)

    # Convert requested destination to goal cell once
    requested_goal = destination_to_goal_cell(dest_x_cm, dest_y_cm, WIDTH, HEIGHT, RES_CM)

    last_step_vec = None  # track last (dy,dx) we executed

    try:
        while True:
            # --- Ensure car is stationary during scan ---
            _wait_set_steer(px, STRAIGHT_DEG)
            px.stop()
            time.sleep(0.05)

            # 1) Sense: build occupancy (stationary)
            occ = build_map(px, width=WIDTH, height=HEIGHT, res_cm=RES_CM)

            print("\n--- Occupancy grid (0=free, 1=occupied) ---")
            print(occ)
            print("-------------------------------------------")

            # 2) Inflate obstacles (forward-biased)
            costmap = inflate_obstacles(occ, INFLATE_SIDE_CELLS, INFLATE_FWD_CELLS, INFLATE_BACK_CELLS)

            print("\n--- Inflated grid -------------------------")
            print(costmap)
            print("-------------------------------------------")

            # 3) Position estimates (fake odom from cells)
            cur_x_cm, cur_y_cm = cell_to_world_cm(
                cell_y=start_cell[0],
                cell_x=start_cell[1],
                start_y=start_cell_origin[0],
                start_x=start_cell_origin[1],
                res_cm=RES_CM
            )
            dx = dest_x_cm - cur_x_cm
            dy = dest_y_cm - cur_y_cm
            dist_to_dest = math.hypot(dx, dy)
            print(f"Position from origin (cm): x={cur_x_cm:.1f}, y={cur_y_cm:.1f}")
            print(f"Distance to destination (cm): {dist_to_dest:.1f}")
            print("-------------------------------------------")

            # 4) Snap requested goal to nearest free cell
            gy_req, gx_req = requested_goal
            goal = nearest_free_cell(costmap, gy_req, gx_req, max_radius=10)
            if goal is None:
                px.stop()
                print("No reachable goal vicinity; rescanning...")
                time.sleep(0.3)
                continue

            # 5) Plan with A*
            path = a_star(costmap, start_cell, goal)
            if path is None or len(path) < 2:
                px.stop()
                print(f"No path from {start_cell} to {goal}; rescanning...")
                time.sleep(0.3)
                continue

            # Arrived check: if we're already at goal cell (or one step away), stop.
            if (start_cell == goal) or (len(path) <= 2):
                px.stop()
                print("Arrived at destination vicinity.")
                break

            # 6) Execute exactly one step (uses 3-point turn on 90° changes)
            last_step_vec = execute_next_step(px, path, start_cell, last_step_vec)

            # 7) Advance our internal "pose" by consuming the next path cell (fake odometry)
            start_cell = path[1]

            # Loop back to scan again (stationary) and replan

    finally:
        try:
            px.stop()
            _wait_set_steer(px, 0)
        except Exception:
            pass

if __name__ == "__main__":
    main()

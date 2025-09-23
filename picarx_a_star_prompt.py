# picarx_a_star_prompt_stationary.py
# Stationary-scan loop for PiCar-X:
# 1) stop, scan (ultrasonic sweep) -> occupancy grid
# 2) print grid
# 3) inflate obstacles + A* (4-connected)
# 4) steer LEFT/STRAIGHT/RIGHT and drive forward briefly
# 5) stop, update "pose", repeat

import time
import math
import heapq
import numpy as np
from picarx import Picarx

# =========================
# Tunables (safe defaults)
# =========================
WIDTH, HEIGHT   = 25, 25      # grid size (cells). 25x25 @ 5 cm ? 1.25 m square
RES_CM          = 5           # centimeters per grid cell
INFLATE_CELLS   = 1           # safety buffer (cells)
DRIVE_POWER     = 20          # forward power (0..100)
LEFT_DEG        = -30
RIGHT_DEG       = 30
STRAIGHT_DEG    = 0
STEP_DRIVE_SEC  = 0.3        # drive time before "consuming" next path cell
PAN_SETTLE_SEC  = 0.08        # servo settle time per pan step
MAX_RANGE_CM    = 100         # ignore readings beyond this

# =========================
# Mapping (ultrasonic sweep)
# =========================
def build_map(px, width=WIDTH, height=HEIGHT, res_cm=RES_CM):
    """
    Sweep camera pan from -45..+43 deg in 2 deg steps.
    Mark 1s in grid where the ultrasonic detects an obstacle.
    Robot-centric coordinates (cm): +y forward, +x right; robot sits near (center, bottom).
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    for ang in range(-45, 45, 2):
        px.set_cam_pan_angle(ang)
        time.sleep(PAN_SETTLE_SEC)  # let servo settle
        d = px.ultrasonic.read()  # distance in cm
        if (d is None) or (d <= 0) or (d > MAX_RANGE_CM):
            continue
        x_cm, y_cm = polar_to_xy(ang, d)
        gx, gy = world_to_grid(x_cm, y_cm, height, width, res_cm)
        grid[gy, gx] = 1
    return grid

def polar_to_xy(angle_deg, distance_cm):
    """
    Convert polar (angle from forward axis, distance cm) to robot-centered (x,y) in cm.
    y is forward, x is right. +10cm bias nudges very close hits ahead of the origin.
    """
    theta = math.radians(angle_deg)
    x = distance_cm * math.sin(theta)
    y = distance_cm * math.cos(theta) + 10.0
    return x, y

def world_to_grid(x_cm, y_cm, height, width, res_cm):
    """
    Map robot-centric (x_cm, y_cm) to grid indices (gx, gy).
    Robot is at bottom-center; y forward increases gy upward.
    """
    gx = int(round(x_cm / res_cm)) + width // 2
    gy = int(round(y_cm / res_cm))
    gx = max(0, min(width - 1, gx))
    gy = max(0, min(height - 1, gy))
    return gx, gy
    
    
# =========================
# Inflation (safety buffer)
# =========================
def inflate_obstacles(grid, inflation_cells=INFLATE_CELLS):
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

# =========================
# A* (4-connected only)
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
# Discrete steering
# =========================
def next_steer_from_path(path, current_cell):
    """
    Look at the immediate next cell; compare x to decide LEFT/RIGHT/STRAIGHT.
    """
    if path is None or len(path) < 2:
        return STRAIGHT_DEG
    y0, x0 = current_cell
    y1, x1 = path[1]
    if x1 > x0:
        return RIGHT_DEG
    elif x1 < x0:
        return LEFT_DEG
    else:
        return STRAIGHT_DEG

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
# Main
# =========================
def main():
    # Prompt for destination
    dest_x_cm, dest_y_cm = prompt_destination_cm()

    # Init car
    px = Picarx()
    np.set_printoptions(linewidth=140, threshold=np.inf)

    try:
        width, height, res_cm = WIDTH, HEIGHT, RES_CM
        inflate = INFLATE_CELLS

        # Assume robot starts at bottom-center, facing "up" the grid
        start_cell = (height - 1, width // 2)
        start_cell_0 = (height - 1, width // 2)   # origin cell
        start_cell   = tuple(start_cell_0) # current "pose" cell

        # Convert requested destination to goal cell once
        requested_goal = destination_to_goal_cell(dest_x_cm, dest_y_cm, width, height, res_cm)

        while True:
            # --- Ensure car is stationary during scan ---
            px.set_dir_servo_angle(STRAIGHT_DEG)
            px.stop()
            time.sleep(0.05)

            # 1) Sense: build occupancy (stationary)
            occ = build_map(px, width=width, height=height, res_cm=res_cm)

            # 2) Print the 2D occupancy grid after every scan
            print("\n--- Occupancy grid (0=free, 1=occupied) ---")
            print(occ)
            print("-------------------------------------------")

            # 3) Inflate obstacles
            costmap = inflate_obstacles(occ, inflate)
            
            print("\n--- Inflated grid -------------------------")
            print(costmap)
            print("---------------------------------------------")
            
            cur_x_cm, cur_y_cm = cell_to_world_cm(
                cell_y=start_cell[0],
                cell_x=start_cell[1],
                start_y=start_cell_0[0],
                start_x=start_cell_0[1],
                res_cm=res_cm
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

            # 6) Compute discrete steering from next step
            steer_deg = next_steer_from_path(path, start_cell)
            px.set_dir_servo_angle(steer_deg)

            # 7) Drive forward a short step, then stop again for the next scan
            px.forward(DRIVE_POWER)
            time.sleep(STEP_DRIVE_SEC)
            px.stop()

            # 8) Advance our internal "pose" by consuming the next path cell (fake odometry)
            start_cell = path[1]

            # Loop back to scan again (stationary) and replan

    finally:
        try:
            px.stop()
        except:
            pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import math
import sys
import time
import heapq
import numpy as np
from picarx import Picarx

# ----------------- Config (tune on robot) -----------------
CELL_CM = 20
GRID_X_MIN, GRID_X_MAX = -5, 5    # x in [-5..5]
GRID_Y_MIN, GRID_Y_MAX = 0, 10    # y in [0..10]
FREE, OCC = 0, 1                  # unknown treated as FREE

SCAN_MIN_DEG = -45
SCAN_MAX_DEG = 45
SCAN_STEP_DEG = 2
MAX_RANGE_CM = 100

SPEED_FORWARD = 30
SPEED_BACKWARD = 25
T_FORWARD_CELL_SEC = 0.80
T_BACKWARD_CELL_SEC = 0.90
STEER_MAX_DEG = 30
TURN_LEFT_SEQ = (0.45, 0.40, 0.35)
TURN_RIGHT_SEQ = (0.45, 0.40, 0.35)

# ----------------- Grid -----------------
class Grid:
    def __init__(self):
        self.w = GRID_X_MAX - GRID_X_MIN + 1
        self.h = GRID_Y_MAX - GRID_Y_MIN + 1
        self.grid = np.full((self.h, self.w), FREE, dtype=np.int8)  # row=y, col=x

    def world_to_idx(self, x, y):
        return (y - GRID_Y_MIN, x - GRID_X_MIN)

    def in_bounds_xy(self, x, y):
        return GRID_X_MIN <= x <= GRID_X_MAX and GRID_Y_MIN <= y <= GRID_Y_MAX

    def mark_occ(self, x, y):
        if not self.in_bounds_xy(x, y): return
        r, c = self.world_to_idx(x, y)
        self.grid[r, c] = OCC

    def mark_free(self, x, y):
        if not self.in_bounds_xy(x, y): return
        r, c = self.world_to_idx(x, y)
        if self.grid[r, c] != OCC:
            self.grid[r, c] = FREE

# ----------------- A* (4-connected, unknown==free) -----------------
def astar(grid: Grid, start, goal):
    sx, sy = start
    gx, gy = goal
    if not (grid.in_bounds_xy(gx, gy) and grid.in_bounds_xy(sx, sy)):
        return []

    def is_free(x, y):
        r, c = grid.world_to_idx(x, y)
        return grid.grid[r, c] != OCC

    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    openq = []
    heapq.heappush(openq, (h(start, goal), 0, start))
    came = {}
    gscore = {start: 0}

    def neigh(n):
        x, y = n
        for dx, dy in ((0,1),(1,0),(0,-1),(-1,0)):
            nx, ny = x+dx, y+dy
            if grid.in_bounds_xy(nx, ny) and (is_free(nx, ny) or (nx, ny)==goal):
                yield (nx, ny)

    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return list(reversed(path))
        for nb in neigh(cur):
            ng = g + 1
            if ng < gscore.get(nb, 1e9):
                came[nb] = cur
                gscore[nb] = ng
                heapq.heappush(openq, (ng + h(nb, goal), ng, nb))
    return []

# ----------------- Scan and integrate -----------------
def sweep_and_update(px: Picarx, grid: Grid):
    pts = []
    for ang in range(SCAN_MIN_DEG, SCAN_MAX_DEG + 1, SCAN_STEP_DEG):
        px.set_cam_pan_angle(ang)  # use camera/ultrasonic pan servo
        time.sleep(0.01)
        d = px.get_distance()
        if d is None:
            d = MAX_RANGE_CM
        d = max(0, min(MAX_RANGE_CM, d))
        pts.append((ang, d))
    px.set_cam_pan_angle(0)

    # Project to grid: raycast, mark FREE along ray, OCC at hit (ignore y<=0 to avoid self-mark)
    step = CELL_CM / 4.0
    for ang_deg, rng in pts:
        if rng <= 0: continue
        rad = math.radians(ang_deg)
        max_s = min(rng, MAX_RANGE_CM)
        s = step
        while s <= max_s + 1e-6:
            x_cm = math.sin(rad) * s
            y_cm = math.cos(rad) * s
            x = int(round(x_cm / CELL_CM))
            y = int(round(y_cm / CELL_CM))
            if y <= 0:  # don't mark ego row/behind
                s += step
                continue
            if not grid.in_bounds_xy(x, y):
                break
            if rng < MAX_RANGE_CM and abs(s - rng) <= step*1.2:
                grid.mark_occ(x, y)
                break
            else:
                grid.mark_free(x, y)
            s += step

# ----------------- Minimal motion: 1-cell forward or 3-pt turn + forward -----------------
def forward_one(px: Picarx):
    px.set_dir_servo_angle(0)
    px.forward(SPEED_FORWARD)
    time.sleep(T_FORWARD_CELL_SEC)
    px.stop()

def backward_one(px: Picarx):
    px.set_dir_servo_angle(0)
    px.backward(SPEED_BACKWARD)
    time.sleep(T_BACKWARD_CELL_SEC)
    px.stop()

def three_point_left(px: Picarx):
    t1,t2,t3 = TURN_LEFT_SEQ
    px.set_dir_servo_angle(+STEER_MAX_DEG); px.forward(SPEED_FORWARD); time.sleep(t1); px.stop()
    px.set_dir_servo_angle(-STEER_MAX_DEG); px.backward(SPEED_BACKWARD); time.sleep(t2); px.stop()
    px.set_dir_servo_angle(+STEER_MAX_DEG); px.forward(SPEED_FORWARD); time.sleep(t3); px.stop()
    px.set_dir_servo_angle(0)

def three_point_right(px: Picarx):
    t1,t2,t3 = TURN_RIGHT_SEQ
    px.set_dir_servo_angle(-STEER_MAX_DEG); px.forward(SPEED_FORWARD); time.sleep(t1); px.stop()
    px.set_dir_servo_angle(+STEER_MAX_DEG); px.backward(SPEED_BACKWARD); time.sleep(t2); px.stop()
    px.set_dir_servo_angle(-STEER_MAX_DEG); px.forward(SPEED_FORWARD); time.sleep(t3); px.stop()
    px.set_dir_servo_angle(0)

# Ego heading is +y; decide primitive for next step (dx,dy)
def execute_step(px: Picarx, cur, nxt):
    dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
    if (dx, dy) == (0, 1):   # forward
        forward_one(px); return "FWD"
    if (dx, dy) == (-1, 0):  # left turn + forward
        three_point_left(px); forward_one(px); return "L+F"
    if (dx, dy) == (1, 0):   # right turn + forward
        three_point_right(px); forward_one(px); return "R+F"
    if (dx, dy) == (0, -1):  # backward 1
        backward_one(px); return "BACK"
    return "NONE"

# ----------------- Main loop (barebones) -----------------
def main():
    px = Picarx()

    g = Grid()

    # Ask once for destination
    print("Enter destination x y (x in [-5..5], y in [0..10]) relative to start (0,0):")
    while True:
        parts = sys.stdin.readline().strip().split()
        if len(parts) == 2:
            try:
                goal = (int(parts[0]), int(parts[1]))
                if g.in_bounds_xy(*goal):
                    break
            except:
                pass
        print("Invalid. Try again (e.g., 2 5):")

    cur = (0, 0)

    while True:
        # Plan
        path = astar(g, cur, goal)
        print("Path:", path)

        if not path:
            print("No path. Scanning to update map...")
            sweep_and_update(px, g)
            continue

        if len(path) == 1 and path[0] == goal and cur == goal:
            print("Reached destination.")
            px.stop()
            break

        if len(path) < 2:
            sweep_and_update(px, g)
            continue

        # Before moving, sweep and check if any cell on path up to a small horizon is now occupied
        sweep_and_update(px, g)
        horizon = min(5, len(path)-1)
        blocked = False
        for i in range(1, horizon+1):
            x, y = path[i]
            r, c = g.world_to_idx(x, y)
            if g.grid[r, c] == OCC:
                blocked = True
                break
        if blocked:
            print("Path blocked. Replanning...")
            continue

        # Execute one step
        nxt = path[1]
        tag = execute_step(px, cur, nxt)
        print("Move:", tag, "to", nxt)
        cur = nxt

        time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")

#!/usr/bin/env python3
import math
import sys
import time
import heapq
import numpy as np
from picarx import Picarx

# ----------------- Config (tune on robot) -----------------
CELL_CM = 30
GRID_X_MIN, GRID_X_MAX = -2, 2    # x in [-5..5]
GRID_Y_MIN, GRID_Y_MAX = 0, 5    # y in [0..10]
FREE, OCC = 0, 1                  # unknown treated as FREE

SCAN_MIN_DEG = -45
SCAN_MAX_DEG = 45
SCAN_STEP_DEG = 2
MAX_RANGE_CM = 50

SPEED_FORWARD = 30
SPEED_BACKWARD = 30
T_FORWARD_CELL_SEC = 1.2
T_BACKWARD_CELL_SEC = 1.2
T_FORWARD_TURN_SEC = 0.32
STEER_MAX_DEG = 30
TURN_LEFT_SEQ = (0.5, 1.4, 1.3)
TURN_RIGHT_SEQ = (0.5, 1.3, 1.3)

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
def astar(grid: Grid, start, goal, heading = "North"):
    sx, sy = start
    gx, gy = goal
    if not (grid.in_bounds_xy(gx, gy) and grid.in_bounds_xy(sx, sy)):
        return []

    def is_free(x, y):
        r, c = grid.world_to_idx(x, y)
        return grid.grid[r, c] != OCC

    def heuristic(p, q):
        (px, py) = p
        (qx, qy) = q
        return abs(px - qx) + abs(py - qy)

    def turn_penalty(from_heading, to_heading):
        return 0 if from_heading == to_heading else 2

    def heading_to(from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 0 and dy == 1: return "north"
        if dx == 1 and dy == 0: return "east"
        if dx == 0 and dy == -1: return "south"
        if dx == -1 and dy == 0: return "west"
        return "unknown"

    openq = []
    heapq.heappush(openq, (heuristic(start, goal), 0, start, heading))
    came = {}
    gscore = {(start, heading): 0}

    def neighbors(node):
        x, y = node
        for dx, dy in ((0,1),(1,0),(0,-1),(-1,0)):
            nx, ny = x + dx, y + dy
            if grid.in_bounds_xy(nx, ny) and (is_free(nx, ny) or (nx, ny) == goal):
                yield (nx, ny)

    while openq:
        f_score, g, cur, cur_h = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            key = (cur, cur_h)
            while key in came:
                cur, cur_h = came[key]
                path.append(cur)
                key = (cur, cur_h)
            return list(reversed(path))

        for nb in neighbors(cur):
            new_h = heading_to(cur, nb)
            ng = g + 1 + turn_penalty(cur_h, new_h)
            key_nb = (nb, new_h)
            if ng < gscore.get(key_nb, 1e9):
                came[key_nb] = (cur, cur_h)
                gscore[key_nb] = ng
                f_new = ng + heuristic(nb, goal)
                heapq.heappush(openq, (f_new, ng, nb, new_h))
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

# ----------------- Check left/right scan for obstacle avoidance -----------------
def scan_for_escape_direction(px: Picarx):
    """Quick scan left and right to determine best escape direction"""
    # Look left
    px.set_cam_pan_angle(-45)
    time.sleep(0.1)
    left_dist = px.get_distance()
    if left_dist is None:
        left_dist = MAX_RANGE_CM
    
    # Look right
    px.set_cam_pan_angle(45)
    time.sleep(0.1)
    right_dist = px.get_distance()
    if right_dist is None:
        right_dist = MAX_RANGE_CM
    
    # Reset to center
    px.set_cam_pan_angle(0)
    time.sleep(0.1)
    
    print(f"Escape scan - Left: {left_dist:.1f}cm, Right: {right_dist:.1f}cm")
    return left_dist, right_dist

# ----------------- Minimal motion: 1-cell forward or 3-pt turn + forward -----------------
def forward_one(px: Picarx):
    px.set_dir_servo_angle(0)
    px.forward(SPEED_FORWARD)
    time.sleep(T_FORWARD_CELL_SEC)
    px.stop()


def forward_turn(px: Picarx):
    px.set_dir_servo_angle(0)
    px.forward(SPEED_FORWARD)
    time.sleep(T_FORWARD_TURN_SEC)
    px.stop()

def backward_one(px: Picarx):
    px.set_dir_servo_angle(0)
    px.backward(SPEED_BACKWARD)
    time.sleep(T_BACKWARD_CELL_SEC)
    px.stop()

def three_point_left(px: Picarx):
    t1,t2,t3 = TURN_LEFT_SEQ
    px.forward(SPEED_FORWARD); time.sleep(t1); px.stop()
    px.set_dir_servo_angle(+STEER_MAX_DEG); px.backward(SPEED_BACKWARD); time.sleep(t2); px.stop()
    px.set_dir_servo_angle(-STEER_MAX_DEG); px.forward(SPEED_FORWARD); time.sleep(t3); px.stop()
    px.set_dir_servo_angle(0)

def three_point_right(px: Picarx):
    t1,t2,t3 = TURN_RIGHT_SEQ
    px.forward(SPEED_FORWARD); time.sleep(t1); px.stop()
    px.set_dir_servo_angle(-STEER_MAX_DEG); px.backward(SPEED_BACKWARD); time.sleep(t2); px.stop()
    px.set_dir_servo_angle(+STEER_MAX_DEG); px.forward(SPEED_FORWARD); time.sleep(t3); px.stop()
    px.set_dir_servo_angle(0)

# Ego heading is +y; decide primitive for next step (dx,dy)
def execute_step(px: Picarx, cur, nxt, current_heading):
    dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
    
    # Calculate what direction we need to move relative to current heading
    if current_heading == "North":
        # Car facing north (+y), so (dx, dy) is relative to north
        if (dx, dy) == (0, 1):   # forward
            forward_one(px)
            return "North"
        elif (dx, dy) == (-1, 0):  # left turn + forward
            three_point_left(px)
            forward_turn(px)
            return "West"
        elif (dx, dy) == (1, 0):   # right turn + forward
            three_point_right(px)
            forward_turn(px)
            return "East"
        elif (dx, dy) == (0, -1):  # backward
            backward_one(px)
            return "South"
            
    elif current_heading == "East":
        # Car facing east (+x), so (dx, dy) needs to be rotated
        if (dx, dy) == (1, 0):   # forward (east)
            forward_one(px)
            return "East"
        elif (dx, dy) == (0, 1):  # left turn + forward (north)
            three_point_left(px)
            forward_turn(px)
            return "North"
        elif (dx, dy) == (0, -1):   # right turn + forward (south)
            three_point_right(px)
            forward_turn(px)
            return "South"
        elif (dx, dy) == (-1, 0):  # backward (west)
            backward_one(px)
            return "West"
            
    elif current_heading == "South":
        # Car facing south (-y)
        if (dx, dy) == (0, -1):   # forward (south)
            forward_one(px)
            return "South"
        elif (dx, dy) == (1, 0):  # left turn + forward (east)
            three_point_left(px)
            forward_turn(px)
            return "East"
        elif (dx, dy) == (-1, 0):   # right turn + forward (west)
            three_point_right(px)
            forward_turn(px)
            return "West"
        elif (dx, dy) == (0, 1):  # backward (north)
            backward_one(px)
            return "North"
            
    elif current_heading == "West":
        # Car facing west (-x)
        if (dx, dy) == (-1, 0):   # forward (west)
            forward_one(px)
            return "West"
        elif (dx, dy) == (0, -1):  # left turn + forward (south)
            three_point_left(px)
            forward_turn(px)
            return "South"
        elif (dx, dy) == (0, 1):   # right turn + forward (north)
            three_point_right(px)
            forward_turn(px)
            return "North"
        elif (dx, dy) == (1, 0):  # backward (east)
            backward_one(px)
            return "East"
    
    return current_heading  # no movement

# ----------------- Main loop with active obstacle avoidance -----------------
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
    heading = "North"
    consecutive_blocked = 0
    last_escape_direction = None

    while True:
        # Plan
        path = astar(g, cur, goal, heading)
        print("Path:", path)
        print(g.grid)

        if not path:
            print("No path found. Exploring by turning...")
            # No path exists - try to explore by turning
            left_dist, right_dist = scan_for_escape_direction(px)
            
            # Turn towards the more open direction
            if left_dist > right_dist:
                print("Turning left to explore...")
                three_point_left(px)
                heading = "West" if heading == "North" else \
                         "North" if heading == "East" else \
                         "East" if heading == "South" else "South"
            else:
                print("Turning right to explore...")
                three_point_right(px)
                heading = "East" if heading == "North" else \
                         "South" if heading == "East" else \
                         "West" if heading == "South" else "North"
            
            # After turning, scan the new area
            time.sleep(0.2)
            sweep_and_update(px, g)
            continue

        if len(path) == 1 and path[0] == goal and cur == goal:
            print("Reached destination!")
            px.stop()
            break

        if len(path) < 2:
            sweep_and_update(px, g)
            continue

        # Before moving, sweep and check if any cell on path up to a small horizon is blocked
        sweep_and_update(px, g)
        horizon = min(2, len(path)-1)  # Reduced horizon for quicker response
        blocked = False
        blocked_at = None
        for i in range(1, horizon+1):
            x, y = path[i]
            r, c = g.world_to_idx(x, y)
            if g.grid[r, c] == OCC:
                blocked = True
                blocked_at = i
                break
        
        if blocked:
            consecutive_blocked += 1
            print(f"Path blocked at step {blocked_at}. Count: {consecutive_blocked}")
            
            # Active avoidance: turn to explore around the obstacle
            if consecutive_blocked >= 2:  # After 2 attempts, actively avoid
                print("Executing obstacle avoidance maneuver...")
                
                # Scan to decide which way to turn
                left_dist, right_dist = scan_for_escape_direction(px)
                
                # Alternate escape direction if we keep getting stuck
                if consecutive_blocked >= 4 and last_escape_direction:
                    # Try opposite of last escape
                    if last_escape_direction == "left":
                        print("Forcing right turn after repeated blocks...")
                        three_point_right(px)
                        heading = "East" if heading == "North" else \
                                 "South" if heading == "East" else \
                                 "West" if heading == "South" else "North"
                        last_escape_direction = "right"
                    else:
                        print("Forcing left turn after repeated blocks...")
                        three_point_left(px)
                        heading = "West" if heading == "North" else \
                                 "North" if heading == "East" else \
                                 "East" if heading == "South" else "South"
                        last_escape_direction = "left"
                    consecutive_blocked = 0
                    
                # Normal escape based on scan
                elif left_dist > right_dist + 10:  # Significant difference
                    print(f"Turning left (left: {left_dist:.1f}cm > right: {right_dist:.1f}cm)")
                    three_point_left(px)
                    heading = "West" if heading == "North" else \
                             "North" if heading == "East" else \
                             "East" if heading == "South" else "South"
                    last_escape_direction = "left"
                    
                elif right_dist > left_dist + 10:  # Significant difference
                    print(f"Turning right (right: {right_dist:.1f}cm > left: {left_dist:.1f}cm)")
                    three_point_right(px)
                    heading = "East" if heading == "North" else \
                             "South" if heading == "East" else \
                             "West" if heading == "South" else "North"
                    last_escape_direction = "right"
                    
                else:
                    # Similar distances, try backing up first
                    print("Similar clearances, backing up...")
                    backward_one(px)
                    # Then turn based on small preference
                    if left_dist >= right_dist:
                        three_point_left(px)
                        heading = "West" if heading == "North" else \
                                 "North" if heading == "East" else \
                                 "East" if heading == "South" else "South"
                        last_escape_direction = "left"
                    else:
                        three_point_right(px)
                        heading = "East" if heading == "North" else \
                                 "South" if heading == "East" else \
                                 "West" if heading == "South" else "North"
                        last_escape_direction = "right"
                
                # After escape maneuver, do a new scan
                time.sleep(0.2)
                sweep_and_update(px, g)
                
                # Move forward one cell to get away from obstacle
                forward_one(px)
                # Update position based on new heading
                if heading == "North": cur = (cur[0], cur[1]+1)
                elif heading == "East": cur = (cur[0]+1, cur[1])
                elif heading == "South": cur = (cur[0], cur[1]-1)
                elif heading == "West": cur = (cur[0]-1, cur[1])
                
                consecutive_blocked = 0  # Reset after escape
                
            continue

        # Path is clear, execute one step
        consecutive_blocked = 0  # Reset blocked counter
        nxt = path[1]
        dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
        new_heading = execute_step(px, cur, nxt, heading)
        print(f"Move: ({dx}, {dy}) to {nxt}, new heading: {new_heading}")
        cur = nxt
        heading = new_heading
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        Picarx().stop()

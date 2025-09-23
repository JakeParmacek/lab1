#!/usr/bin/env python3
import math
import sys
import time
from collections import deque

try:
    import numpy as np
except ImportError:
    print("numpy is required. Install with: pip3 install numpy", file=sys.stderr)
    sys.exit(1)

# ---------- Constants (calibrate on device) ----------
CELL_CM = 20
GRID_X_MIN, GRID_X_MAX = -5, 5    # inclusive
GRID_Y_MIN, GRID_Y_MAX = 0, 10
UNKNOWN, FREE, OCC = -1, 0, 1

SCAN_MIN_DEG = -45
SCAN_MAX_DEG = 45
SCAN_STEP_DEG = 2
MAX_RANGE_CM = 100
STOP_DISTANCE_CM = 30

SPEED_FORWARD = 30
SPEED_BACKWARD = 25
T_FORWARD_CELL_SEC = 0.80
T_BACKWARD_CELL_SEC = 0.90
STEER_MAX_DEG = 30

TURN_LEFT_SEQ = (0.45, 0.40, 0.35)   # t1, t2, t3 (seconds) — calibrate
TURN_RIGHT_SEQ = (0.45, 0.40, 0.35)  # mirror of left — calibrate

# ---------- Hardware Abstraction ----------
class Hardware:
    def __init__(self):
        self.px = None
        self.hw_ok = False
        self._init_px()

    def _init_px(self):
        try:
            from picarx import Picarx
            self.px = Picarx()
            self.hw_ok = True
        except Exception as e:
            print(f"[WARN] Picarx init failed: {e}\nRunning in logic-only mode.", file=sys.stderr)
            self.px = None
            self.hw_ok = False

    # Steering for driving
    def set_steer(self, angle_deg):
        if not self.px:
            print(f"[SIM] steer {angle_deg}°")
            return
        try:
            self.px.set_dir_servo_angle(angle_deg)
        except Exception as e:
            print(f"[WARN] set_dir_servo_angle failed: {e}", file=sys.stderr)

    # Pan servo for ultrasonic sweep (fallback to steering servo if no pan)
    def set_scan_servo(self, angle_deg):
        if not self.px:
            print(f"[SIM] scan_servo {angle_deg}°")
            return
        for name in ["set_cam_pan_angle", "set_ultrasonic_servo_angle", "set_servo_pulse"]:
            fn = getattr(self.px, name, None)
            if callable(fn):
                try:
                    fn(angle_deg)
                    return
                except Exception:
                    pass
        # Fallback: nudge steering servo just to move sensor if mounted there
        try:
            self.px.set_dir_servo_angle(angle_deg)
        except Exception as e:
            print(f"[WARN] scan servo control failed: {e}", file=sys.stderr)

    def forward(self, speed):
        if not self.px:
            print(f"[SIM] forward speed={speed}")
            return
        try:
            self.px.forward(speed)
        except Exception as e:
            print(f"[WARN] forward failed: {e}", file=sys.stderr)

    def backward(self, speed):
        if not self.px:
            print(f"[SIM] backward speed={speed}")
            return
        try:
            self.px.backward(speed)
        except Exception as e:
            print(f"[WARN] backward failed: {e}", file=sys.stderr)

    def stop(self):
        if not self.px:
            print("[SIM] stop")
            return
        try:
            self.px.stop()
        except Exception as e:
            print(f"[WARN] stop failed: {e}", file=sys.stderr)

    def get_distance_cm(self):
        if not self.px:
            # Simulated: no obstacle
            return MAX_RANGE_CM
        # Try px.get_distance() first
        gd = getattr(self.px, "get_distance", None)
        if callable(gd):
            try:
                d = gd()
                return d
            except Exception:
                pass
        # Try robot_hat Ultrasonic as last resort (pins need config; skip here)
        print("[WARN] distance read not available; assuming MAX_RANGE", file=sys.stderr)
        return MAX_RANGE_CM

# ---------- Utils ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rotate_point_left(p):
    x, y = p
    return (-y, x)

def rotate_point_right(p):
    x, y = p
    return (y, -x)

# ---------- Egocentric Grid ----------
class EgocentricGrid:
    def __init__(self):
        self.w = GRID_X_MAX - GRID_X_MIN + 1
        self.h = GRID_Y_MAX - GRID_Y_MIN + 1
        self.grid = np.full((self.h, self.w), UNKNOWN, dtype=np.int8)  # row=y, col=x
        self.inflated = self.grid.copy()

    def world_to_idx(self, x, y):
        col = x - GRID_X_MIN
        row = y - GRID_Y_MIN
        return row, col

    def idx_to_world(self, row, col):
        x = col + GRID_X_MIN
        y = row + GRID_Y_MIN
        return x, y

    def in_bounds_xy(self, x, y):
        return GRID_X_MIN <= x <= GRID_X_MAX and GRID_Y_MIN <= y <= GRID_Y_MAX

    def idx_in_bounds(self, row, col):
        return 0 <= row < self.h and 0 <= col < self.w

    def integrate_scan(self, scan_points_cm):
        # scan_points_cm: list of (angle_deg, range_cm or None)
        # Ego frame: origin at (0,0), forward +y, angles positive to the left (CCW)
        for ang_deg, rng in scan_points_cm:
            if rng is None:
                continue
            rng = clamp(rng, 0, MAX_RANGE_CM)
            # Step through the ray in small increments
            step = CELL_CM / 4.0  # cm
            steps = max(1, int(rng / step))
            hit_set = False
            for i in range(1, steps + 1):
                s = i * step
                # If the measured range is less than MAX_RANGE, mark hit cell at the first step >= rng
                if s > rng:
                    break
                rad = math.radians(ang_deg)
                x_cm = math.sin(rad) * s  # left is +x
                y_cm = math.cos(rad) * s  # forward is +y
                x = int(round(x_cm / CELL_CM))
                y = int(round(y_cm / CELL_CM))
                if not self.in_bounds_xy(x, y):
                    break
                r, c = self.world_to_idx(x, y)
                if rng < MAX_RANGE_CM and abs(s - rng) < (step * 1.1):
                    self.grid[r, c] = OCC
                    hit_set = True
                    break
                else:
                    if self.grid[r, c] == UNKNOWN:
                        self.grid[r, c] = FREE
            # If the range was MAX_RANGE (no hit), cells along the whole ray were freed above.

    def inflate(self):
        g = self.grid
        inf = np.array(g, copy=True)
        # Treat borders as occupied by inflating them: create an occupied border layer
        for r in range(self.h):
            for c in range(self.w):
                if g[r, c] == OCC:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < self.h and 0 <= cc < self.w:
                                inf[rr, cc] = OCC
        # Inflate map borders
        inf[0, :] = OCC
        inf[:, 0] = OCC
        inf[self.h - 1, :] = OCC
        inf[:, self.w - 1] = OCC
        self.inflated = inf

    def shift_after_forward(self, cells=1):
        cells = int(cells)
        if cells <= 0:
            return
        # Moving forward: world features move toward row-1 (closer)
        self.grid = np.roll(self.grid, -cells, axis=0)
        self.grid[self.h - cells :, :] = UNKNOWN
        self.inflated = np.roll(self.inflated, -cells, axis=0)
        self.inflated[self.h - cells :, :] = UNKNOWN

    def shift_after_backward(self, cells=1):
        cells = int(cells)
        if cells <= 0:
            return
        self.grid = np.roll(self.grid, cells, axis=0)
        self.grid[:cells, :] = UNKNOWN
        self.inflated = np.roll(self.inflated, cells, axis=0)
        self.inflated[:cells, :] = UNKNOWN

    def rotate_after_turn_left(self):
        # Robot turned left; rotate map clockwise to keep ego heading +y
        self.grid = np.rot90(self.grid, k=-1)
        self.inflated = np.rot90(self.inflated, k=-1)

    def rotate_after_turn_right(self):
        # Robot turned right; rotate map counter-clockwise
        self.grid = np.rot90(self.grid, k=1)
        self.inflated = np.rot90(self.inflated, k=1)

# ---------- Planner (A*) ----------
def a_star_path(inflated_grid, start_xy, goal_xy):
    h, w = inflated_grid.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    def in_bounds(x, y):
        return GRID_X_MIN <= x <= GRID_X_MAX and GRID_Y_MIN <= y <= GRID_Y_MAX

    def is_free(x, y):
        r, c = y - GRID_Y_MIN, x - GRID_X_MIN
        if not (0 <= r < h and 0 <= c < w):
            return False
        return inflated_grid[r, c] == FREE

    if not in_bounds(gx, gy):
        return []

    # Treat unknown as blocked; start cell allowed even if unknown
    open_set = []
    came = {}
    gscore = {}
    fscore = {}

    start = (sx, sy)
    goal = (gx, gy)

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    import heapq
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))
    gscore[start] = 0
    fscore[start] = manhattan(start, goal)

    def neighbors(node):
        x, y = node
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            r, c = ny - GRID_Y_MIN, nx - GRID_X_MIN
            v = inflated_grid[r, c]
            if v == FREE or (nx, ny) == goal:
                yield (nx, ny)

    while open_set:
        _, g, cur = heapq.heappop(open_set)
        if cur == goal:
            # reconstruct
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path

        for nb in neighbors(cur):
            tentative = g + 1
            if tentative < gscore.get(nb, 1e9):
                came[nb] = cur
                gscore[nb] = tentative
                f = tentative + manhattan(nb, goal)
                fscore[nb] = f
                heapq.heappush(open_set, (f, tentative, nb))

    return []  # no path

# ---------- Scanner ----------
class Scanner:
    def __init__(self, hw: Hardware):
        self.hw = hw

    def sweep(self):
        points = []
        # Sweep pan from min..max and back to 0 at end
        for ang in range(SCAN_MIN_DEG, SCAN_MAX_DEG + 1, SCAN_STEP_DEG):
            self.hw.set_scan_servo(ang)
            time.sleep(0.01)
            d = self.hw.get_distance_cm()
            if d is None or d <= 0:
                rng = MAX_RANGE_CM
            else:
                rng = clamp(d, 0, MAX_RANGE_CM)
            points.append((ang, rng))
        # Return scan servo to 0 deg
        self.hw.set_scan_servo(0)
        return points

    def forward_distance(self):
        # Quick 0° shot
        self.hw.set_scan_servo(0)
        time.sleep(0.01)
        d = self.hw.get_distance_cm()
        if d is None or d <= 0:
            return None
        return clamp(d, 0, MAX_RANGE_CM)

# ---------- Executor ----------
class Executor:
    def __init__(self, hw: Hardware):
        self.hw = hw

    def stop(self):
        self.hw.stop()

    def forward_one(self):
        self.hw.set_steer(0)
        self.hw.forward(SPEED_FORWARD)
        time.sleep(T_FORWARD_CELL_SEC)
        self.hw.stop()

    def backward_one(self):
        self.hw.set_steer(0)
        self.hw.backward(SPEED_BACKWARD)
        time.sleep(T_BACKWARD_CELL_SEC)
        self.hw.stop()

    def three_point_left(self):
        t1, t2, t3 = TURN_LEFT_SEQ
        # 1) steer max-left, forward
        self.hw.set_steer(+STEER_MAX_DEG)
        self.hw.forward(SPEED_FORWARD)
        time.sleep(t1)
        self.hw.stop()
        # 2) steer max-right, reverse
        self.hw.set_steer(-STEER_MAX_DEG)
        self.hw.backward(SPEED_BACKWARD)
        time.sleep(t2)
        self.hw.stop()
        # 3) steer max-left, forward
        self.hw.set_steer(+STEER_MAX_DEG)
        self.hw.forward(SPEED_FORWARD)
        time.sleep(t3)
        self.hw.stop()
        # straighten
        self.hw.set_steer(0)

    def three_point_right(self):
        t1, t2, t3 = TURN_RIGHT_SEQ
        # 1) steer max-right, forward
        self.hw.set_steer(-STEER_MAX_DEG)
        self.hw.forward(SPEED_FORWARD)
        time.sleep(t1)
        self.hw.stop()
        # 2) steer max-left, reverse
        self.hw.set_steer(+STEER_MAX_DEG)
        self.hw.backward(SPEED_BACKWARD)
        time.sleep(t2)
        self.hw.stop()
        # 3) steer max-right, forward
        self.hw.set_steer(-STEER_MAX_DEG)
        self.hw.forward(SPEED_FORWARD)
        time.sleep(t3)
        self.hw.stop()
        # straighten
        self.hw.set_steer(0)

    def follow_next_step(self, curr_xy, next_xy):
        cx, cy = curr_xy
        nx, ny = next_xy
        dx, dy = nx - cx, ny - cy
        # Ego heading is +y
        if dx == 0 and dy == 1:
            self.forward_one()
            return "FORWARD"
        if dx == -1 and dy == 0:
            self.three_point_left()
            return "TURN_LEFT"
        if dx == 1 and dy == 0:
            self.three_point_right()
            return "TURN_RIGHT"
        if dx == 0 and dy == -1:
            # Prefer turning twice over back; but allow back if tight
            # Simple policy: back up one cell
            self.backward_one()
            return "BACKWARD"
        # Unexpected step (diagonal or out of 4-neighbor)
        return "NONE"

# ---------- I/O ----------
def prompt_goal_once():
    print("Enter destination grid coordinates x y (x in [-5..5], y in [0..10]), relative to start (0,0):")
    while True:
        line = sys.stdin.readline()
        if not line:
            print("No input received; exiting.", file=sys.stderr)
            sys.exit(1)
        try:
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError("need two integers")
            x = int(parts[0])
            y = int(parts[1])
            if not (GRID_X_MIN <= x <= GRID_X_MAX and GRID_Y_MIN <= y <= GRID_Y_MAX):
                raise ValueError("out of range")
            return (x, y)
        except Exception as e:
            print(f"Invalid input ({e}). Please enter: x y within ranges.", file=sys.stderr)

# ---------- Main Loop ----------
def main():
    hw = Hardware()
    scanner = Scanner(hw)
    execu = Executor(hw)
    grid = EgocentricGrid()

    goal = prompt_goal_once()
    print(f"Goal set to {goal}")

    # Main loop
    while True:
        # 1) Scan
        scans = scanner.sweep()
        # 2) Integrate and inflate
        grid.integrate_scan(scans)
        grid.inflate()
        # 3) Plan
        path = a_star_path(grid.inflated, (0, 0), goal)
        print("A* path:", path)

        # Reached?
        if path and len(path) == 1 and path[0] == (0, 0) and goal == (0, 0):
            print("Destination reached.")
            execu.stop()
            break

        if not path or len(path) < 2:
            # Blocked or goal currently inflated; wait and rescan
            print("No path available; rescanning...")
            time.sleep(0.2)
            continue

        # 4) Safety check ahead if next step is forward
        next_xy = path[1]
        if next_xy == (0, 1):
            dist = scanner.forward_distance()
            if dist is not None and dist < STOP_DISTANCE_CM:
                print(f"Obstacle too close ahead ({dist} cm). Replanning...")
                execu.stop()
                time.sleep(0.1)
                continue

        # 5) Execute one primitive toward next step
        primitive = execu.follow_next_step((0, 0), next_xy)

        # 6) Ego-frame transform map and goal
        if primitive == "FORWARD":
            grid.shift_after_forward(1)
            goal = (goal[0], goal[1] - 1)
        elif primitive == "BACKWARD":
            grid.shift_after_backward(1)
            goal = (goal[0], goal[1] + 1)
        elif primitive == "TURN_LEFT":
            grid.rotate_after_turn_left()
            goal = rotate_point_right(goal)
        elif primitive == "TURN_RIGHT":
            grid.rotate_after_turn_right()
            goal = rotate_point_left(goal)
        else:
            # Nothing executed; small pause
            time.sleep(0.05)

        execu.stop()
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping.")

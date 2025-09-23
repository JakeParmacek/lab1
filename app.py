# picarx_nav/app.py

import time
import math
import numpy as np
from picarx import Picarx

from .config import (
    WIDTH, HEIGHT, RES_CM, INFLATE_CELLS, DRIVE_POWER,
    LEFT_DEG, RIGHT_DEG, STRAIGHT_DEG, STEP_DRIVE_SEC
)
from .mapping import build_map, world_to_grid
from .planning import inflate_obstacles, a_star, destination_to_goal_cell, nearest_free_cell
from .control import next_steer_from_path, cell_to_world_cm

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
        start_cell_0 = (height - 1, width // 2)   # origin cell
        start_cell   = tuple(start_cell_0)        # current "pose" cell

        # Convert requested destination to goal cell once
        requested_goal = destination_to_goal_cell(
            dest_x_cm, dest_y_cm, width, height, res_cm, world_to_grid_fn=world_to_grid
        )

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

            # Pose & distance readout
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

            # Arrived check
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

    finally:
        try:
            px.stop()
        except:
            pass

if __name__ == "__main__":
    main()

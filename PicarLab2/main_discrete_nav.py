#!/usr/bin/env python3
"""
Main discrete navigation program for PiCar-X
Uses 90-degree turns and grid-based movement
"""

import time
import numpy as np
from picarx import Picarx

from grid_mapping import GridMapper
from pathfinding import PathFinder
from robot_controller import GridRobot, ActionPlanner

def prompt_destination_cm():
    """Get destination from user in centimeters"""
    print("\n" + "="*50)
    print("Enter destination in ROBOT frame (centimeters).")
    print("  x: right is positive; left is negative")
    print("  y: forward is positive; behind is negative")
    print("="*50)
    
    while True:
        try:
            x = float(input("Destination X (cm): ").strip())
            y = float(input("Destination Y (cm): ").strip())
            
            if y < 0:
                print("Warning: Negative Y means behind robot. Are you sure? (y/n)")
                if input().lower() != 'y':
                    continue
                    
            return x, y
        except ValueError:
            print("Please enter numeric values. Try again.\n")

def main():
    # Configuration
    WIDTH, HEIGHT = 25, 25  # Grid dimensions
    RES_CM = 5  # Cell size in centimeters
    INFLATE_CELLS = 1  # Safety buffer
    EMERGENCY_STOP_CM = 20  # Stop if obstacle closer than this
    
    # Get destination
    dest_x_cm, dest_y_cm = prompt_destination_cm()
    print(f"\nNavigating to: ({dest_x_cm:.1f}, {dest_y_cm:.1f}) cm")
    
    # Initialize components
    px = Picarx()
    mapper = GridMapper(width=WIDTH, height=HEIGHT, res_cm=RES_CM)
    pathfinder = PathFinder()
    robot = GridRobot(width=WIDTH, height=HEIGHT)
    planner = ActionPlanner()
    
    # Convert destination to grid cell
    goal_cell = mapper.destination_to_goal_cell(dest_x_cm, dest_y_cm)
    print(f"Goal cell: {goal_cell}")
    
    # For numpy printing
    np.set_printoptions(linewidth=140, threshold=np.inf)
    
    try:
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration}")
            print(f"Robot position: cell ({robot.x}, {robot.y}), orientation: {robot.orientation}")
            print(f"{'='*50}")
            
            # Stop and scan
            px.stop()
            time.sleep(0.1)
            
            print("Scanning environment...")
            occ_grid = mapper.build_map(px)
            
            # Print occupancy grid
            print("\n--- Occupancy Grid (0=free, 1=obstacle) ---")
            print(occ_grid)
            
            # Inflate obstacles
            costmap = mapper.inflate_obstacles(occ_grid, INFLATE_CELLS)
            
            print("\n--- Inflated Costmap ---")
            print(costmap)
            
            # Find reachable goal
            reachable_goal = pathfinder.nearest_free_cell(
                costmap, goal_cell[0], goal_cell[1]
            )
            
            if reachable_goal is None:
                print("ERROR: No reachable goal found!")
                break
            
            # Check if we've reached the goal
            current_cell = (robot.y, robot.x)
            if current_cell == reachable_goal:
                print("\n*** GOAL REACHED! ***")
                px.stop()
                break
            
            # Plan path
            print(f"\nPlanning path from {current_cell} to {reachable_goal}...")
            path = pathfinder.a_star(costmap, current_cell, reachable_goal)
            
            if not path or len(path) < 2:
                print("No path found! Stopping.")
                break
            
            print(f"Path found with {len(path)} cells")
            
            # Convert path to actions
            actions = planner.plan_action_sequence(path, robot)
            
            if not actions:
                print("No actions to execute!")
                break
            
            print(f"Action sequence ({len(actions)} actions): {actions[:10]}...")
            
            # Execute actions with safety checks
            actions_executed = 0
            for action in actions:
                # Safety check before each move
                if action == 'forward':
                    distance = robot.quick_forward_scan(px)
                    
                    if distance and distance < EMERGENCY_STOP_CM:
                        print(f"\n! Obstacle detected at {distance:.1f}cm! Replanning...")
                        break
                
                # Execute the action
                print(f"  Executing: {action}")
                robot.execute_action(px, action)
                actions_executed += 1
                
                # Limit actions per iteration to allow replanning
                if actions_executed >= 5:  # Execute max 5 actions then rescan
                    print("  Rescanning after 5 actions...")
                    break
            
            # Small pause before next iteration
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nStopping robot...")
        px.stop()
        time.sleep(0.2)
        print("Navigation terminated.")

if __name__ == "__main__":
    main()
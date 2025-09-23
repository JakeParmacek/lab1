import time
import math
import numpy as np

class GridMapper:
    def __init__(self, width=25, height=25, res_cm=5):
        self.width = width
        self.height = height
        self.res_cm = res_cm
        self.max_range_cm = 100
        self.pan_settle_sec = 0.08
        
    def build_map(self, px):
        """
        Sweep camera pan from -45..+45 deg in 2 deg steps.
        Mark 1s in grid where the ultrasonic detects an obstacle.
        """
        grid = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for ang in range(-45, 45, 2):
            px.set_cam_pan_angle(ang)
            time.sleep(self.pan_settle_sec)
            d = px.ultrasonic.read()
            
            if (d is None) or (d <= 0) or (d > self.max_range_cm):
                continue
                
            x_cm, y_cm = self.polar_to_xy(ang, d)
            gx, gy = self.world_to_grid(x_cm, y_cm)
            grid[gy, gx] = 1
            
        return grid
    
    def polar_to_xy(self, angle_deg, distance_cm):
        """
        Convert polar to robot-centered (x,y) in cm.
        y is forward, x is right. +10cm bias for close hits.
        """
        theta = math.radians(angle_deg)
        x = distance_cm * math.sin(theta)
        y = distance_cm * math.cos(theta) + 10.0
        return x, y
    
    def world_to_grid(self, x_cm, y_cm):
        """Map robot-centric (x_cm, y_cm) to grid indices (gx, gy)."""
        gx = int(round(x_cm / self.res_cm)) + self.width // 2
        gy = int(round(y_cm / self.res_cm))
        gx = max(0, min(self.width - 1, gx))
        gy = max(0, min(self.height - 1, gy))
        return gx, gy
    
    def inflate_obstacles(self, grid, inflation_cells=1):
        """Add safety buffer around obstacles"""
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

    def destination_to_goal_cell(self, dest_x_cm, dest_y_cm):
        """Convert destination in cm to grid cell"""
        gx, gy = self.world_to_grid(dest_x_cm, dest_y_cm)
        return (gy, gx)
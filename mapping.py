# picarx_nav/mapping.py

import time
import math
import numpy as np
from .config import PAN_SETTLE_SEC, MAX_RANGE_CM, RES_CM, SWEEP_START, SWEEP_END, SWEEP_STEP

def build_map(px, width, height, res_cm=RES_CM):
    """
    Sweep camera pan from SWEEP_START..SWEEP_END in SWEEP_STEP degrees.
    Mark 1s in grid where the ultrasonic detects an obstacle.
    Robot-centric coordinates (cm): +y forward, +x right.
    Robot sits near (center, bottom).
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    for ang in range(SWEEP_START, SWEEP_END, SWEEP_STEP):
        px.set_cam_pan_angle(ang)
        time.sleep(PAN_SETTLE_SEC)  # let servo settle
        d = px.ultrasonic.read()    # distance in cm
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

# picarx_nav/control.py

from .config import LEFT_DEG, RIGHT_DEG, STRAIGHT_DEG

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

def cell_to_world_cm(cell_y, cell_x, start_y, start_x, res_cm):
    """
    Convert current cell (y,x) displacement from start cell into (x_cm, y_cm)
    in the robot's start-centered frame: x right (+), y forward (+).
    """
    dx_cells = (cell_x - start_x)           # +right
    dy_cells = (start_y - cell_y)           # moving "up" is +forward
    x_cm = dx_cells * res_cm
    y_cm = dy_cells * res_cm
    return x_cm, y_cm

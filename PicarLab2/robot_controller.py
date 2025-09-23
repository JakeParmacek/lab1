import time

class GridRobot:
    def __init__(self, width=25, height=25):
        self.x = width // 2  # Start center-bottom
        self.y = height - 1
        self.width = width
        self.height = height
        self.orientation = 'north'  # Facing up initially
        
        # Calibration parameters
        self.turn_90_duration = 0.8  # Time for 90-degree turn (needs tuning)
        self.cell_forward_duration = 0.5  # Time to move one cell (needs tuning)
        self.turn_power = 25
        self.forward_power = 30
        
    def get_next_cell(self, action='forward'):
        """Get the cell we'd move to with given action"""
        dx, dy = 0, 0
        
        if action == 'forward':
            if self.orientation == 'north':
                dy = -1
            elif self.orientation == 'south':
                dy = 1
            elif self.orientation == 'east':
                dx = 1
            elif self.orientation == 'west':
                dx = -1
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check bounds
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            return new_x, new_y
        return self.x, self.y
    
    # In robot_controller.py, REPLACE the execute_90_degree_turn method (around lines 35-56) with:

def execute_90_degree_turn(self, px, direction):
    """
    Execute a 90-degree turn using backward-forward technique
    This creates a much tighter turning radius
    """
    if direction == 'left':
        print("  Executing 90° left turn (backward-forward technique)...")
        
        # Phase 1: Backward with right steering
        px.set_dir_servo_angle(30)  # Steer right
        px.backward(30)  # Go backward
        time.sleep(0.5)
        
        # Phase 2: Forward with left steering  
        px.set_dir_servo_angle(-35)  # Steer left
        px.forward(30)
        time.sleep(0.8)
        
        # Phase 3: Additional forward to complete turn if needed
        px.set_dir_servo_angle(-30)
        px.forward(25)
        time.sleep(0.3)
        
        # Stop and straighten
        px.stop()
        px.set_dir_servo_angle(0)
        
        # Update orientation
        orientations = ['north', 'west', 'south', 'east']
        idx = orientations.index(self.orientation)
        self.orientation = orientations[(idx + 1) % 4]
        
    elif direction == 'right':
        print("  Executing 90° right turn (backward-forward technique)...")
        
        # Phase 1: Backward with left steering
        px.set_dir_servo_angle(-30)  # Steer left
        px.backward(30)  # Go backward
        time.sleep(0.5)
        
        # Phase 2: Forward with right steering
        px.set_dir_servo_angle(35)  # Steer right
        px.forward(30)
        time.sleep(0.8)
        
        # Phase 3: Additional forward to complete turn if needed
        px.set_dir_servo_angle(30)
        px.forward(25)
        time.sleep(0.3)
        
        # Stop and straighten
        px.stop()
        px.set_dir_servo_angle(0)
        
        # Update orientation
        orientations = ['north', 'east', 'south', 'west']
        idx = orientations.index(self.orientation)
        self.orientation = orientations[(idx + 1) % 4]
    
    def move_forward_one_cell(self, px):
        """Move forward approximately one grid cell"""
        px.set_dir_servo_angle(0)
        px.forward(self.forward_power)
        time.sleep(self.cell_forward_duration)
        px.stop()
        
        # Update position
        new_x, new_y = self.get_next_cell('forward')
        self.x, self.y = new_x, new_y
    
    def execute_action(self, px, action):
        """Execute forward/left/right action"""
        if action == 'forward':
            self.move_forward_one_cell(px)
        elif action == 'left':
            self.execute_90_degree_turn(px, 'left')
        elif action == 'right':
            self.execute_90_degree_turn(px, 'right')
    
    def quick_forward_scan(self, px):
        """Quick ultrasonic check straight ahead"""
        px.set_cam_pan_angle(0)
        time.sleep(0.05)
        distance = px.ultrasonic.read()
        return distance

class ActionPlanner:
    def __init__(self):
        pass
    
    def calculate_turns(self, current_orientation, required_orientation):
        """Calculate turn sequence needed to face required orientation"""
        if current_orientation == required_orientation:
            return []
        
        # Define turn mapping
        orientation_to_index = {
            'north': 0, 'east': 1, 'south': 2, 'west': 3
        }
        
        current_idx = orientation_to_index[current_orientation]
        required_idx = orientation_to_index[required_orientation]
        
        # Calculate shortest turn direction
        diff = (required_idx - current_idx) % 4
        
        if diff == 1:
            return ['right']
        elif diff == 2:
            return ['right', 'right']  # or ['left', 'left']
        elif diff == 3:
            return ['left']
        
        return []
    
    def plan_action_sequence(self, path, robot_state):
        """
        Convert path to sequence of 'forward', 'left', 'right' commands
        """
        if not path or len(path) < 2:
            return []
        
        actions = []
        current_orientation = robot_state.orientation
        
        for i in range(len(path) - 1):
            current = path[i]
            next_cell = path[i + 1]
            
            # Determine required orientation
            dy = next_cell[0] - current[0]
            dx = next_cell[1] - current[1]
            
            required_orientation = None
            if dy == -1:
                required_orientation = 'north'
            elif dy == 1:
                required_orientation = 'south'
            elif dx == 1:
                required_orientation = 'east'
            elif dx == -1:
                required_orientation = 'west'
            
            # Add turn commands if needed
            turns_needed = self.calculate_turns(current_orientation, required_orientation)
            actions.extend(turns_needed)
            
            # Add forward command
            actions.append('forward')
            
            current_orientation = required_orientation
        

        return actions

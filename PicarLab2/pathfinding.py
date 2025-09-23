import heapq
import numpy as np

class PathFinder:
    def __init__(self):
        # 4-connected motions: (dy, dx, step_cost)
        self.MOTIONS = [
            (0, 1, 1.0),   # right
            (1, 0, 1.0),   # down
            (0, -1, 1.0),  # left
            (-1, 0, 1.0),  # up
        ]
    
    def manhattan(self, ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)
    
    def a_star(self, grid, start, goal):
        """
        grid: 2D np.array, 0=free, 1=occupied
        start, goal: (gy, gx)
        returns list[(gy,gx)] path or None
        """
        h, w = grid.shape
        inb = lambda y, x: 0 <= y < h and 0 <= x < w
        blocked = lambda y, x: grid[y, x] == 1
        
        sy, sx = start
        gy, gx = goal
        
        if blocked(gy, gx):
            return None
        
        openpq = []
        g_cost = {(sy, sx): 0.0}
        came = {}
        
        f0 = self.manhattan(sx, sy, gx, gy)
        heapq.heappush(openpq, (f0, 0.0, (sy, sx)))
        
        visited = set()
        
        while openpq:
            f, gc, (y, x) = heapq.heappop(openpq)
            
            if (y, x) == (gy, gx):
                return self._reconstruct_path(came, (y, x))
            
            if (y, x) in visited:
                continue
            visited.add((y, x))
            
            for dy, dx, step_cost in self.MOTIONS:
                ny, nx = y + dy, x + dx
                
                if not inb(ny, nx) or blocked(ny, nx):
                    continue
                    
                ng = gc + step_cost
                
                if ng < g_cost.get((ny, nx), float('inf')):
                    g_cost[(ny, nx)] = ng
                    came[(ny, nx)] = (y, x)
                    hf = self.manhattan(nx, ny, gx, gy)
                    heapq.heappush(openpq, (ng + hf, ng, (ny, nx)))
        
        return None
    
    def _reconstruct_path(self, came, node):
        path = [node]
        while node in came:
            node = came[node]
            path.append(node)
        path.reverse()
        return path
    
    def nearest_free_cell(self, costmap, gy, gx, max_radius=10):
        """Find nearest free cell if goal is blocked"""
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
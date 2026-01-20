import heapq
import math

INF = math.inf

# 8-neighborhood moves
MOVES = [
    (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
    (1, 1, 1.414), (-1, 1, 1.414),
    (1, -1, 1.414), (-1, -1, 1.414),
]

class DStarLite:
    def __init__(self, start, goal, grid):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.last = start
        self.km = 0.0
        
        self.g = {}
        self.rhs = {}
        self.U = []
        self.U_set = set()  # Track what's in queue to avoid duplicates

        H, W = self.grid.shape
        # Only initialize cells that are reachable (sparse initialization)
        self.rhs[self.goal] = 0.0
        self.g[self.goal] = INF
        
        key = self.key(self.goal)
        heapq.heappush(self.U, (key, self.goal))
        self.U_set.add(self.goal)

    def in_bounds(self, s):
        x, y = s
        H, W = self.grid.shape
        return 0 <= x < W and 0 <= y < H

    def traversable(self, s):
        x, y = s
        c = float(self.grid[y, x])
        return (c > 0.0) and (not math.isinf(c))

    def h(self, a, b):
        """Octile distance heuristic."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + 0.414 * min(dx, dy)

    def key(self, s):
        g_val = self.g.get(s, INF)
        rhs_val = self.rhs.get(s, INF)
        min_val = min(g_val, rhs_val)
        return (min_val + self.h(self.start, s) + self.km, min_val)

    def cost(self, a, b):
        """Move cost from a to b."""
        bx, by = b
        cell_cost = float(self.grid[by, bx])
        
        if cell_cost <= 0.0 or math.isinf(cell_cost):
            return INF

        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dist = 1.414 if (dx + dy > 1) else 1.0
        
        return dist * cell_cost

    def successors(self, s):
        sx, sy = s
        for dx, dy, _ in MOVES:
            nx, ny = sx + dx, sy + dy
            ns = (nx, ny)
            if not self.in_bounds(ns):
                continue
            
            if not self.traversable(ns):
                continue
            
            # Prevent corner cutting
            if dx != 0 and dy != 0:
                if not (self.traversable((sx + dx, sy)) and self.traversable((sx, sy + dy))):
                    continue
            yield ns

    def predecessors(self, s):
        return self.successors(s)

    def update_vertex(self, u):
        """Update vertex and add to priority queue."""
        # Remove from queue if present
        if u in self.U_set:
            self.U_set.discard(u)
            # Note: We don't actually remove from heap, just mark as stale
        
        if u != self.goal:
            best = INF
            for s in self.successors(u):
                c = self.cost(u, s)
                if c != INF:
                    val = self.g.get(s, INF) + c
                    if val < best:
                        best = val
            self.rhs[u] = best
        
        # Only add to queue if inconsistent
        g_val = self.g.get(u, INF)
        rhs_val = self.rhs.get(u, INF)
        
        if abs(g_val - rhs_val) > 1e-5:
            key = self.key(u)
            heapq.heappush(self.U, (key, u))
            self.U_set.add(u)

    def compute(self, max_steps=None):
        """Compute shortest path with early termination."""
        steps = 0
        
        while self.U:
            if max_steps and steps >= max_steps:
                break
            
            # Pop from queue
            k_old, u = heapq.heappop(self.U)
            
            # Skip stale entries
            if u not in self.U_set:
                continue
            
            self.U_set.discard(u)
            
            k_new = self.key(u)

            # Reinsert if key changed
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
                self.U_set.add(u)
                continue
            
            g_val = self.g.get(u, INF)
            rhs_val = self.rhs.get(u, INF)
            
            # Skip if already consistent
            if abs(g_val - rhs_val) < 1e-5:
                continue

            steps += 1
            
            if g_val > rhs_val:
                # Overconsistent - make consistent
                self.g[u] = rhs_val
                for p in self.predecessors(u):
                    self.update_vertex(p)
            else:
                # Underconsistent
                self.g[u] = INF
                self.update_vertex(u)
                for p in self.predecessors(u):
                    self.update_vertex(p)
            
            # Early termination: if start is consistent and has lower key than top of queue
            start_g = self.g.get(self.start, INF)
            start_rhs = self.rhs.get(self.start, INF)
            
            if abs(start_g - start_rhs) < 1e-5:
                # Start is consistent
                if not self.U or not self.U_set:
                    break
                # Check if we can terminate
                top_key = None
                for k, v in self.U:
                    if v in self.U_set:
                        top_key = k
                        break
                
                if top_key is None or top_key >= self.key(self.start):
                    break

        return steps

    def update_cell_cost(self, cell, new_cost):
        """Update a cell's cost and propagate changes."""
        x, y = cell
        self.grid[y, x] = float(new_cost)
        
        # Update affected vertices
        self.update_vertex(cell)
        for p in self.predecessors(cell):
            self.update_vertex(p)

    def move_start(self, new_start):
        """Move the start position."""
        if new_start == self.start:
            return
        self.km += self.h(self.last, new_start)
        self.last = new_start
        self.start = new_start

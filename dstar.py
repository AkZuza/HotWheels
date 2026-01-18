import heapq
import math

# Use math.inf to be safe
INF = math.inf

# 8-neighborhood moves (dx, dy, cost)
MOVES = [
    (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
    (1, 1, 1.414), (-1, 1, 1.414),
    (1, -1, 1.414), (-1, -1, 1.414),
]

class DStarLite:
    def __init__(self, start, goal, grid):
        self.grid = grid  # [H,W]
        self.start = start
        self.goal = goal
        self.last = start
        self.km = 0.0
        
        self.g = {}
        self.rhs = {}
        self.U = []

        H, W = self.grid.shape
        for y in range(H):
            for x in range(W):
                self.g[(x, y)] = INF
                self.rhs[(x, y)] = INF

        self.rhs[self.goal] = 0.0
        heapq.heappush(self.U, (self.key(self.goal), self.goal))

    def in_bounds(self, s):
        x, y = s
        H, W = self.grid.shape
        return 0 <= x < W and 0 <= y < H

    def traversable(self, s):
        x, y = s
        c = float(self.grid[y, x])
        return (c > 0.0) and (not math.isinf(c))

    def h(self, a, b):
        # OCTILE DISTANCE (Perfect for 8-grid movement)
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + 0.414 * min(dx, dy)

    def key(self, s):
        min_g_rhs = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (min_g_rhs + self.h(self.start, s) + self.km, min_g_rhs)

    def cost(self, a, b):
        """Move cost a->b."""
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
            if not self.in_bounds(ns): continue
            
            if not self.traversable(ns): continue
            
            # PREVENT CORNER CUTTING (Safety)
            if dx != 0 and dy != 0:
                if not (self.traversable((sx + dx, sy)) and self.traversable((sx, sy + dy))):
                    continue
            yield ns

    def predecessors(self, s):
        return self.successors(s)

    def update_vertex(self, u):
        if u != self.goal:
            best = INF
            for s in self.successors(u):
                c = self.cost(u, s)
                if c != INF:
                    val = self.g.get(s, INF) + c
                    if val < best: best = val
            self.rhs[u] = best
        
        heapq.heappush(self.U, (self.key(u), u))

    def compute(self, max_steps=None):
        steps = 0
        while self.U:
            if max_steps and steps > max_steps: break
            
            k_old, u = heapq.heappop(self.U)
            k_new = self.key(u)

            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
                continue
                
            if self.rhs.get(u, INF) == self.g.get(u, INF) and k_old == k_new:
                if u == self.start: break
                continue

            steps += 1
            if self.g.get(u, INF) > self.rhs.get(u, INF):
                self.g[u] = self.rhs[u]
                for p in self.predecessors(u): self.update_vertex(p)
            else:
                self.g[u] = INF
                self.update_vertex(u)
                for p in self.predecessors(u): self.update_vertex(p)
                
            if self.rhs.get(self.start, INF) == self.g.get(self.start, INF):
                if self.U and self.U[0][0] >= self.key(self.start):
                    break

        return steps

    def update_cell_cost(self, cell, new_cost):
        x, y = cell
        self.grid[y, x] = float(new_cost)
        self.update_vertex(cell)
        for p in self.predecessors(cell): self.update_vertex(p)

    def move_start(self, new_start):
        if new_start == self.start: return
        self.km += self.h(self.last, new_start)
        self.last = new_start
        self.start = new_start
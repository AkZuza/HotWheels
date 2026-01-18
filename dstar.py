import heapq
import math

INF = float('inf')

MOVES = [
    (1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1),      # cardinal
    (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)  # diagonals
]


class DStarLite:
    def __init__(self, start, goal, grid):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.km = 0

        self.g = {}
        self.rhs = {}
        self.U = []

        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                self.g[(x,y)] = INF
                self.rhs[(x,y)] = INF

        self.rhs[self.goal] = 0
        heapq.heappush(self.U, (self.key(self.goal), self.goal))

    def h(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def key(self, s):
        return (min(self.g[s], self.rhs[s]) + self.h(self.start, s),
                min(self.g[s], self.rhs[s]))


    def successors(self, s):
        for dx, dy, base in MOVES:
            x, y = s[0] + dx, s[1] + dy
            if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                if self.grid[x, y] > 0:
                    yield (x, y), base * self.grid[x, y]


    def update_vertex(self, u):
        if u != self.goal:
            succ = list(self.successors(u))
            if not succ:
                new_rhs = INF
            else:
                new_rhs = min(self.g[s] + c for s, c in succ)
        else:
            new_rhs = self.rhs[u]

        if new_rhs != self.rhs[u]:
            self.rhs[u] = new_rhs
            heapq.heappush(self.U, (self.key(u), u))


    def compute(self):
        while self.U:
            k_old, u = heapq.heappop(self.U)

            if k_old > self.key(u):
                heapq.heappush(self.U, (self.key(u), u))

            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s, _ in self.successors(u):
                    self.update_vertex(s)

            else:
                self.g[u] = INF
                self.update_vertex(u)
                for s, _ in self.successors(u):
                    self.update_vertex(s)

            if not (self.U and (self.U[0][0] < self.key(self.start) or self.rhs[self.start] != self.g[self.start])):
                break


    def update_obstacle(self, cell):
        self.grid[cell[0], cell[1]] = 0
        self.update_vertex(cell)


    def next(self):
        self.compute()

        best, best_cost = None, INF
        for s, c in self.successors(self.start):
            cost = self.g[s] + c
            if cost < best_cost:
                best, best_cost = s, cost

        if best is None:
            return None

        self.start = best
        return best


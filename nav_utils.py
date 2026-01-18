import os
import pickle
import numpy as np
import heapq
import math
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from map_creator import MapPixel, FuncID
from dstar import DStarLite

@dataclass
class TeleportInfo:
    position: Tuple[int, int]
    func_type: int
    destinations: List[str]
    cost: int
    dept_id: int
    floor: int

@dataclass
class MapInfo:
    map_id: str
    width: int
    height: int
    map_data: np.ndarray
    teleports: Dict[str, List[TeleportInfo]]

class NavigationGraph:
    def __init__(self):
        self.maps: Dict[str, MapInfo] = {}
        self.graph: Dict[str, Set[str]] = {}

    def load_maps(self, maps_directory: str):
        if not os.path.exists(maps_directory):
            return
        for f in os.listdir(maps_directory):
            if f.endswith(".bin"):
                self._load_map(os.path.join(maps_directory, f))
        self._build_graph()

    def _load_map(self, filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            mid = data.get("map_id", os.path.basename(filepath)[:-4])
            w, h = data["width"], data["height"]

            pixels = np.array([[MapPixel.from_tuple(p) for p in row] for row in data["map_data"]], dtype=object)
            self.maps[mid] = MapInfo(mid, w, h, pixels, self._find_teleports(pixels, w, h))
        except:
            pass

    def _find_teleports(self, pixels, w, h):
        tps = {}
        for y in range(h):
            for x in range(w):
                p = pixels[y][x]
                if p.func_id in [FuncID.DOOR, FuncID.ELEVATOR, FuncID.RAMP, FuncID.STAIR] and p.identifier:
                    info = TeleportInfo((x, y), p.func_id, p.identifier, p.cost, p.dept_id, p.floor)
                    for d in p.identifier:
                        tps.setdefault(d, []).append(info)
        return tps

    def _build_graph(self):
        self.graph = {m: set() for m in self.maps}
        for m, info in self.maps.items():
            for d in info.teleports:
                if d in self.maps:
                    self.graph[m].add(d)
                    self.graph[d].add(m)

    def find_map_path(self, start, end):
        if start == end:
            return [start]
        q = [(start, [start])]
        vis = {start}
        while q:
            u, path = q.pop(0)
            for v in self.graph.get(u, []):
                if v not in vis:
                    if v == end:
                        return path + [v]
                    vis.add(v)
                    q.append((v, path + [v]))
        return None

    def get_teleport_to(self, curr, target):
        t = self.maps[curr].teleports.get(target, [])
        return t[0].position if t else None

    def get_entry_teleport_from(self, current_map: str, next_map: str, goal_hint=None):
        next_info = self.maps.get(next_map)
        if not next_info:
            return None
        candidates = next_info.teleports.get(current_map, [])
        if not candidates:
            return None
        if goal_hint is None:
            return candidates[0].position
        return min(
            candidates,
            key=lambda tp: abs(tp.position[0] - goal_hint[0]) + abs(tp.position[1] - goal_hint[1])
        ).position

class DynamicLocalPlanner:
    def __init__(self, map_info, start, goal, initial_visible_cells=None):
        self.map = map_info
        self.start = start
        self.goal = goal

        # 1. Build Static Costs
        self.gt_costs = self._build_ground_truth_cost_grid()
        
        # 2. Add SAFETY Buffer (Anti-Wall Hugging)
        self.clearance_cost = self._build_clearance_cost(self.gt_costs)

        # 3. Initialize Perception
        self.known_costs = np.zeros_like(self.gt_costs, dtype=np.float32)
        H, W = self.known_costs.shape
        for y in range(H):
            for x in range(W):
                base = self.gt_costs[y, x]
                if math.isinf(base): self.known_costs[y, x] = math.inf
                else: self.known_costs[y, x] = base + self.clearance_cost[y, x]

        self._sanitize_start_goal()

        # 4. Initialize D* Lite
        self.planner = DStarLite(self.start, self.goal, self.known_costs)
        
        # 5. Initial Compute - Run until FULL consistency to get perfect static path
        # FIX: Using math.inf here instead of undefined INF
        print("Computing initial static path...")
        self.planner.compute(max_steps=math.inf)
        
        # 6. Strict Path Cache
        self.active_path = [] 
        self._extract_full_path()

        if initial_visible_cells:
            self.sense_and_update(initial_visible_cells)

    def _build_ground_truth_cost_grid(self):
        w, h = self.map.width, self.map.height
        grid = np.ones((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                p = self.map.map_data[y][x]
                if p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999:
                    grid[y, x] = math.inf
                elif p.func_id == FuncID.EMPTY:
                    grid[y, x] = 10.0
                else:
                    grid[y, x] = max(1.0, float(p.cost))
        return grid

    def _build_clearance_cost(self, gt):
        """Creates a repulsion field near walls to prevent hugging."""
        H, W = gt.shape
        dist = np.full((H, W), 9999, dtype=np.int32)
        q = []

        # Seed BFS with obstacles
        for y in range(H):
            for x in range(W):
                if math.isinf(float(gt[y, x])):
                    dist[y, x] = 0
                    q.append((x, y))

        head = 0
        while head < len(q):
            x, y = q[head]
            head += 1
            if dist[y, x] > 8: continue 

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<W and 0<=ny<H:
                    if dist[ny, nx] > dist[y, x] + 1:
                        dist[ny, nx] = dist[y, x] + 1
                        q.append((nx, ny))

        penalty = np.zeros((H, W), dtype=np.float32)
        SAFE_DIST = 4 
        MAX_PENALTY = 200.0
        
        for y in range(H):
            for x in range(W):
                d = dist[y, x]
                if d < SAFE_DIST:
                    penalty[y, x] = MAX_PENALTY * ((SAFE_DIST - d) / SAFE_DIST) ** 2
        return penalty

    def _sanitize_start_goal(self):
        if math.isinf(self.known_costs[self.start[1], self.start[0]]):
            print("Warning: Start inside obstacle/wall buffer!")
        if math.isinf(self.known_costs[self.goal[1], self.goal[0]]):
            print("Warning: Goal inside obstacle/wall buffer!")

    def sense_and_update(self, visible_cells):
        map_changed = False
        
        for (x, y) in visible_cells:
            p = self.map.map_data[y][x]
            
            real_cost = float(self.gt_costs[y, x])
            if p.func_id == FuncID.OBSTACLE: real_cost = math.inf
            
            if not math.isinf(real_cost):
                real_cost += self.clearance_cost[y, x]

            current_belief = self.known_costs[y, x]
            
            if abs(real_cost - current_belief) > 0.1 or (math.isinf(real_cost) != math.isinf(current_belief)):
                self.known_costs[y, x] = real_cost
                self.planner.update_cell_cost((x, y), real_cost)
                map_changed = True

        if map_changed:
            self.planner.compute(max_steps=50000)

    def _extract_full_path(self):
        """Standard gradient descent to extract full path from D* g-values."""
        path = []
        curr = self.start
        
        for _ in range(500):
            path.append(curr)
            if curr == self.goal: break
            
            best = None
            best_cost = float('inf')
            
            for s in self.planner.successors(curr):
                c = self.planner.cost(curr, s)
                if c == float('inf'): continue
                
                score = self.planner.g.get(s, float('inf')) + c
                
                if score < best_cost:
                    best_cost = score
                    best = s
            
            if best and best != curr:
                curr = best
            else:
                break
                
        self.active_path = path

    def _is_path_blocked(self):
        if not self.active_path: return True
        for x, y in self.active_path[1:]:
            if math.isinf(self.known_costs[y, x]):
                return True
        return False

    def step(self, compute_budget=None):
        if not self.active_path or self._is_path_blocked():
            print("Path blocked or empty! Rerouting...")
            self.planner.compute(max_steps=50000)
            self._extract_full_path()
            
            if not self.active_path or len(self.active_path) < 2:
                # Try simple neighbor move if totally stuck
                return None

        if self.active_path[0] != self.start:
            if self.start in self.active_path:
                idx = self.active_path.index(self.start)
                self.active_path = self.active_path[idx:]
            else:
                self._extract_full_path()
        
        if len(self.active_path) > 1:
            next_node = self.active_path[1]
            self.planner.move_start(next_node)
            self.start = next_node
            self.active_path.pop(0)
            return next_node
            
        return self.start

def navigate_multi_map(nav_graph, start_map, start_pos, end_map, end_pos):
    result = {"success": False, "map_sequence": [], "checkpoints": {}, "paths": {}}
    seq = nav_graph.find_map_path(start_map, end_map)
    if not seq: return result

    result["success"] = True
    result["map_sequence"] = seq

    cur_pos = start_pos
    for i, m in enumerate(seq):
        is_last = (i == len(seq) - 1)
        target = end_pos if is_last else nav_graph.get_teleport_to(m, seq[i + 1])
        if target is None:
            result["success"] = False
            return result
        result["checkpoints"][m] = {"start": cur_pos, "goal": target}
        if not is_last:
            next_start = nav_graph.get_entry_teleport_from(m, seq[i + 1], goal_hint=target)
            cur_pos = next_start if next_start else target

    return result
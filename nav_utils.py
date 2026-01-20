import os
import pickle
import numpy as np
import heapq
import math
from typing import Dict, List, Tuple, Set, Optional
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
        # Cache for static paths between teleports
        self.static_path_cache: Dict[Tuple[str, Tuple[int,int], Tuple[int,int]], List[Tuple[int,int]]] = {}

    def load_maps(self, maps_directory: str, precompute_paths=False):
        if not os.path.exists(maps_directory):
            return
        for f in os.listdir(maps_directory):
            if f.endswith(".bin"):
                self._load_map(os.path.join(maps_directory, f))
        self._build_graph()
        # Optional: Pre-compute static paths between all teleport pairs
        if precompute_paths:
            self._precompute_static_paths()

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

    def _precompute_static_paths(self):
        """Pre-compute static A* paths between all teleport pairs on each map."""
        print("Pre-computing static paths...")
        for map_id, map_info in self.maps.items():
            # Get all teleport positions on this map
            teleport_positions = set()
            for teleport_list in map_info.teleports.values():
                for tp in teleport_list:
                    teleport_positions.add(tp.position)
            
            # Compute paths between all pairs
            positions = list(teleport_positions)
            for i, start in enumerate(positions):
                for goal in positions[i+1:]:
                    if start == goal:
                        continue
                    # Use fast A* for static pre-computation
                    path = self._compute_astar_path(map_info, start, goal)
                    if path:
                        self.static_path_cache[(map_id, start, goal)] = path
                        self.static_path_cache[(map_id, goal, start)] = list(reversed(path))
        
        print(f"Cached {len(self.static_path_cache)} static paths")

    def _compute_astar_path(self, map_info, start, goal):
        """Ultra-fast A* for static path computation with clearance awareness."""
        def heuristic(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return max(dx, dy) + 0.414 * min(dx, dy)
        
        # Build clearance cost for this map if not already done
        cache_key = f"{map_info.map_id}_clearance"
        if not hasattr(self, '_clearance_cache'):
            self._clearance_cache = {}
        
        if cache_key not in self._clearance_cache:
            # Build ground truth costs
            w, h = map_info.width, map_info.height
            gt = np.ones((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    p = map_info.map_data[y][x]
                    if p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999:
                        gt[y, x] = math.inf
                    elif p.func_id == FuncID.EMPTY:
                        gt[y, x] = 10.0
                    else:
                        gt[y, x] = max(1.0, float(p.cost))
            
            # Build clearance costs
            clearance = self._build_clearance_for_astar(gt, map_info)
            self._clearance_cache[cache_key] = (gt, clearance)
        
        gt, clearance = self._clearance_cache[cache_key]
        
        def get_cost(pos):
            x, y = pos
            if not (0 <= x < map_info.width and 0 <= y < map_info.height):
                return math.inf
            base_cost = float(gt[y, x])
            if math.isinf(base_cost):
                return math.inf
            return base_cost + clearance[y, x]
        
        # Quick check if start/goal are valid
        if math.isinf(get_cost(start)) or math.isinf(get_cost(goal)):
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()
        
        # Limit iterations to prevent infinite loops
        max_iter = min(10000, map_info.width * map_info.height)
        iterations = 0
        
        while open_set and iterations < max_iter:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current in closed:
                continue
            closed.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            cx, cy = current
            curr_g = g_score[current]
            
            # 8-directional movement
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                if neighbor in closed:
                    continue
                
                cost = get_cost(neighbor)
                if math.isinf(cost):
                    continue
                
                # Prevent corner cutting for diagonals
                if dx != 0 and dy != 0:
                    if math.isinf(get_cost((cx + dx, cy))) or math.isinf(get_cost((cx, cy + dy))):
                        continue
                
                move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                tentative_g = curr_g + move_cost * cost
                
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    # Use weighted heuristic for faster search
                    f_score = tentative_g + 1.0 * heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None
    
    def _build_clearance_for_astar(self, gt, map_info):
        """Build clearance cost field for A* path planning."""
        H, W = gt.shape
        dist = np.full((H, W), 9999, dtype=np.int32)
        q = []

        # BFS from obstacles
        for y in range(H):
            for x in range(W):
                if math.isinf(float(gt[y, x])):
                    dist[y, x] = 0
                    q.append((x, y))

        head = 0
        MAX_DIST = 5
        while head < len(q):
            x, y = q[head]
            head += 1
            if dist[y, x] >= MAX_DIST:
                continue

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<W and 0<=ny<H:
                    new_dist = dist[y, x] + 1
                    if dist[ny, nx] > new_dist:
                        dist[ny, nx] = new_dist
                        q.append((nx, ny))

        penalty = np.zeros((H, W), dtype=np.float32)
        SAFE_DIST = 3
        MAX_PENALTY = 500.0
        EMPTY_PENALTY = 50.0
        
        for y in range(H):
            for x in range(W):
                if math.isinf(gt[y, x]):
                    continue
                
                d = dist[y, x]
                if d < SAFE_DIST:
                    clearance_penalty = MAX_PENALTY * math.exp(-d * 0.7)
                else:
                    clearance_penalty = 0.0
                
                cell = map_info.map_data[y][x]
                type_penalty = 0.0
                if cell.func_id == FuncID.EMPTY:
                    type_penalty = EMPTY_PENALTY
                
                penalty[y, x] = clearance_penalty + type_penalty
        
        return penalty

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
    
    def get_static_path(self, map_id: str, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        """Get static path - compute on-demand if not cached."""
        cache_key = (map_id, start, goal)
        
        # Check if already cached
        if cache_key in self.static_path_cache:
            return self.static_path_cache[cache_key]
        
        # Compute on-demand
        map_info = self.maps.get(map_id)
        if not map_info:
            return None
        
        path = self._compute_astar_path(map_info, start, goal)
        if path:
            # Cache both directions
            self.static_path_cache[cache_key] = path
            self.static_path_cache[(map_id, goal, start)] = list(reversed(path))
            return path
        
        return None


class DynamicLocalPlanner:
    """Optimized planner using static paths and D* Lite only for dynamic obstacles."""
    
    def __init__(self, map_info, start, goal, nav_graph=None, initial_visible_cells=None):
        self.map = map_info
        self.start = start
        self.goal = goal
        self.nav_graph = nav_graph

        # Build cost grids
        self.gt_costs = self._build_ground_truth_cost_grid()
        self.clearance_cost = self._build_clearance_cost(self.gt_costs)

        # Initialize known costs (optimistic - assume all unknown is walkable)
        self.known_costs = np.ones_like(self.gt_costs, dtype=np.float32)
        
        # Only mark definitely known obstacles
        H, W = self.known_costs.shape
        for y in range(H):
            for x in range(W):
                if math.isinf(self.gt_costs[y, x]):
                    self.known_costs[y, x] = math.inf

        self._sanitize_start_goal()

        # Try to use static path first (computed on-demand)
        self.using_static_path = False
        self.static_path = None
        
        if nav_graph:
            self.static_path = nav_graph.get_static_path(map_info.map_id, start, goal)
            if self.static_path:
                print(f"Using cached/computed static path ({len(self.static_path)} nodes)")
                self.using_static_path = True
                self.active_path = list(self.static_path)
                self.planner = None
                
                # Sense initial area
                if initial_visible_cells:
                    self.sense_and_update(initial_visible_cells)
                return

        # Fall back to D* Lite
        print("Using D* Lite for dynamic planning...")
        self.planner = DStarLite(self.start, self.goal, self.known_costs)
        
        # Quick initial compute with limited budget
        self.planner.compute(max_steps=10000)
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
        """
        Create clearance cost field with:
        1. Heavy penalties near obstacles (safety buffer)
        2. Prefer WALKABLE cells over EMPTY cells
        """
        H, W = gt.shape
        dist = np.full((H, W), 9999, dtype=np.int32)
        q = []

        # BFS from all obstacles to compute distance field
        for y in range(H):
            for x in range(W):
                if math.isinf(float(gt[y, x])):
                    dist[y, x] = 0
                    q.append((x, y))

        head = 0
        MAX_DIST = 5  # Compute clearance up to 5 cells away
        while head < len(q):
            x, y = q[head]
            head += 1
            if dist[y, x] >= MAX_DIST:
                continue

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<W and 0<=ny<H:
                    new_dist = dist[y, x] + 1
                    if dist[ny, nx] > new_dist:
                        dist[ny, nx] = new_dist
                        q.append((nx, ny))

        penalty = np.zeros((H, W), dtype=np.float32)
        
        # Clearance penalty parameters
        SAFE_DIST = 3  # Safe distance from obstacles (cells)
        MAX_PENALTY = 500.0  # Very high penalty for being close to walls
        
        # Cell type preference penalty
        EMPTY_PENALTY = 50.0  # Additional cost for EMPTY cells vs WALKABLE
        
        for y in range(H):
            for x in range(W):
                # Skip obstacles
                if math.isinf(gt[y, x]):
                    continue
                
                # 1. Clearance penalty (exponential decay from obstacles)
                d = dist[y, x]
                if d < SAFE_DIST:
                    # Exponential penalty - very high when close, drops off quickly
                    clearance_penalty = MAX_PENALTY * math.exp(-d * 0.7)
                else:
                    clearance_penalty = 0.0
                
                # 2. Cell type preference (prefer WALKABLE over EMPTY)
                cell = self.map.map_data[y][x]
                type_penalty = 0.0
                if cell.func_id == FuncID.EMPTY:
                    type_penalty = EMPTY_PENALTY
                # WALKABLE gets 0 additional penalty (preferred)
                
                penalty[y, x] = clearance_penalty + type_penalty
        
        return penalty

    def _sanitize_start_goal(self):
        if math.isinf(self.known_costs[self.start[1], self.start[0]]):
            print("Warning: Start inside obstacle!")
        if math.isinf(self.known_costs[self.goal[1], self.goal[0]]):
            print("Warning: Goal inside obstacle!")

    def sense_and_update(self, visible_cells):
        """Only update if we detect NEW obstacles that block our path."""
        if self.using_static_path and self.static_path:
            # Check if any visible obstacles block the static path
            path_blocked = False
            for (x, y) in visible_cells:
                if (x, y) in self.static_path:
                    p = self.map.map_data[y][x]
                    real_cost = float(self.gt_costs[y, x])
                    if p.func_id == FuncID.OBSTACLE or math.isinf(real_cost):
                        path_blocked = True
                        break
            
            if path_blocked:
                print("Static path blocked! Switching to D* Lite...")
                self.using_static_path = False
                # Initialize D* Lite
                self.planner = DStarLite(self.start, self.goal, self.known_costs)
                self.planner.compute(max_steps=10000)
                self._extract_full_path()
                # Fall through to normal update
            else:
                return  # Static path still good

        # D* Lite update
        if not self.planner:
            return
            
        map_changed = False
        
        for (x, y) in visible_cells:
            p = self.map.map_data[y][x]
            
            real_cost = float(self.gt_costs[y, x])
            if p.func_id == FuncID.OBSTACLE:
                real_cost = math.inf
            
            if not math.isinf(real_cost):
                real_cost += self.clearance_cost[y, x]

            current_belief = self.known_costs[y, x]
            
            # Only update if significantly different
            if abs(real_cost - current_belief) > 0.1 or (math.isinf(real_cost) != math.isinf(current_belief)):
                self.known_costs[y, x] = real_cost
                self.planner.update_cell_cost((x, y), real_cost)
                map_changed = True

        if map_changed:
            # Limited recompute
            self.planner.compute(max_steps=5000)  # Reduced from 50000

    def _extract_full_path(self):
        """Extract path from D* g-values."""
        if not self.planner:
            return
            
        path = []
        curr = self.start
        visited = set()
        
        for _ in range(500):
            if curr in visited:
                break
            path.append(curr)
            visited.add(curr)
            
            if curr == self.goal:
                break
            
            best = None
            best_cost = float('inf')
            
            for s in self.planner.successors(curr):
                if s in visited:
                    continue
                c = self.planner.cost(curr, s)
                if c == float('inf'):
                    continue
                
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
        if not self.active_path:
            return True
        for x, y in self.active_path[1:]:
            if math.isinf(self.known_costs[y, x]):
                return True
        return False

    def step(self, compute_budget=None):
        """Take one step along the path."""
        if self.using_static_path:
            # Simple static path following
            if not self.active_path or len(self.active_path) < 2:
                return self.start
            
            # Update to current position
            if self.start in self.active_path:
                idx = self.active_path.index(self.start)
                self.active_path = self.active_path[idx:]
            
            if len(self.active_path) > 1:
                next_node = self.active_path[1]
                self.start = next_node
                self.active_path.pop(0)
                return next_node
            
            return self.start
        
        # D* Lite dynamic following
        if not self.planner:
            return self.start
            
        if not self.active_path or self._is_path_blocked():
            print("Path blocked! Rerouting...")
            self.planner.compute(max_steps=5000)
            self._extract_full_path()
            
            if not self.active_path or len(self.active_path) < 2:
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
    """Optimized multi-map navigation - pre-determine entire route."""
    result = {"success": False, "map_sequence": [], "checkpoints": {}, "paths": {}}
    
    # 1. Find map sequence (this is static)
    seq = nav_graph.find_map_path(start_map, end_map)
    if not seq:
        return result

    result["success"] = True
    result["map_sequence"] = seq

    # 2. Determine all checkpoints upfront
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

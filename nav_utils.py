import os
import pickle
import numpy as np
import heapq
import math
import traceback
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt
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
        self.static_path_cache: Dict[Tuple[str, Tuple[int,int], Tuple[int,int]], List[Tuple[int,int]]] = {}

    def load_maps(self, maps_directory: str, precompute_paths=False):
        if not os.path.exists(maps_directory):
            return
        for f in os.listdir(maps_directory):
            if f.endswith(".bin"):
                self._load_map(os.path.join(maps_directory, f))
        self._build_graph()
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
        print("Pre-computing static paths...")
        for map_id, map_info in self.maps.items():
            teleport_positions = set()
            for teleport_list in map_info.teleports.values():
                for tp in teleport_list:
                    teleport_positions.add(tp.position)
            
            positions = list(teleport_positions)
            for i, start in enumerate(positions):
                for goal in positions[i+1:]:
                    if start == goal:
                        continue
                    path = self._compute_astar_path(map_info, start, goal)
                    if path:
                        self.static_path_cache[(map_id, start, goal)] = path
                        self.static_path_cache[(map_id, goal, start)] = list(reversed(path))
        print(f"Cached {len(self.static_path_cache)} static paths")

    def _compute_astar_path(self, map_info, start, goal):
        def heuristic(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return max(dx, dy) + 0.414 * min(dx, dy)
        
        w, h = map_info.width, map_info.height
        if not (0 <= start[0] < w and 0 <= start[1] < h): return None
        if not (0 <= goal[0] < w and 0 <= goal[1] < h): return None

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in closed: continue
            closed.add(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path))
            
            cx, cy = current
            curr_g = g_score[current]
            
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                if not (0 <= nx < w and 0 <= ny < h): continue
                if neighbor in closed: continue

                p = map_info.map_data[ny][nx]
                if p.func_id == FuncID.OBSTACLE or p.cost >= 999: continue

                if dx != 0 and dy != 0:
                    p1 = map_info.map_data[cy][nx]
                    p2 = map_info.map_data[ny][cx]
                    if (p1.func_id == FuncID.OBSTACLE) or (p2.func_id == FuncID.OBSTACLE):
                        continue

                move_cost = 1.414 if (dx!=0 and dy!=0) else 1.0
                tentative_g = curr_g + move_cost
                
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    def find_map_path(self, start, end):
        if start == end: return [start]
        q = [(start, [start])]
        vis = {start}
        while q:
            u, path = q.pop(0)
            for v in self.graph.get(u, []):
                if v not in vis:
                    if v == end: return path + [v]
                    vis.add(v)
                    q.append((v, path + [v]))
        return None

    def get_teleport_to(self, curr, target):
        t = self.maps[curr].teleports.get(target, [])
        return t[0].position if t else None

    def get_entry_teleport_from(self, current_map: str, next_map: str, goal_hint=None):
        next_info = self.maps.get(next_map)
        if not next_info: return None
        candidates = next_info.teleports.get(current_map, [])
        if not candidates: return None
        if goal_hint is None: return candidates[0].position
        return min(candidates, key=lambda tp: abs(tp.position[0]-goal_hint[0])+abs(tp.position[1]-goal_hint[1])).position


class DynamicLocalPlanner:
    def __init__(self, map_info, start, goal, nav_graph=None, initial_visible_cells=None):
        self.map = map_info
        self.start = start
        self.goal = goal
        self.nav_graph = nav_graph
        
        # --- INITIALIZE PROBABILISTIC SAFETY MAP ---
        # 1. Compute Signed Distance Field (SDF) for Safety Coefficients
        self.sdf_map = self._compute_sdf_map()
        
        # 2. Build cost grid using SDF + Distance-Dependent Probability
        self.gt_costs = self._build_probabilistic_cost_grid()
        
        # Initialize belief
        self.known_costs = np.ones_like(self.gt_costs, dtype=np.float32)
        H, W = self.known_costs.shape
        for y in range(H):
            for x in range(W):
                if math.isinf(self.gt_costs[y, x]):
                    self.known_costs[y, x] = math.inf

        self.planner = DStarLite(self.start, self.goal, self.known_costs)
        
        try:
            self.planner.compute(max_steps=50000)
            self.active_path = []
            self._extract_full_path()
        except Exception:
            traceback.print_exc()
            self.active_path = []

        self.reroute_count = 0

        if initial_visible_cells:
            self.sense_and_update(initial_visible_cells)

    def _compute_sdf_map(self):
        """
        Computes the Signed Distance Field (SDF).
        Result: Each cell contains the exact distance (in pixels) to the nearest static obstacle.
        """
        w, h = self.map.width, self.map.height
        
        # Binary map: 1 = Space, 0 = Wall
        binary_map = np.ones((h, w), dtype=int)
        
        for y in range(h):
            for x in range(w):
                p = self.map.map_data[y][x]
                if p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999:
                    binary_map[y, x] = 0
                    
        # Compute Euclidean Distance Transform
        sdf = distance_transform_edt(binary_map)
        return sdf

    def _build_probabilistic_cost_grid(self):
        """
        Builds a cost grid with 'Turn Safety' and 'Diagonal Safety'.
        
        CHANGES:
        1. Increased COLLISION_RADIUS (3.0): Forces wider turns around corners.
        2. High PREFERRED_CLEARANCE (20.0): Keeps robot centered.
        3. Inverse Square Penalty: Prevents wall skirting (diagonal or straight).
        """
        w, h = self.map.width, self.map.height
        grid = np.ones((h, w), dtype=np.float32)
        gx, gy = self.goal
        
        # --- SAFETY PARAMETERS ---
        # 1. HARD LIMIT (Turn Safety)
        # We treat 3.0 pixels from the wall as a "Physical Wall".
        # This prevents the robot from clipping corners during diagonal moves.
        COLLISION_RADIUS = 3.0  
        
        # 2. COMFORT ZONE (Cruising Safety)
        # We prefer to be at least 20px from walls when cruising.
        PREFERRED_CLEARANCE = 20.0
        
        # 3. DOCKING (Goal Approach)
        # When within 15px of the goal, we relax the rules to allow parking.
        DOCKING_DISTANCE = 15.0
        
        # 4. PENALTY WEIGHTS
        MAX_WALL_PENALTY = 1000.0 # Extremely high cost to prevent skirting
        
        for y in range(h):
            for x in range(w):
                p = self.map.map_data[y][x]
                
                # --- A. HARD OBSTACLES ---
                if p.func_id == FuncID.OBSTACLE or float(p.cost) >= 999:
                    grid[y, x] = math.inf
                    continue
                
                # --- B. DISTANCE CALCULATIONS ---
                dist_to_goal = math.hypot(x - gx, y - gy)
                dist_wall = self.sdf_map[y, x]
                
                # --- C. MODE DETERMINATION ---
                is_docking = dist_to_goal <= DOCKING_DISTANCE
                
                # --- D. SAFETY COST CALCULATION ---
                safety_cost = 0.0
                
                if dist_wall <= COLLISION_RADIUS:
                    # Physical Collision Zone (Expanded for turn safety)
                    safety_cost = math.inf
                elif dist_wall < PREFERRED_CLEARANCE:
                    # We are in the "Buffer Zone"
                    
                    # Normalize distance (0.0 = Safe Line, 1.0 = Wall Boundary)
                    unsafe_factor = (PREFERRED_CLEARANCE - dist_wall) / (PREFERRED_CLEARANCE - COLLISION_RADIUS)
                    
                    if is_docking:
                        # DOCKING MODE: Relaxed
                        # Allow getting closer (linear penalty up to 15.0 cost)
                        # This lets it "squeeze" into the final spot
                        safety_cost = 15.0 * unsafe_factor
                    else:
                        # CRUISING MODE: Extreme Safety
                        # Use a Power of 4 curve.
                        # This creates a "vertical wall" of cost.
                        # 4px from wall -> Cost ~600
                        # 10px from wall -> Cost ~50
                        # 18px from wall -> Cost ~0
                        # Result: Robot stays >10px away unless absolutely forced.
                        safety_cost = MAX_WALL_PENALTY * (unsafe_factor ** 10)

                # --- E. TERRAIN COST ---
                base_cost = 1.0
                if p.func_id == FuncID.EMPTY:
                    if is_docking:
                        base_cost = 2.0
                    else:
                        # High penalty for empty space to force pavement/safe-zone usage
                        base_cost = 20.0 
                
                grid[y, x] = base_cost + safety_cost
                
                # Goal is always free
                if (x, y) == self.goal:
                    grid[y, x] = 0.1
                    
        return grid

    def sense_and_update(self, visible_cells):
        if not self.planner: return
        map_changed = False
        obstacle_detected = False
        path_invalidated = False
        
        try:
            # We collect updates to apply them in batch or carefully
            updates = {}

            for (x, y) in visible_cells:
                pixel = self.map.map_data[y][x]
                current_belief = self.known_costs[y, x]
                
                # Check for DYNAMIC OBSTACLES (User placed)
                is_dynamic_obs = (pixel.func_id == FuncID.OBSTACLE)
                
                final_cost = float(self.gt_costs[y, x])
                
                if is_dynamic_obs:
                    final_cost = math.inf
                    
                    # --- INFLATION LOGIC ---
                    # If we see a new obstacle, we artificially inflate its neighbors
                    # to prevent the robot from trying to squeeze past it.
                    if not math.isinf(current_belief):
                        # Apply high cost penalty to 8-neighbors
                        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < self.map.width and 0 <= ny < self.map.height:
                                # Don't overwrite existing walls, just add "Fear Factor"
                                updates[(nx, ny)] = max(updates.get((nx, ny), 0), 200.0)

                # Standard Cost Update
                if math.isinf(final_cost):
                    updates[(x, y)] = math.inf
                elif not math.isinf(current_belief):
                    # Only update if changed significantly
                    if abs(final_cost - current_belief) > 0.1:
                        updates[(x, y)] = final_cost

            # Apply all updates
            for (bx, by), new_cost in updates.items():
                old_cost = self.known_costs[by, bx]
                
                # If we are inflating a neighbor, adding cost to existing belief
                if not math.isinf(new_cost) and new_cost >= 200.0:
                    if not math.isinf(old_cost):
                        # Add penalty, don't replace if it's already an obstacle
                         final_val = old_cost + new_cost
                         if final_val != old_cost:
                             self.known_costs[by, bx] = final_val
                             self.planner.update_cell_cost((bx, by), final_val)
                             map_changed = True
                
                # Standard update (Obstacle or clearing)
                elif new_cost != old_cost:
                    self.known_costs[by, bx] = new_cost
                    self.planner.update_cell_cost((bx, by), new_cost)
                    map_changed = True
                    
                    if math.isinf(new_cost):
                        obstacle_detected = True
                        if (bx, by) in self.active_path:
                            path_invalidated = True

            if map_changed:
                if not self.active_path or path_invalidated or obstacle_detected:
                    self.planner.compute(max_steps=60000)
                    self._extract_full_path()
                    
                    if math.isinf(self.planner.g.get(self.start, float('inf'))):
                        print("⛔ BLOCKED: No path possible!")
                        self.active_path = []
                    elif obstacle_detected:
                        self.reroute_count += 1
                        
        except Exception:
            traceback.print_exc()

    def _extract_full_path(self):
        """Robust path extraction."""
        if not self.planner: return
        path = []
        curr = self.start
        visited = set()
        
        if math.isinf(self.planner.g.get(curr, float('inf'))):
            self.active_path = []
            return

        for _ in range(600):
            if curr in visited: break
            path.append(curr)
            visited.add(curr)
            if curr == self.goal: break
            
            best = None
            best_cost = float('inf')
            best_h = float('inf')
            
            for s in self.planner.successors(curr):
                if s in visited: continue
                
                move_cost = self.planner.cost(curr, s)
                if math.isinf(move_cost): continue
                
                g_val = self.planner.g.get(s, float('inf'))
                if math.isinf(g_val): continue
                
                total_score = g_val + move_cost
                h_val = self.planner.h(s, self.goal)
                
                # Tie-breaking logic: Prefer lower cost, then closer heuristic
                if total_score < best_cost - 1e-5:
                    best_cost = total_score
                    best = s
                    best_h = h_val
                elif abs(total_score - best_cost) < 1e-5:
                    if h_val < best_h:
                        best_cost = total_score
                        best = s
                        best_h = h_val
            
            if best: curr = best
            else: break
                
        self.active_path = path

    def step(self, compute_budget=None):
        """Robust step function."""
        if not self.planner: return self.start
        
        try:
            # 1. IMMEDIATE SAFETY CHECK
            if self.active_path:
                if len(self.active_path) > 0:
                    nx, ny = self.active_path[0]
                    if math.isinf(self.known_costs[ny, nx]):
                        self.active_path = [] 
                    else:
                        pix = self.map.map_data[ny][nx]
                        if pix.func_id == FuncID.OBSTACLE:
                            self.known_costs[ny, nx] = math.inf
                            self.planner.update_cell_cost((nx, ny), math.inf)
                            self.active_path = [] 

            # 2. Replan if needed
            if not self.active_path:
                self.planner.compute(max_steps=50000)
                self._extract_full_path()
                if not self.active_path: 
                    return self.start

            # 3. Sync
            if self.active_path[0] != self.start:
                if self.start in self.active_path:
                    idx = self.active_path.index(self.start)
                    self.active_path = self.active_path[idx:]
                else:
                    self._extract_full_path()
                    if not self.active_path: 
                        return self.start

            # 4. Move
            if len(self.active_path) > 1:
                nxt = self.active_path[1]
                self.planner.move_start(nxt)
                self.start = nxt
                self.active_path.pop(0)
                return nxt
                
        except Exception:
            traceback.print_exc()
            return self.start
            
        return self.start

def navigate_multi_map(nav_graph, start_map, start_pos, end_map, end_pos):
    result = {"success": False, "map_sequence": [], "checkpoints": {}}
    seq = nav_graph.find_map_path(start_map, end_map)
    if not seq: return result

    result["success"] = True
    result["map_sequence"] = seq
    cur_pos = start_pos
    
    for i, m in enumerate(seq):
        is_last = (i == len(seq) - 1)
        target = end_pos if is_last else nav_graph.get_teleport_to(m, seq[i + 1])
        if target is None:
            result["success"] = False; return result
            
        result["checkpoints"][m] = {"start": cur_pos, "goal": target}
        if not is_last:
            cur_pos = nav_graph.get_entry_teleport_from(m, seq[i + 1], goal_hint=target) or target

    return result
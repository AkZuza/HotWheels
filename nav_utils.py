import os
import pickle
import numpy as np
import heapq
import math
from typing import Dict, List, Tuple, Optional, Set
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
        if not os.path.exists(maps_directory): return
        for f in os.listdir(maps_directory):
            if f.endswith('.bin'):
                self._load_map(os.path.join(maps_directory, f))
        self._build_graph()

    def _load_map(self, filepath):
        try:
            with open(filepath, 'rb') as f: data = pickle.load(f)
            mid = data.get('map_id', os.path.basename(filepath)[:-4])
            w, h = data['width'], data['height']
            # Load pixels safely
            pixels = np.array([[MapPixel.from_tuple(p) for p in row] for row in data['map_data']], dtype=object)
            self.maps[mid] = MapInfo(mid, w, h, pixels, self._find_teleports(pixels, w, h))
        except: pass

    def _find_teleports(self, pixels, w, h):
        tps = {}
        for y in range(h):
            for x in range(w):
                p = pixels[y][x]
                if p.func_id in [FuncID.DOOR, FuncID.ELEVATOR, FuncID.RAMP, FuncID.STAIR] and p.identifier:
                    info = TeleportInfo((x, y), p.func_id, p.identifier, p.cost, p.dept_id, p.floor)
                    for d in p.identifier: tps.setdefault(d, []).append(info)
        return tps

    def _build_graph(self):
        self.graph = {m: set() for m in self.maps}
        for m, info in self.maps.items():
            for d in info.teleports:
                if d in self.maps:
                    self.graph[m].add(d)
                    self.graph[d].add(m)   # <<< FORCE BIDIRECTIONAL


    def find_map_path(self, start, end):
        if start == end: return [start]
        q = [(start, [start])]; vis = {start}
        while q:
            u, p = q.pop(0)
            for v in self.graph.get(u, []):
                if v not in vis:
                    if v == end: return p + [v]
                    vis.add(v); q.append((v, p + [v]))
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
        return min(candidates, key=lambda tp: abs(tp.position[0]-goal_hint[0]) + abs(tp.position[1]-goal_hint[1])).position


# -------- Dynamic Planner Wrapper --------

class DynamicLocalPlanner:
    def __init__(self, map_info, start, goal):
        self.map = map_info
        self.start = start
        self.goal = goal

        # USE YOUR MAP CREATOR COST FIELD
        self.costs = self._build_cost_grid()

        # Pass weighted grid into D*
        self.planner = DStarLite(start, goal, self.costs)

        # Initial planning
        self.planner.compute()


    def _build_cost_grid(self):
        w, h = self.map.width, self.map.height
        grid = np.zeros((w, h))

        for y in range(h):
            for x in range(w):
                p = self.map.map_data[y][x]
                # Global costmap only
                grid[x, y] = max(1.0, float(p.cost))

        return grid



    def sense_and_update(self, visible_cells):
        for (x, y) in visible_cells:
            p = self.map.map_data[y][x]

            # Discover obstacle only when seen
            if p.func_id == FuncID.OBSTACLE:
                self.planner.update_obstacle((x, y))



    def step(self):
        nxt = self.planner.next()
        if nxt is None:
            return None
        return nxt



# -------- Multi-map --------

def navigate_multi_map(nav_graph: NavigationGraph, start_map: str, start_pos: Tuple[int,int], end_map: str, end_pos: Tuple[int,int]) -> Dict:
    # Initialize result dictionary with 'paths' to avoid KeyError
    result = {'success': False, 'map_sequence': [], 'checkpoints': {}, 'paths': {}}
    
    seq = nav_graph.find_map_path(start_map, end_map)
    if not seq: return result

    result['success'] = True
    result['map_sequence'] = seq
    
    # 1. Calculate checkpoints (entry/exit points for each map)
    cur_pos = start_pos
    for i, m in enumerate(seq):
        is_last = (i == len(seq) - 1)
        target = end_pos if is_last else nav_graph.get_teleport_to(m, seq[i+1])
        
        # Fallback if target is None (broken map link)
        if target is None:
            print(f"Error: No link found from {m} to {seq[i+1] if i+1 < len(seq) else 'Goal'}")
            result['success'] = False
            return result

        result['checkpoints'][m] = {'start': cur_pos, 'goal': target}
        
        if not is_last:
            # Prepare start pos for next map
            next_start = nav_graph.get_entry_teleport_from(m, seq[i+1], goal_hint=target)
            cur_pos = next_start if next_start else target

    # 2. Calculate local paths for each map segment
    for m in seq:
        cp = result['checkpoints'][m]
        if m in nav_graph.maps:
            result['paths'][m] = []
        else:
            result['paths'][m] = []

    return result
    
def get_accessible_destinations(nav_graph: NavigationGraph,
                                from_map: str,
                                wheelchair_accessible_only: bool = True) -> List[str]:
    """
    Get all maps accessible from the given map
    If wheelchair_accessible_only, exclude paths that require stairs
    """
    if from_map not in nav_graph.maps:
        return []

    accessible = set()
    visited = {from_map}
    queue = [from_map]

    while queue:
        current = queue.pop(0)
        map_info = nav_graph.maps[current]

        for dest, teleports in map_info.teleports.items():
            if dest in nav_graph.maps and dest not in visited:
                # Check if wheelchair accessible
                if wheelchair_accessible_only:
                    has_accessible_route = any(
                        tp.func_type in [FuncID.DOOR, FuncID.ELEVATOR, FuncID.RAMP]
                        for tp in teleports
                    )
                    if not has_accessible_route:
                        continue

                accessible.add(dest)
                visited.add(dest)
                queue.append(dest)

    return sorted(accessible)


def validate_map_connections(nav_graph: NavigationGraph) -> Dict[str, List[str]]:
    """
    Validate that all teleport connections are bidirectional
    Returns dict of issues found
    """
    issues = {
        'one_way_connections': [],
        'missing_destinations': [],
        'isolated_maps': []
    }

    for map_id, map_info in nav_graph.maps.items():
        # Check if map is isolated
        if not map_info.teleports:
            issues['isolated_maps'].append(map_id)
            continue

        # Check each destination
        for dest in map_info.teleports.keys():
            # Check if destination map exists
            if dest not in nav_graph.maps:
                issues['missing_destinations'].append(f"{map_id} -> {dest} (map not found)")
                continue

            # Check if destination has return path
            dest_map = nav_graph.maps[dest]
            if map_id not in dest_map.teleports:
                issues['one_way_connections'].append(f"{map_id} -> {dest} (no return path)")

    return issues
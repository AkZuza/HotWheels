"""
Navigation Simulator - Fixed Icons & Crash Protection
"""

import pygame
import math
import numpy as np
import os
from nav_utils import DynamicLocalPlanner, NavigationGraph, navigate_multi_map
from map_creator import MapPixel, FuncID
from wheelchair import Wheelchair
from ui_manager import UIManager, THEME

class Simulator:
    def __init__(self, maps_dir='maps'):
        pygame.init()
        self.nav_graph = NavigationGraph()
        self.nav_graph.load_maps(maps_dir)
        if not self.nav_graph.maps: raise Exception("No maps found!")

        self.w, self.h = 1280, 720
        self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        pygame.display.set_caption("Wheelchair Nav System v5.1 (Fixed)")

        self.ui = UIManager(self.w, self.h, self.on_map_select)
        self.ui.update_maps(list(self.nav_graph.maps.keys()))

        # State
        self.cur_map = None
        self.map_data = None
        self.features = []
        self.wheelchair = Wheelchair(0, 0)
        self.wc_map_id = None
        self.traced_path = []

        # Assets
        self.assets = {}
        self._load_assets()

        # View
        self.zoom = 1.0
        self.off_x, self.off_y = 0, 0

        # Navigation
        self.anim_frame = 0
        self.input_mode = None
        self.start_pt = None
        self.goal_pt = None
        self.path = []
        self.sequence = []
        self.full_res = None
        self.is_moving = False
        self.path_idx = 0

        self.local_planner = None
        self.seq_idx = 0

        # Load initial map
        self.load_map(list(self.nav_graph.maps.keys())[0])

    def _load_assets(self):
        # Tries to load icons, fails gracefully if missing
        icon_files = {
            'lift': 'lift_icon.png',
            'door': 'door_icon.png',
            'stairs': 'stairs_icon.png'
        }
        for key, filename in icon_files.items():
            path = os.path.join("assets", filename)
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    self.assets[key] = img
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
            else:
                pass

    def on_map_select(self, map_id):
        self.load_map(map_id)

    def load_map(self, map_id):
        if map_id not in self.nav_graph.maps: return
        self.cur_map = map_id
        info = self.nav_graph.maps[map_id]
        self.map_data = info.map_data
        self.mw, self.mh = info.width, info.height

        # Smart Auto-Fit
        view_w, view_h = self.ui.rect_map.width, self.ui.rect_map.height
        base_w, base_h = self.mw * 30, self.mh * 30

        scale_x = (view_w - 100) / base_w
        scale_y = (view_h - 100) / base_h
        self.zoom = min(scale_x, scale_y)

        final_w, final_h = base_w * self.zoom, base_h * self.zoom
        self.off_x = (view_w - final_w) / 2
        self.off_y = (view_h - final_h) / 2

        self._scan_features()

    def _scan_features(self):
        self.features = []
        if self.map_data is None: return
        visited = set()
        rows, cols = len(self.map_data), len(self.map_data[0])

        for y in range(rows):
            for x in range(cols):
                if (x,y) in visited: continue

                # Ensure pixel is treated as object, not tuple
                pixel = self.map_data[y][x]
                if isinstance(pixel, tuple):
                    pixel = MapPixel.from_tuple(pixel)
                    self.map_data[y][x] = pixel

                pid = pixel.func_id

                if pid in [FuncID.ELEVATOR, FuncID.DOOR, FuncID.STAIR]:
                    cluster = []
                    q = [(x,y)]
                    visited.add((x,y))
                    label = ""
                    if pixel.identifier:
                        label = str(pixel.identifier[0]).replace("_", " ").title()

                    while q:
                        cx, cy = q.pop(0)
                        cluster.append((cx,cy))
                        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nx, ny = cx+dx, cy+dy
                            if 0<=nx<cols and 0<=ny<rows:
                                if (nx,ny) not in visited:
                                    np_pix = self.map_data[ny][nx]
                                    if isinstance(np_pix, tuple):
                                        np_pix = MapPixel.from_tuple(np_pix)
                                        self.map_data[ny][nx] = np_pix

                                    if np_pix.func_id == pid:
                                        visited.add((nx,ny))
                                        q.append((nx,ny))

                    xs, ys = [p[0] for p in cluster], [p[1] for p in cluster]
                    self.features.append({
                        'type': pid,
                        'pos': ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2),
                        'size': max(len(set(xs)), len(set(ys))),
                        'label': label
                    })

    def map_to_screen(self, mx, my):
        ts = 30 * self.zoom
        return (self.off_x + mx * ts, self.off_y + my * ts)

    def screen_to_map(self, sx, sy):
        ts = 30 * self.zoom
        if ts <= 0.1: return None
        mx = int((sx - self.off_x) / ts)
        my = int((sy - self.off_y) / ts)
        if 0 <= mx < self.mw and 0 <= my < self.mh: return mx, my
        return None

    def reset_path(self):
        self.path = []
        self.traced_path=[]
        self.full_res = None
        self.is_moving = False
        self.path_idx = 0
        self.wc_map_id = None
        self.local_planner = None
        self.sequence = []
        self.seq_idx = 0

    def start_nav(self):
        if not self.start_pt or not self.goal_pt:
            return "Set Start & Goal First!"

        res = navigate_multi_map(
            self.nav_graph,
            self.start_pt[0], self.start_pt[1],
            self.goal_pt[0], self.goal_pt[1]
        )

        if not res['success'] or not res['map_sequence']:
            self.is_moving = False
            return "Path Blocked!"

        self.full_res = res
        self.sequence = res['map_sequence']

        self.cur_map = self.sequence[0]
        self.load_map(self.cur_map)

        self.seq_idx = 0
        cp = self.full_res['checkpoints'][self.sequence[0]]
        print("START", cp["start"], "GOAL", cp["goal"])
        
        temp_wc = Wheelchair(*cp['start'])
        initial_visible = temp_wc.visible_cells(self.nav_graph.maps[self.sequence[0]].map_data, radius=5)

        self.local_planner = DynamicLocalPlanner(
            self.nav_graph.maps[self.sequence[0]],
            cp['start'],
            cp['goal'],
            nav_graph=self.nav_graph,
            initial_visible_cells = initial_visible
        )

        self.wheelchair = Wheelchair(*cp['start'])
        self.wc_map_id = self.cur_map
        self.traced_path = [cp['start']]
        self.is_moving = True
        return "Navigating..."

    def add_obstacle(self, map_coords):
        cx, cy = map_coords
        R = 4

        for dy in range(-R, R+1):
            for dx in range(-R, R+1):
                x, y = cx+dx, cy+dy
                if 0 <= x < self.mw and 0 <= y < self.mh:
                    if math.sqrt(dx*dx + dy*dy) <= R:
                        p = self.map_data[y][x]
                        if isinstance(p, tuple):
                            p = MapPixel.from_tuple(p)
                            self.map_data[y][x] = p

                        p.func_id = FuncID.OBSTACLE
                        p.cost = 999

        return "Dynamic obstacle added"

    def update(self):
        try:
            self.anim_frame = (self.anim_frame + 1) % 60

            if self.is_moving and self.local_planner:
                visible = self.wheelchair.visible_cells(self.map_data, radius=5)
                if self.anim_frame % 3 == 0:
                    self.local_planner.sense_and_update(visible)

                nxt = self.local_planner.step(compute_budget=1200)

                if nxt is None:
                    self.no_move_frames = getattr(self, "no_move_frames", 0) + 1
                    if self.no_move_frames > 60:
                        self.is_moving = False
                        print("D* Lite: Path Blocked (timeout), stopping safely")
                    return
                else:
                    self.no_move_frames = 0

                self.wheelchair.update_pos(nxt)
                self.traced_path.append(nxt)

                # reached local goal (teleport or final)
                if nxt == self.local_planner.goal:
                    self.seq_idx += 1

                    if self.seq_idx >= len(self.sequence):
                        self.is_moving = False
                        print("Navigation complete.")
                        return

                    # switch map
                    self.cur_map = self.sequence[self.seq_idx]
                    self.load_map(self.cur_map)

                    self.traced_path = []

                    cp = self.full_res['checkpoints'][self.cur_map]
                    self.local_planner = DynamicLocalPlanner(
                        self.nav_graph.maps[self.cur_map],
                        cp['start'],
                        cp['goal']
                    )

                    self.wheelchair = Wheelchair(*cp['start'])
                    self.wc_map_id = self.cur_map
        except Exception:
            import traceback
            traceback.print_exc()
            self.is_moving = False

    def draw_marker(self, map_coords, color, label):
        sx, sy = self.map_to_screen(map_coords[0]+0.5, map_coords[1]+0.5)
        offset = math.sin(self.anim_frame * 0.1) * 5

        shadow_scale = 1.0 - (offset + 5) / 40.0
        pygame.draw.ellipse(self.screen, (0,0,0,50),
                          (sx - 10*shadow_scale, sy - 5*shadow_scale, 20*shadow_scale, 10*shadow_scale))

        float_y = sy - 30 - offset

        pts = [(sx, float_y + 10), (sx - 10, float_y - 5), (sx + 10, float_y - 5)]
        pygame.draw.polygon(self.screen, color, pts)
        pygame.draw.circle(self.screen, color, (sx, float_y - 5), 16)
        pygame.draw.circle(self.screen, (255,255,255), (sx, float_y - 5), 6)

        if label:
            font = pygame.font.SysFont("Segoe UI", 12, bold=True)
            txt = font.render(label, True, (255,255,255))
            bg_rect = pygame.Rect(0, 0, txt.get_width() + 12, txt.get_height() + 12)
            bg_rect.midbottom = (sx, float_y - 25)
            pygame.draw.rect(self.screen, THEME['surface'], bg_rect, border_radius=4)
            self.screen.blit(txt, txt.get_rect(center=bg_rect.center))

    def draw_scene(self):
        ts = 30 * self.zoom
        self.screen.fill(THEME['void'])
        self.screen.set_clip(self.ui.rect_map)

        # --- Map Floor ---
        map_rect = pygame.Rect(self.off_x, self.off_y, self.mw * ts, self.mh * ts)
        pygame.draw.rect(self.screen, (0,0,0), (map_rect.x+10, map_rect.y+10, map_rect.w, map_rect.h))
        pygame.draw.rect(self.screen, THEME['map_bg'], map_rect)
        pygame.draw.rect(self.screen, THEME['map_border'], map_rect, 2)

        # --- Tile Renderer (EMPTY=WHITE, WALKABLE=GREEN, OBSTACLE=BLACK) ---
        COL_EMPTY = (255, 255, 255)
        COL_WALK = (0, 200, 0)
        COL_OBS = (0, 0, 0)

        start_x = max(0, int(-self.off_x / ts))
        end_x = min(self.mw, int((self.ui.rect_map.width - self.off_x) / ts) + 1)
        start_y = max(0, int(-self.off_y / ts))
        end_y = min(self.mh, int((self.ui.rect_map.height - self.off_y) / ts) + 1)

        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                px = self.map_data[y][x]
                rx, ry = self.map_to_screen(x, y)
                rect = pygame.Rect(int(rx), int(ry), int(ts) + 1, int(ts) + 1)

                if px.func_id == FuncID.OBSTACLE or px.cost > 50:
                    pygame.draw.rect(self.screen, COL_OBS, rect)
                elif px.func_id == FuncID.WALKABLE:
                    pygame.draw.rect(self.screen, COL_WALK, rect)
                else:
                    pygame.draw.rect(self.screen, COL_EMPTY, rect)

        # --- Traced Path ---
        if len(self.traced_path) > 1 and self.wc_map_id == self.cur_map:
            pts = [self.map_to_screen(p[0]+0.5, p[1]+0.5) for p in self.traced_path]
            pygame.draw.lines(self.screen, (40, 200, 255), False, pts, max(6, int(8*self.zoom)))

        # --- Icons (RESTORED) ---
        font_label = pygame.font.SysFont("Segoe UI", 12)
        for f in self.features:
            cx, cy = self.map_to_screen(f['pos'][0]+0.5, f['pos'][1]+0.5)
            sz = int(max(36, ts * 1.4))

            icn = None
            if f['type'] == FuncID.ELEVATOR: icn = self.assets.get('lift')
            elif f['type'] == FuncID.DOOR: icn = self.assets.get('door')
            elif f['type'] == FuncID.STAIR: icn = self.assets.get('stairs')

            if icn:
                i_rect = icn.get_rect()
                scale = min(sz / i_rect.width, sz / i_rect.height)
                new_size = (int(i_rect.width * scale), int(i_rect.height * scale))
                s_icn = pygame.transform.smoothscale(icn, new_size)
                self.screen.blit(s_icn, s_icn.get_rect(center=(cx, cy)))

            if f['label']:
                lbl_surf = font_label.render(f['label'], True, THEME['text_dark'])
                bg_rect = lbl_surf.get_rect(midtop=(cx, cy + sz//2 + 4))
                bg_rect.inflate_ip(8, 4)
                pygame.draw.rect(self.screen, (255,255,255, 220), bg_rect, border_radius=4)
                pygame.draw.rect(self.screen, (200,200,200), bg_rect, 1, border_radius=4)
                self.screen.blit(lbl_surf, bg_rect.move(4,2))

        # --- Markers ---
        if self.start_pt and self.start_pt[0] == self.cur_map:
            self.draw_marker(self.start_pt[1], THEME['success'], "START")

        if self.goal_pt and self.goal_pt[0] == self.cur_map:
            self.draw_marker(self.goal_pt[1], THEME['danger'], "GOAL")

        # --- Wheelchair ---
        if self.wc_map_id == self.cur_map:
            self.wheelchair.draw(self.screen, ts, self.off_x, self.off_y)

        # --- Placement Preview ---
        if self.input_mode:
            mpos = pygame.mouse.get_pos()
            coord = self.screen_to_map(*mpos)
            if coord:
                lbl, col = "", (255,255,255)
                if self.input_mode=="START": lbl, col = "START?", THEME['success']
                elif self.input_mode=="GOAL": lbl, col = "GOAL?", THEME['danger']
                elif self.input_mode=="ADD_OBSTACLE": lbl, col = "BLOCK?", THEME['warning']
                self.draw_marker(coord, col, lbl)

        self.screen.set_clip(None)

    def run(self):
        clock = pygame.time.Clock()
        status = "Select Map & Set Points"
        running = True

        while running:
            route_str = ""
            if self.sequence:
                route_str = " -> ".join([s.replace("_", " ").title() for s in self.sequence])

            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False

                act = self.ui.handle_input(e)
                if act:
                    if act == "START":
                        self.input_mode = "START"
                        self.reset_path()
                        status = "Click Map for START"
                    elif act == "GOAL":
                        self.input_mode = "GOAL"
                        self.reset_path()
                        status = "Click Map for GOAL"
                    elif act == "ADD_OBSTACLE":
                        self.input_mode = "ADD_OBSTACLE"
                        status = "Click to Add Obstacle"
                    elif act == "NAV": status = self.start_nav()
                    elif act == "RESET":
                        self.start_pt = None
                        self.goal_pt = None
                        self.traced_path = []
                        self.sequence = []
                        self.local_planner = None
                        self.is_moving = False
                        self.wc_map_id = None
                        status = "Reset Complete."
                    elif act == "ZOOM_IN": self.zoom *= 1.2
                    elif act == "ZOOM_OUT": self.zoom /= 1.2
                    elif act == "MAP_CHANGED":
                        self.traced_path = []
                        self.sequence = []
                        self.local_planner = None
                        self.is_moving = False
                        self.wc_map_id = None
                        status = f"Viewing {self.cur_map}"

                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and not act:
                    if self.ui.rect_map.collidepoint(e.pos) and self.input_mode:
                        mp = self.screen_to_map(*e.pos)
                        if mp:
                            if self.input_mode == "START":
                                self.start_pt = (self.cur_map, mp)
                                status = "Start Set."
                                self.input_mode = None
                            elif self.input_mode == "GOAL":
                                self.goal_pt = (self.cur_map, mp)
                                status = "Goal Set."
                                self.input_mode = None
                            elif self.input_mode == "ADD_OBSTACLE":
                                status = self.add_obstacle(mp)
                                # Keep input mode active for multiple obstacles

                if e.type == pygame.MOUSEMOTION and e.buttons[0] and not self.input_mode:
                    if self.ui.rect_map.collidepoint(e.pos):
                        self.off_x += e.rel[0]
                        self.off_y += e.rel[1]

            self.update()
            self.draw_scene()
            self.ui.draw(self.screen, status, self.is_moving, route_str)

            if self.input_mode:
                msg = f"PLACING {self.input_mode}"
                f = pygame.font.SysFont("Segoe UI", 20, bold=True)
                s = f.render(msg, True, (255,255,255))
                bg = pygame.Rect(0,0, s.get_width()+40, 40)
                bg.center = (self.ui.rect_map.width//2, 40)
                pygame.draw.rect(self.screen, (0,0,0,180), bg, border_radius=20)
                self.screen.blit(s, s.get_rect(center=bg.center))

            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    try:
        Simulator().run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("CRASHED. Press Enter to exit...")


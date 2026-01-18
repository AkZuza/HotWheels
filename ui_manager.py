import pygame

# --- TAILWIND-INSPIRED THEME ---
THEME = {
    'void': (15, 23, 42), 'surface': (30, 41, 59), 'surface_hover': (51, 65, 85),
    'map_bg': (248, 250, 252), 'map_border': (148, 163, 184),
    'wall_fill': (71, 85, 105), 'wall_outline': (51, 65, 85),
    'text_main': (241, 245, 249), 'text_muted': (148, 163, 184), 'text_dark': (15, 23, 42),
    'accent': (99, 102, 241), 'accent_hover': (79, 70, 229),
    'danger': (239, 68, 68), 'danger_hover': (220, 38, 38),
    'success': (34, 197, 94), 'warning': (234, 179, 8), 'warning_hover': (202, 138, 4),
    'shadow': (0, 0, 0, 60),
}

def draw_rounded_rect(surface, rect, color, radius=10, shadow=True):
    if shadow:
        shadow_rect = pygame.Rect(rect.x + 4, rect.y + 4, rect.width, rect.height)
        pygame.draw.rect(surface, THEME['shadow'], shadow_rect, border_radius=radius)
    pygame.draw.rect(surface, color, rect, border_radius=radius)

class Button:
    def __init__(self, rect, text, action_id, bg_color=None, hover_color=None, text_color=None):
        self.rect = rect
        self.text = text
        self.action_id = action_id
        self.base_color = bg_color if bg_color else THEME['surface']
        self.hover_color = hover_color if hover_color else THEME['surface_hover']
        self.text_color = text_color if text_color else THEME['text_main']
        self.hovered = False

    def draw(self, screen, font):
        color = self.hover_color if self.hovered else self.base_color
        draw_rounded_rect(screen, self.rect, color, radius=8, shadow=True)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered: return self.action_id
        return None

class MapListWidget:
    def __init__(self, rect, map_names, callback):
        self.rect = rect
        self.map_names = sorted(map_names)
        self.callback = callback
        self.scroll_y = 0
        self.item_h = 50
        self.hover_idx = -1

    def draw(self, screen, font):
        draw_rounded_rect(screen, self.rect, THEME['void'], radius=8, shadow=False)
        clip_rect = self.rect.inflate(-4, -4)
        screen.set_clip(clip_rect)
        start_y = self.rect.y + self.scroll_y + 5
        for i, name in enumerate(self.map_names):
            y = start_y + i * (self.item_h + 5)
            if y + self.item_h < self.rect.top or y > self.rect.bottom: continue
            item_rect = pygame.Rect(self.rect.x + 10, y, self.rect.width - 20, self.item_h)
            color = THEME['accent'] if i == self.hover_idx else THEME['surface']
            pygame.draw.rect(screen, color, item_rect, border_radius=6)
            clean_name = name.replace("_", " ").title()
            surf = font.render(clean_name, True, THEME['text_main'])
            screen.blit(surf, (item_rect.x + 15, item_rect.centery - surf.get_height()//2))
        screen.set_clip(None)

    def handle_event(self, event):
        if event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll_y += event.y * 20
            self.scroll_y = min(0, max(self.scroll_y, self.rect.height - len(self.map_names) * (self.item_h + 5)))
            return True
        elif event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                rel_y = event.pos[1] - (self.rect.y + self.scroll_y + 5)
                idx = int(rel_y // (self.item_h + 5))
                self.hover_idx = idx if 0 <= idx < len(self.map_names) else -1
            else: self.hover_idx = -1
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hover_idx != -1 and 0 <= self.hover_idx < len(self.map_names):
                self.callback(self.map_names[self.hover_idx])
                return "MAP_SELECTED"
        return None

class UIManager:
    def __init__(self, w, h, map_cb):
        self.w, self.h = w, h
        self.sidebar_w = 280
        self.bar_h = 100
        self.rect_sidebar = pygame.Rect(w - self.sidebar_w, 0, self.sidebar_w, h)
        self.rect_bottom = pygame.Rect(0, h - self.bar_h, w - self.sidebar_w, self.bar_h)
        self.rect_map = pygame.Rect(0, 0, w - self.sidebar_w, h - self.bar_h)
        self.font_head = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 16)
        self.font_small = pygame.font.SysFont("Segoe UI", 14)
        self.show_list = False
        self.buttons = self._make_buttons()
        self.map_list = MapListWidget(pygame.Rect(self.rect_sidebar.x + 20, 80, self.rect_sidebar.width - 40, h - 180), [], map_cb)

    def _make_buttons(self):
        bx, bw, bh, gap = self.rect_sidebar.x + 20, self.rect_sidebar.width - 40, 50, 15
        y = 80
        return [
            Button(pygame.Rect(bx, y, bw, bh), "SET START POINT", "START"),
            Button(pygame.Rect(bx, y + bh + gap, bw, bh), "SET GOAL POINT", "GOAL"),
            # New Obstacle Button
            Button(pygame.Rect(bx, y + (bh + gap)*2, bw, bh), "ADD OBSTACLE", "ADD_OBSTACLE", THEME['warning'], THEME['warning_hover']),
            
            Button(pygame.Rect(bx, y + (bh + gap)*3, bw, bh), "NAVIGATE", "NAV", THEME['success'], (22, 163, 74)),
            Button(pygame.Rect(bx, y + (bh + gap)*4, bw, bh), "RESET", "RESET", THEME['danger'], THEME['danger_hover']),
            
            Button(pygame.Rect(bx, self.h - 100, bw, bh), "CHANGE MAP", "OPEN_LIST", THEME['accent'], THEME['accent_hover']),
            Button(pygame.Rect(self.rect_map.right - 70, self.rect_map.bottom - 130, 50, 50), "+", "ZOOM_IN"),
            Button(pygame.Rect(self.rect_map.right - 70, self.rect_map.bottom - 70, 50, 50), "-", "ZOOM_OUT"),
        ]

    def update_maps(self, names): self.map_list.map_names = sorted(names)

    def draw(self, screen, status, is_moving, route_str=""):
        pygame.draw.rect(screen, THEME['surface'], self.rect_sidebar)
        pygame.draw.line(screen, (0,0,0,50), (self.rect_sidebar.x, 0), (self.rect_sidebar.x, self.h), 2)
        if self.show_list:
            self.map_list.draw(screen, self.font_body)
            Button(pygame.Rect(self.rect_sidebar.x+20, self.h-80, self.rect_sidebar.width-40, 50), "CANCEL", "CLOSE_LIST", THEME['danger']).draw(screen, self.font_head)
        else:
            screen.blit(self.font_head.render("CONTROLS", True, THEME['text_main']), (self.rect_sidebar.x + 20, 30))
            for btn in self.buttons: btn.draw(screen, self.font_head)
        
        pygame.draw.rect(screen, THEME['surface'], self.rect_bottom)
        pygame.draw.circle(screen, THEME['success'] if is_moving else THEME['accent'], (40, self.rect_bottom.top + 30), 6)
        screen.blit(self.font_head.render(status, True, THEME['text_main']), (60, self.rect_bottom.top + 15))
        if route_str:
            screen.blit(self.font_small.render("PATH:", True, THEME['text_muted']), (60, self.rect_bottom.top + 55))
            screen.blit(self.font_body.render(route_str, True, THEME['accent']), (110, self.rect_bottom.top + 52))

    def handle_input(self, event):
        if self.show_list:
            res = self.map_list.handle_event(event)
            if res == "MAP_SELECTED": self.show_list = False; return "MAP_CHANGED"
            if event.type == pygame.MOUSEBUTTONDOWN and pygame.Rect(self.rect_sidebar.x+20, self.h-80, self.rect_sidebar.width-40, 50).collidepoint(event.pos): self.show_list = False
            return None
        for btn in self.buttons:
            act = btn.handle_event(event)
            if act:
                if act == "OPEN_LIST": self.show_list = True
                else: return act
        return None
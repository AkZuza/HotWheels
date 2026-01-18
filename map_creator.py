"""
Wheelchair Accessibility Map Creator
A pygame-based 2D map editor for marking wheelchair accessible routes
"""

import pygame
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import IntEnum

# Constants
class FuncID(IntEnum):
    """Functional identifiers for map elements"""
    EMPTY = 0
    WALKABLE = 1
    RAMP = 2
    DOOR = 3
    ELEVATOR = 4
    STAIR = 5
    OBSTACLE = 6

@dataclass
class MapPixel:
    """Data structure for each pixel on the map"""
    cost: int = 0  # Movement cost for pathfinding
    dept_id: int = 0  # Department ID
    floor: int = 0  # Floor level
    func_id: int = FuncID.EMPTY  # Functional type
    identifier: List[str] = field(default_factory=list)  # List of destination maps for teleports
    
    def to_tuple(self):
        return (self.cost, self.dept_id, self.floor, self.func_id, self.identifier.copy())
    
    @staticmethod
    def from_tuple(data):
        """Load MapPixel with backward compatibility"""
        # Handle old format without identifier (4 elements)
        if len(data) == 4:
            return MapPixel(data[0], data[1], data[2], data[3], [])
        
        # Handle old format with string identifier (5 elements, last is str)
        if len(data) == 5:
            if isinstance(data[4], str):
                # Convert old string format to list
                # Split by comma and strip whitespace
                destinations = [d.strip() for d in data[4].split(',') if d.strip()]
                return MapPixel(data[0], data[1], data[2], data[3], destinations)
            elif isinstance(data[4], list):
                # New format - already a list
                return MapPixel(data[0], data[1], data[2], data[3], data[4])
        
        # Fallback
        return MapPixel(data[0], data[1], data[2], data[3], [])

class MapCreator:
    """Main map creator class"""
    
    # Color scheme
    COLORS = {
        'background': (240, 240, 240),
        'grid': (200, 200, 200),
        'ui_bg': (50, 50, 50),
        'ui_text': (255, 255, 255),
        'button': (70, 70, 70),
        'button_hover': (100, 100, 100),
        'button_active': (120, 150, 200),
        FuncID.EMPTY: (255, 255, 255),
        FuncID.WALKABLE: (200, 255, 200),
        FuncID.RAMP: (150, 200, 255),
        FuncID.DOOR: (255, 200, 150),
        FuncID.ELEVATOR: (200, 150, 255),
        FuncID.STAIR: (255, 150, 150),
        FuncID.OBSTACLE: (100, 100, 100),
    }
    
    def __init__(self, map_width=100, map_height=100, window_width=1200, window_height=800, map_id="unnamed_map"):
        pygame.init()
        
        # Map properties
        self.map_id = map_id  # Identifier for this map (used in cross-map navigation)
        self.map_width = map_width
        self.map_height = map_height
        self.map_data = np.array([[MapPixel() for _ in range(map_width)] 
                                   for _ in range(map_height)], dtype=object)
        
        # Window properties
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height), 
                                               pygame.RESIZABLE)
        pygame.display.set_caption(f"Map Creator - {self.map_id}")
        
        # UI dimensions
        self.toolbar_width = 200
        self.statusbar_height = 30
        self.canvas_width = window_width - self.toolbar_width
        self.canvas_height = window_height - self.statusbar_height
        
        # Start Entry 
        self.x_start = -1 
        self.y_start = -1

        # View properties
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Drawing state
        self.current_tool = 'pen'
        self.current_func_id = FuncID.WALKABLE
        self.current_dept_id = 1
        self.current_floor = 0
        self.current_cost = 1
        self.current_destinations = []  # List of destination maps for teleports
        self.brush_size = 1
        
        # Drawing mode
        self.is_drawing = False
        self.last_draw_pos = None
        
        # Shape drawing
        self.shape_start = None
        self.temp_shape_pixels = []
        
        # UI state
        self.dragging_view = False
        self.drag_start = None
        
        # Text input state
        self.active_textbox = None  # Which textbox is active: 'dept_id', 'destinations', or None
        self.textbox_content = {
            'dept_id': str(self.current_dept_id),
            'destinations': ''  # Comma-separated list
        }
        self.cursor_visible = True
        self.cursor_timer = 0
        
        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Clock
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create UI buttons
        self.buttons = self._create_buttons()
        
        # Performance optimizations
        self.canvas_surface = pygame.Surface((self.canvas_width, self.canvas_height))
        self.canvas_dirty = True
        self.last_visible_area = None
        self.cached_pixel_surfaces = {}  # Cache colored rectangles
        
        # Pre-create pixel surfaces for each function type
        for func_id in FuncID:
            color = self.COLORS[func_id]
            surf = pygame.Surface((1, 1))
            surf.fill(color)
            self.cached_pixel_surfaces[func_id] = surf
        
        # Create color lookup array for ultra-fast color access
        self.color_array = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        self._update_color_array()
    
    def _update_color_array(self):
        """Update the color array from map data - called when map changes"""
        for y in range(self.map_height):
            for x in range(self.map_width):
                pixel = self.map_data[y][x]
                color = self.COLORS.get(pixel.func_id, self.COLORS[FuncID.EMPTY])
                self.color_array[y, x] = color
        
    def _create_buttons(self):
        """Create UI buttons for tools and functions"""
        buttons = []
        y_offset = 10
        button_height = 40
        button_spacing = 10
        
        # Tool buttons
        tools = [
            # ('pen', 'Pen'),
            ('eraser', 'Eraser'),
            ('rect', 'Rectangle'),
            ('filled_rect', 'Filled Rect'),
            # ('circle', 'Circle'),
            # ('filled_circle', 'Filled Circle'),
            ('line', 'Line'),
            ('door', 'Door'),
            ('ramp', 'Ramp'),
            ('elevator', 'elevator')
        ]
        
        buttons.append({'type': 'label', 'text': 'Tools:', 'y': y_offset})
        y_offset += 30
        
        for tool_id, tool_name in tools:
            buttons.append({
                'type': 'tool',
                'id': tool_id,
                'text': tool_name,
                'rect': pygame.Rect(10, y_offset, self.toolbar_width - 20, button_height),
                'y': y_offset
            })
            y_offset += button_height + button_spacing
        
        y_offset += 10
        
        # Function type buttons
        buttons.append({'type': 'label', 'text': 'Function Type:', 'y': y_offset})
        y_offset += 25
        func_types = [
            (FuncID.EMPTY, 'Empty'),
            (FuncID.WALKABLE, 'Walkable'),
            (FuncID.STAIR, 'Stair'),
            (FuncID.OBSTACLE, 'Obstacle'),
        ]
        
        for func_id, func_name in func_types:
            buttons.append({
                'type': 'func',
                'id': func_id,
                'text': func_name,
                'rect': pygame.Rect(10, y_offset, self.toolbar_width - 20, 35),
                'color': self.COLORS[func_id],
                'y': y_offset
            })
            y_offset += 35 + 5
        y_offset += 20
        buttons.append({
            'type': 'textbox',
            'id': 'y_start',
            'label': 'y_START',
            'rect': pygame.Rect(10, y_offset, self.toolbar_width-20, 30)
        })
        y_offset += 40

        buttons.append({
            'type': 'textbox',
            'id': 'x_start',
            'label': 'y_START',
            'rect': pygame.Rect(10, y_offset, self.toolbar_width-20, 30)
        })
        y_offset += 40

        # Text input sections
        buttons.append({'type': 'label', 'text': 'Settings:', 'y': y_offset})
        y_offset += 45  # Increased from 25 to 35
        
        # Department ID textbox
        buttons.append({
            'type': 'textbox',
            'id': 'dept_id',
            'label': 'Dept ID:',
            'rect': pygame.Rect(10, y_offset, self.toolbar_width - 20, 30),
            'y': y_offset
        })
        y_offset += 65  # Increased from 50 to 55
        
        # Destinations textbox (for teleport connections)
        buttons.append({
            'type': 'textbox',
            'id': 'destinations',
            'label': 'Destinations (comma-separated):',
            'rect': pygame.Rect(10, y_offset, self.toolbar_width - 20, 50),
            'y': y_offset,
            'multiline_label': True  # Longer label needs wrapping
        })
        y_offset += 75  # Increased from 70 to 75
        
        return buttons
    
    def world_to_screen(self, wx, wy):
        """Convert world coordinates to screen coordinates"""
        sx = (wx * self.zoom) + self.offset_x
        sy = (wy * self.zoom) + self.offset_y
        return int(sx), int(sy)
    
    def screen_to_world(self, sx, sy):
        """Convert screen coordinates to world coordinates"""
        wx = (sx - self.offset_x) / self.zoom
        wy = (sy - self.offset_y) / self.zoom
        return int(wx), int(wy)
    
    def screen_to_map(self, sx, sy):
        """Convert screen coordinates to map coordinates"""
        wx, wy = self.screen_to_world(sx, sy)
        if 0 <= wx < self.map_width and 0 <= wy < self.map_height:
            return wx, wy
        return None
    
    def draw_pixel(self, x, y):
        """Draw a pixel on the map with current settings"""
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            if self.current_tool == 'eraser':
                # Completely reset the pixel
                self.map_data[y][x] = MapPixel()
                self.color_array[y, x] = self.COLORS[FuncID.EMPTY]
            else:
                # Overwrite with new values completely
                self.map_data[y][x] = MapPixel(
                    cost=self.current_cost,
                    dept_id=self.current_dept_id,
                    floor=self.current_floor,
                    func_id=self.current_func_id,
                    identifier=self.current_destinations.copy()  # Copy the list
                )
                self.color_array[y, x] = self.COLORS[self.current_func_id]
            self.canvas_dirty = True

    def draw_area(self, x, y, width, height):
        if 0 <= x < self.map_width and 0 <= y < self.map_height and 0 < x + width <= self.map_width and 0 < y + height <= self.map_height:
            for i in range(x, width):
                for j in range(y, height):
                    self.map_data[j][i] = MapPixel(
                        cost = self.current_cost,
                        dept_id = self.current_dept_id,
                        floor=self.current_floor,
                        func_id=self.current_func_id,
                        identifier=self.current_destinations.copy()
                    )
        self.canvas_dirty = True

    def draw_line_pixels(self, x0, y0, x1, y1):
        """Draw a line of pixels using Bresenham's algorithm"""
        pixels = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            pixels.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return pixels
    
    def draw_circle_pixels(self, cx, cy, radius, filled=False):
        """Get pixels for a circle"""
        pixels = []
        
        if filled:
            for y in range(-radius, radius + 1):
                for x in range(-radius, radius + 1):
                    if x*x + y*y <= radius*radius:
                        pixels.append((cx + x, cy + y))
        else:
            # Midpoint circle algorithm
            x = 0
            y = radius
            d = 1 - radius
            
            while x <= y:
                for px, py in [(x, y), (-x, y), (x, -y), (-x, -y),
                               (y, x), (-y, x), (y, -x), (-y, -x)]:
                    pixels.append((cx + px, cy + py))
                
                if d < 0:
                    d += 2 * x + 3
                else:
                    d += 2 * (x - y) + 5
                    y -= 1
                x += 1
        
        return pixels
    
    def draw_rect_pixels(self, x0, y0, x1, y1, filled=False):
        """Get pixels for a rectangle"""
        pixels = []
        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)
        
        if filled:
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    pixels.append((x, y))
        else:
            # Draw four sides
            for x in range(min_x, max_x + 1):
                pixels.append((x, min_y))
                pixels.append((x, max_y))
            for y in range(min_y + 1, max_y):
                pixels.append((min_x, y))
                pixels.append((max_x, y))
        
        return pixels
    
    def handle_drawing(self, mouse_pos):
        """Handle drawing operations"""
        map_pos = self.screen_to_map(mouse_pos[0], mouse_pos[1])
        if not map_pos:
            return
        
        mx, my = map_pos
        
        if self.current_tool in ['pen', 'eraser']:
            # Draw with brush
            for dy in range(-self.brush_size + 1, self.brush_size):
                for dx in range(-self.brush_size + 1, self.brush_size):
                    if dx*dx + dy*dy < self.brush_size * self.brush_size:
                        self.draw_pixel(mx + dx, my + dy)
            
            # Draw line from last position for smooth drawing
            if self.last_draw_pos:
                lx, ly = self.last_draw_pos
                for x, y in self.draw_line_pixels(lx, ly, mx, my):
                    for dy in range(-self.brush_size + 1, self.brush_size):
                        for dx in range(-self.brush_size + 1, self.brush_size):
                            if dx*dx + dy*dy < self.brush_size * self.brush_size:
                                self.draw_pixel(x + dx, y + dy)
            
            self.last_draw_pos = (mx, my)
        
        elif self.current_tool in ['rect', 'filled_rect', 'circle', 'filled_circle', 'line', 'door', 'ramp', 'elevator']:
            # Shape drawing - store preview
            if self.shape_start:
                sx, sy = self.shape_start
                
                if self.current_tool == 'line':
                    self.temp_shape_pixels = self.draw_line_pixels(sx, sy, mx, my)
                elif self.current_tool in ['rect', 'filled_rect', 'door', 'ramp', 'elevator']:
                    filled = self.current_tool in ['filled_rect', 'door', 'ramp', 'elevator']
                    self.temp_shape_pixels = self.draw_rect_pixels(sx, sy, mx, my, filled)
                elif self.current_tool in ['circle', 'filled_circle']:
                    filled = self.current_tool == 'filled_circle'
                    radius = int(((mx - sx)**2 + (my - sy)**2)**0.5)
                    self.temp_shape_pixels = self.draw_circle_pixels(sx, sy, radius, filled)
    
    def finalize_shape(self):
        """Finalize shape drawing"""
        if self.temp_shape_pixels:
            for x, y in self.temp_shape_pixels:
                self.draw_pixel(x, y)
            self.temp_shape_pixels = []
            self.canvas_dirty = True
        self.shape_start = None
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.VIDEORESIZE:
                self.window_width = event.w
                self.window_height = event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.canvas_width = self.window_width - self.toolbar_width
                self.canvas_height = self.window_height - self.statusbar_height
                self.canvas_surface = pygame.Surface((self.canvas_width, self.canvas_height))
                self.canvas_dirty = True
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom
                mouse_pos = pygame.mouse.get_pos()
                if mouse_pos[0] < self.canvas_width:
                    old_zoom = self.zoom
                    if event.y > 0:
                        self.zoom = min(self.zoom * 1.05, self.max_zoom)
                    else:
                        self.zoom = max(self.zoom / 1.05, self.min_zoom)
                    # Adjust offset to zoom toward mouse
                    """
                    wx, wy = self.screen_to_world(mouse_pos[0], mouse_pos[1])
                    self.offset_x = mouse_pos[0] - wx * self.zoom
                    self.offset_y = mouse_pos[1] - wy * self.zoom
                    """
                    self.canvas_dirty = True
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check toolbar buttons
                if mouse_pos[0] >= self.canvas_width:
                    clicked_outside_textbox = True
                    for button in self.buttons:
                        if button['type'] in ['tool', 'func', 'textbox'] and 'rect' in button:
                            if button['rect'].collidepoint(mouse_pos[0] - self.canvas_width, mouse_pos[1]):
                                if button['type'] == 'tool':
                                    self.current_tool = button['id']
                                    if self.current_tool == 'door':
                                        self.current_func_id = FuncID.DOOR
                                    elif self.current_tool == 'ramp':
                                        self.current_func_id = FuncID.RAMP
                                    elif self.current_tool == 'elevator':
                                        self.current_func_id = FuncID.RAMP

                                    self.active_textbox = None  # Deactivate textbox
                                elif button['type'] == 'func':
                                    self.current_func_id = button['id']
                                    self.active_textbox = None  # Deactivate textbox
                                elif button['type'] == 'textbox':
                                    self.active_textbox = button['id']
                                    clicked_outside_textbox = False
                                break
                    
                    if clicked_outside_textbox:
                        self.active_textbox = None
                
                # Canvas interactions
                elif mouse_pos[0] < self.canvas_width:
                    self.active_textbox = None  # Deactivate textbox when clicking canvas
                    
                    if event.button == 1:  # Left click
                        if self.current_tool in ['rect', 'filled_rect', 'circle', 'filled_circle', 'line', 'door', 'ramp', 'elevator']:
                            map_pos = self.screen_to_map(mouse_pos[0], mouse_pos[1])
                            if map_pos:
                                self.shape_start = map_pos
                        else:
                            self.is_drawing = True
                            self.handle_drawing(mouse_pos)
                    
                    elif event.button == 2:  # Middle click - pan
                        self.dragging_view = True
                        self.drag_start = mouse_pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if self.current_tool in ['rect', 'filled_rect', 'circle', 'filled_circle', 'line', 'door', 'ramp', 'elevator']:
                        self.finalize_shape()
                    self.is_drawing = False
                    self.last_draw_pos = None
                elif event.button == 2:
                    self.dragging_view = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.is_drawing:
                    self.handle_drawing(event.pos)
                elif self.dragging_view and self.drag_start:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    self.offset_x += dx
                    self.offset_y += dy
                    self.drag_start = event.pos
                    self.canvas_dirty = True
                elif self.shape_start:
                    self.handle_drawing(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                # Handle text input for active textbox
                if self.active_textbox:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        # Apply changes and deactivate
                        self._apply_textbox_value(self.active_textbox)
                        self.active_textbox = None
                    elif event.key == pygame.K_BACKSPACE:
                        self.textbox_content[self.active_textbox] = self.textbox_content[self.active_textbox][:-1]
                    elif event.key == pygame.K_TAB:
                        # Switch to next textbox
                        self._apply_textbox_value(self.active_textbox)
                        textboxes = ['dept_id', 'destinations']
                        current_idx = textboxes.index(self.active_textbox)
                        self.active_textbox = textboxes[(current_idx + 1) % len(textboxes)]
                    elif event.unicode and event.unicode.isprintable():
                        # Add character
                        if self.active_textbox == 'dept_id':
                            # Only allow digits for dept_id
                            if event.unicode.isdigit() and len(self.textbox_content[self.active_textbox]) < 5:
                                self.textbox_content[self.active_textbox] += event.unicode
                        else:
                            # Allow alphanumeric, commas, colons, underscores for destinations
                            if len(self.textbox_content[self.active_textbox]) < 100:
                                self.textbox_content[self.active_textbox] += event.unicode
                else:
                    # Normal keyboard shortcuts
                    if event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.save_map('map_data.bin')
                    elif event.key == pygame.K_o and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.load_map('map_data.bin')
                    elif event.key == pygame.K_n and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.new_map()
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self.brush_size = min(self.brush_size + 1, 20)
                    elif event.key == pygame.K_MINUS:
                        self.brush_size = max(self.brush_size - 1, 1)
                    elif event.key == pygame.K_UP:
                        self.current_floor += 1
                    elif event.key == pygame.K_DOWN:
                        self.current_floor = max(0, self.current_floor - 1)
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.current_cost = max(0, self.current_cost - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.current_cost = min(100, self.current_cost + 1)
                    elif event.key == pygame.K_COMMA:
                        self.current_dept_id = max(0, self.current_dept_id - 1)
                        self.textbox_content['dept_id'] = str(self.current_dept_id)
                    elif event.key == pygame.K_PERIOD:
                        self.current_dept_id = min(255, self.current_dept_id + 1)
                        self.textbox_content['dept_id'] = str(self.current_dept_id)
    
    def _apply_textbox_value(self, textbox_id):
        """Apply the value from textbox to current settings"""
        if textbox_id == 'dept_id':
            try:
                value = int(self.textbox_content['dept_id']) if self.textbox_content['dept_id'] else 0
                self.current_dept_id = max(0, min(255, value))
                self.textbox_content['dept_id'] = str(self.current_dept_id)
            except ValueError:
                self.textbox_content['dept_id'] = str(self.current_dept_id)
        elif textbox_id == 'destinations':
            # Parse comma-separated destinations
            dest_str = self.textbox_content['destinations'].strip()
            if dest_str:
                # Split by comma and clean up each destination
                self.current_destinations = [d.strip() for d in dest_str.split(',') if d.strip()]
            else:
                self.current_destinations = []
        elif textbox_id == 'x_start':
            x_val = self.textbox_content['x_start'].strip()
            self.x_start = int(x_val)
        elif textbox_id == 'y_start':
            y_val = self.textbox_content['y_start'].strip()
            self.y_start = int(y_val)
    
    def render_canvas(self):
        """Render the map canvas with optimizations"""
        # Calculate visible map area
        top_left_world = self.screen_to_world(0, 0)
        bottom_right_world = self.screen_to_world(self.canvas_width, self.canvas_height)
        
        min_x = max(0, int(top_left_world[0]))
        min_y = max(0, int(top_left_world[1]))
        max_x = min(self.map_width, int(bottom_right_world[0]) + 2)
        max_y = min(self.map_height, int(bottom_right_world[1]) + 2)
        
        visible_area = (min_x, min_y, max_x, max_y)
        
        # Only redraw if canvas is dirty or view changed
        if self.canvas_dirty:
            self.canvas_surface.fill(self.COLORS['background'])
            
            pixel_size = max(1, int(self.zoom))
            visible_pixels = (max_x - min_x) * (max_y - min_y)
            
            if visible_pixels > 100000000 and pixel_size <= 2:
                # Ultra-optimized for very large visible areas and low zoom
                # Use pygame.surfarray for direct pixel access
                try:
                    # Extract visible portion of color array
                    visible_colors = self.color_array[min_y:max_y, min_x:max_x]
                    
                    # Create a surface from the numpy array
                    if pixel_size == 1:
                        temp_surf = pygame.surfarray.make_surface(visible_colors.swapaxes(0, 1))
                        dest_x, dest_y = self.world_to_screen(min_x, min_y)
                        self.canvas_surface.blit(temp_surf, (dest_x, dest_y))
                    else:
                        # Need to scale - use transform.scale
                        temp_surf = pygame.surfarray.make_surface(visible_colors.swapaxes(0, 1))
                        scaled_width = (max_x - min_x) * pixel_size
                        scaled_height = (max_y - min_y) * pixel_size
                        scaled_surf = pygame.transform.scale(temp_surf, (scaled_width, scaled_height))
                        dest_x, dest_y = self.world_to_screen(min_x, min_y)
                        self.canvas_surface.blit(scaled_surf, (dest_x, dest_y))
                except:
                    # Fallback to standard rendering if surfarray fails
                    self._render_standard(min_x, min_y, max_x, max_y, pixel_size)
            
            elif pixel_size >= 3:
                # High zoom - draw individual pixels with rect
                for y in range(min_y, max_y):
                    for x in range(min_x, max_x):
                        color = tuple(self.color_array[y, x])
                        sx, sy = self.world_to_screen(x, y)
                        pygame.draw.rect(self.canvas_surface, color, 
                                       (sx, sy, pixel_size, pixel_size))
            
            else:
                # Standard rendering for medium zoom
                self._render_standard(min_x, min_y, max_x, max_y, pixel_size)
            
            # Draw grid if zoomed in
            if self.zoom > 4:
                grid_color = self.COLORS['grid']
                grid_step = max(1, int(10 / self.zoom))
                
                # Draw vertical lines
                for x in range(min_x, max_x + 1, grid_step):
                    sx, _ = self.world_to_screen(x, 0)
                    if 0 <= sx < self.canvas_width:
                        sy1 = max(0, self.world_to_screen(0, min_y)[1])
                        sy2 = min(self.canvas_height, self.world_to_screen(0, max_y)[1])
                        pygame.draw.line(self.canvas_surface, grid_color, (sx, sy1), (sx, sy2), 1)
                
                # Draw horizontal lines
                for y in range(min_y, max_y + 1, grid_step):
                    _, sy = self.world_to_screen(0, y)
                    if 0 <= sy < self.canvas_height:
                        sx1 = max(0, self.world_to_screen(min_x, 0)[0])
                        sx2 = min(self.canvas_width, self.world_to_screen(max_x, 0)[0])
                        pygame.draw.line(self.canvas_surface, grid_color, (sx1, sy), (sx2, sy), 1)
            
            self.canvas_dirty = False
            self.last_visible_area = visible_area
        
        # Blit the canvas to screen
        self.screen.blit(self.canvas_surface, (0, 0))
        
        # Draw temporary shape preview (always on top)
        if self.temp_shape_pixels:
            preview_color = self.COLORS.get(self.current_func_id, self.COLORS[FuncID.WALKABLE])
            pixel_size = max(1, int(self.zoom))
            for x, y in self.temp_shape_pixels:
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    sx, sy = self.world_to_screen(x, y)
                    if 0 <= sx < self.canvas_width and 0 <= sy < self.canvas_height:
                        pygame.draw.rect(self.screen, preview_color, 
                                       (sx, sy, pixel_size, pixel_size))
    
    def _render_standard(self, min_x, min_y, max_x, max_y, pixel_size):
        """Standard rendering method for medium zoom levels"""
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                color = tuple(self.color_array[y, x])
                sx, sy = self.world_to_screen(x, y)
                
                if pixel_size >= 1:
                    pygame.draw.rect(self.canvas_surface, color, 
                                   (sx, sy, pixel_size, pixel_size))
    
    def render_toolbar(self):
        """Render the toolbar"""
        toolbar_rect = pygame.Rect(self.canvas_width, 0, 
                                   self.toolbar_width, self.window_height)
        pygame.draw.rect(self.screen, self.COLORS['ui_bg'], toolbar_rect)
        
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.buttons:
            if button['type'] == 'label':
                text_surf = self.font_medium.render(button['text'], True, self.COLORS['ui_text'])
                self.screen.blit(text_surf, (self.canvas_width + 10, button['y']))
            
            elif button['type'] in ['tool', 'func']:
                rect = button['rect'].copy()
                rect.x += self.canvas_width
                
                # Determine button color
                is_active = (button['type'] == 'tool' and button['id'] == self.current_tool) or \
                           (button['type'] == 'func' and button['id'] == self.current_func_id)
                is_hover = rect.collidepoint(mouse_pos)
                
                if is_active:
                    color = self.COLORS['button_active']
                elif is_hover:
                    color = self.COLORS['button_hover']
                else:
                    color = self.COLORS['button']
                
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                # Draw color indicator for function buttons
                if button['type'] == 'func' and 'color' in button:
                    color_rect = pygame.Rect(rect.x + 5, rect.y + 5, 25, 25)
                    pygame.draw.rect(self.screen, button['color'], color_rect)
                    pygame.draw.rect(self.screen, self.COLORS['ui_text'], color_rect, 1)
                
                # Draw text
                text_surf = self.font_small.render(button['text'], True, self.COLORS['ui_text'])
                text_x = rect.x + (35 if button['type'] == 'func' else 10)
                text_y = rect.centery - text_surf.get_height() // 2
                self.screen.blit(text_surf, (text_x, text_y))
            
            elif button['type'] == 'textbox':
                rect = button['rect'].copy()
                rect.x += self.canvas_width
                
                # Draw label (might be multiline)
                if button.get('multiline_label'):
                    # Draw two-line label
                    label_lines = ['Destinations', '(comma-separated):']
                    for i, line in enumerate(label_lines):
                        label_surf = self.font_small.render(line, True, self.COLORS['ui_text'])
                        self.screen.blit(label_surf, (rect.x, rect.y - 32 + i * 14))
                else:
                    label_surf = self.font_small.render(button['label'], True, self.COLORS['ui_text'])
                    self.screen.blit(label_surf, (rect.x, rect.y - 18))
                
                # Draw textbox background
                is_active = self.active_textbox == button['id']
                bg_color = (80, 80, 80) if is_active else (60, 60, 60)
                pygame.draw.rect(self.screen, bg_color, rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLORS['ui_text'] if is_active else (100, 100, 100), 
                               rect, 2, border_radius=3)
                
                # Draw text content (with scrolling if too long)
                content = self.textbox_content.get(button['id'], '')
                
                # Truncate content if too long to fit
                text_surf = self.font_small.render(content, True, self.COLORS['ui_text'])
                if text_surf.get_width() > rect.width - 16:
                    # Show only the end of the text
                    visible_chars = len(content)
                    while visible_chars > 0:
                        truncated = '...' + content[-visible_chars:]
                        text_surf = self.font_small.render(truncated, True, self.COLORS['ui_text'])
                        if text_surf.get_width() <= rect.width - 16:
                            break
                        visible_chars -= 1
                
                text_rect = text_surf.get_rect(midleft=(rect.x + 8, rect.centery))
                self.screen.blit(text_surf, text_rect)
                
                # Draw cursor if active
                if is_active and self.cursor_visible:
                    cursor_x = min(text_rect.right + 2, rect.right - 5)
                    cursor_y1 = rect.y + 5
                    cursor_y2 = rect.bottom - 5
                    pygame.draw.line(self.screen, self.COLORS['ui_text'], 
                                   (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)
    
    def render_statusbar(self):
        """Render the status bar"""
        statusbar_rect = pygame.Rect(0, self.canvas_height, 
                                     self.window_width, self.statusbar_height)
        pygame.draw.rect(self.screen, self.COLORS['ui_bg'], statusbar_rect)
        
        # Status text
        mouse_pos = pygame.mouse.get_pos()
        map_pos = self.screen_to_map(mouse_pos[0], mouse_pos[1]) if mouse_pos[0] < self.canvas_width else None
        
        if map_pos:
            mx, my = map_pos
            if 0 <= mx < self.map_width and 0 <= my < self.map_height:
                pixel = self.map_data[my][mx]
                dest_str = f' →[{", ".join(pixel.identifier)}]' if pixel.identifier else ''
                status_text = f"Pos: ({mx}, {my}) | Cost: {pixel.cost} | Dept: {pixel.dept_id} | Floor: {pixel.floor} | Type: {FuncID(pixel.func_id).name}{dest_str}"
            else:
                status_text = f"Pos: ({mx}, {my}) | Out of bounds"
        else:
            status_text = f"Zoom: {self.zoom:.2f}x"
        
        dest_display = f' →[{", ".join(self.current_destinations)}]' if self.current_destinations else ''
        status_text += f" | Brush: {self.brush_size} | Setting: Cost={self.current_cost} Dept={self.current_dept_id} Floor={self.current_floor}{dest_display}"
        
        text_surf = self.font_small.render(status_text, True, self.COLORS['ui_text'])
        self.screen.blit(text_surf, (10, self.canvas_height + 5))
    
    def render_legend(self):
        """Render legend overlay"""
        legend_width = 180
        legend_height = 240
        legend_x = 10
        legend_y = 10
        
        # Semi-transparent background
        legend_surface = pygame.Surface((legend_width, legend_height))
        legend_surface.set_alpha(220)
        legend_surface.fill(self.COLORS['ui_bg'])
        self.screen.blit(legend_surface, (legend_x, legend_y))
        
        # Title
        title_surf = self.font_medium.render("Legend", True, self.COLORS['ui_text'])
        self.screen.blit(title_surf, (legend_x + 10, legend_y + 10))
        
        # Function types
        y_offset = legend_y + 40
        for func_id in [FuncID.EMPTY, FuncID.WALKABLE, FuncID.RAMP, FuncID.DOOR, 
                        FuncID.ELEVATOR, FuncID.STAIR, FuncID.OBSTACLE]:
            color = self.COLORS[func_id]
            color_rect = pygame.Rect(legend_x + 10, y_offset, 20, 20)
            pygame.draw.rect(self.screen, color, color_rect)
            pygame.draw.rect(self.screen, self.COLORS['ui_text'], color_rect, 1)
            
            text_surf = self.font_small.render(FuncID(func_id).name, True, self.COLORS['ui_text'])
            self.screen.blit(text_surf, (legend_x + 35, y_offset + 2))
            y_offset += 25

    def render(self):
        """Main render function"""
        self.screen.fill(self.COLORS['background'])
        
        self.render_canvas()
        self.render_toolbar()
        self.render_statusbar()
        self.render_legend()
        
        # Update cursor blink
        self.cursor_timer += 1
        if self.cursor_timer > 30:  # Blink every 30 frames
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
        
        pygame.display.flip()
    
    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
    
    def save_map(self, filename):
        """Save map data to binary file"""
        data = {
            'map_id': self.map_id,
            'width': self.map_width,
            'height': self.map_height,
            'map_data': [[pixel.to_tuple() for pixel in row] for row in self.map_data]
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Map saved to {filename} (ID: {self.map_id})")
    
    def load_map(self, filename):
        """Load map data from binary file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Load map_id if present (backward compatibility)
            self.map_id = data.get('map_id', 'unnamed_map')
            pygame.display.set_caption(f"Map Creator - {self.map_id}")
            
            self.map_width = data['width']
            self.map_height = data['height']
            self.map_data = np.array([[MapPixel.from_tuple(pixel_data) 
                                       for pixel_data in row] 
                                      for row in data['map_data']], dtype=object)
            
            # Rebuild color array
            self.color_array = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
            self._update_color_array()
            
            self.canvas_dirty = True
            print(f"Map loaded from {filename} (ID: {self.map_id})")
        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error loading map: {e}")
    
    def new_map(self, width=None, height=None):
        """Create a new map"""
        if width:
            self.map_width = width
        if height:
            self.map_height = height
        
        self.map_data = np.array([[MapPixel() for _ in range(self.map_width)] 
                                   for _ in range(self.map_height)], dtype=object)
        
        # Rebuild color array
        self.color_array = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        self._update_color_array()
        
        self.canvas_dirty = True
        print(f"New map created: {self.map_width}x{self.map_height}")

def create_map(width=100, height=100, window_width=1200, window_height=800, map_id="unnamed_map"):
    """Helper function to create and run the map creator"""
    creator = MapCreator(width, height, window_width, window_height, map_id)
    creator.run()
    return creator

if __name__ == "__main__":
    # Example usage
    creator = create_map(width=150, height=150, map_id="campus")

import pygame
import math

class Wheelchair:
    def __init__(self, x, y):
        self.pos = [x, y]
        # Direction vector (x, y). Default facing right.
        self.direction = [1, 0]
        
        # --- NEW: Load the navigation marker image ---
        try:
            # Load image and scale it down slightly to fit nicely in a cell
            loaded_img = pygame.image.load("assets/nav_arrow.png").convert_alpha()
            self.original_image = pygame.transform.scale(loaded_img, (30, 30))
        except FileNotFoundError:
            print("Warning: assets/nav_arrow.png not found. Using fallback shape.")
            self.original_image = None
        self.image = self.original_image

    def update_pos(self, new_pos):
        # Calculate direction before updating position if we moved
        if new_pos != self.pos:
            dx = new_pos[0] - self.pos[0]
            dy = new_pos[1] - self.pos[1]
            # Normalize direction if possible
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                self.direction = [dx/length, dy/length]
        
        self.pos = new_pos
        self.rotate_image()

    def rotate_image(self):
        if self.original_image is None: return

        # Calculate angle from direction vector.
        # pygame rotates counter-clockwise, and atan2 returns radians.
        # We assume the default image points UP (negative Y).
        angle_rad = math.atan2(self.direction[1], self.direction[0])
        angle_deg = math.degrees(angle_rad)
        
        # Adjust because 0 degrees in pygame rotate is usually 'up' depending on asset
        # If your asset points UP by default, use: -angle_deg - 90
        # If your asset points RIGHT by default, use: -angle_deg
        # Let's assume the asset points UP:
        final_angle = -angle_deg - 90
        
        self.image = pygame.transform.rotate(self.original_image, final_angle)


    def draw(self, screen, cell_size, offset_x, offset_y):
        pixel_x = offset_x + self.pos[0] * cell_size
        pixel_y = offset_y + self.pos[1] * cell_size
        center_x = pixel_x + cell_size // 2
        center_y = pixel_y + cell_size // 2

        if self.image:
            # Get rect of rotated image and center it on the cell center
            rect = self.image.get_rect(center=(center_x, center_y))
            screen.blit(self.image, rect)
        else:
            # Fallback if image is missing: Draw the old blue circle
            pygame.draw.circle(screen, (0, 122, 255), (center_x, center_y), cell_size // 2 - 2)

    def visible_cells(self, map_data, radius=5):
        # (This method remains unchanged from previous version)
        visible = set()
        cx, cy = self.pos
        rows = len(map_data)
        if rows == 0: return visible
        cols = len(map_data[0])

        for r in range(max(0, cy - radius), min(rows, cy + radius + 1)):
            for c in range(max(0, cx - radius), min(cols, cx + radius + 1)):
                if math.sqrt((r - cy)**2 + (c - cx)**2) <= radius:
                    # Simple line-of-sight check could go here, 
                    # for now just radius check
                    visible.add((c, r))
        return visible
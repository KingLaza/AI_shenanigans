import math
import pygame
from models.state import GameState, PlayerState, Configs, Line
from models.vec2 import Vec2

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT)
        )
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.Font(None, 24)
        
        self.player = PlayerState(
            position=Vec2(100, 100),
            velocity=Vec2(0, 0),
            on_ground=False,
            charge=0
        )
        
        self.was_charging = False
        
        self.lines = [
            Line(Vec2(0, 550), Vec2(300, 550)),
            Line(Vec2(0, 350), Vec2(300, 350)),
            Line(Vec2(300, 550), Vec2(500, 400)),
            Line(Vec2(500, 400), Vec2(700, 400)),
            Line(Vec2(700, 400), Vec2(800, 550)),
            Line(Vec2(0, 550), Vec2(0, 200)),
            Line(Vec2(800, 550), Vec2(800, 200)),
        ]
        self.lines.sort(key=lambda line: abs(line.p1.y - line.p2.y) <= 1)
    
    def get_input(self) -> tuple[int, bool, bool]:
        keys = pygame.key.get_pressed()
        direction = 0
        if keys[pygame.K_LEFT]:
            direction = -1
        elif keys[pygame.K_RIGHT]:
            direction = 1
        charging = keys[pygame.K_SPACE]
        
        just_released = self.was_charging and not charging
        self.was_charging = charging
        
        return direction, charging, just_released
    
    def is_slope(self, line: Line) -> bool:
        return abs(line.p1.y - line.p2.y) > 1
    
    def get_slope_angle(self, line: Line) -> float:
        dx = line.p2.x - line.p1.x
        dy = line.p2.y - line.p1.y        
        return math.atan2(dy, dx)

    def project_velocity_onto_slope(self, line: Line) -> Vec2:
        p = self.player
        
        dx = line.p2.x - line.p1.x
        dy = line.p2.y - line.p1.y
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return Vec2(0, 0)
        
        slope_dir_x = dx / length
        slope_dir_y = dy / length
        
        if slope_dir_y < 0:
            slope_dir_x = -slope_dir_x
            slope_dir_y = -slope_dir_y
        
        dot = p.velocity.x * slope_dir_x + p.velocity.y * slope_dir_y
        gravity_along_slope = Configs.GRAVITY * slope_dir_y
        total_speed = dot + gravity_along_slope
        
        return Vec2(slope_dir_x * total_speed, slope_dir_y * total_speed)

    def collide_with_lines(self):
        p = self.player
        
        p.on_ground = False
        
        for line in self.lines:
            line_y = line.y_at_x(p.position.x)
            if line_y is None:
                continue
            
            if p.position.y >= line_y and p.position.y < line_y + 20:
                if p.velocity.y >= 0:
                    p.position = Vec2(p.position.x, line_y)
                    
                    if self.is_slope(line):
                        p.velocity = self.project_velocity_onto_slope(line)
                        p.on_ground = False
                    else:
                        p.velocity = Vec2(p.velocity.x, 0)
                        p.on_ground = True
                    break
    
    def collide_with_walls(self):
        p = self.player
        half_w = Configs.PLAYER_WIDTH / 2
        
        if p.position.x <= half_w:
            p.position = Vec2(half_w, p.position.y)
            p.velocity = Vec2(-p.velocity.x, p.velocity.y)
        
        if p.position.x >= Configs.VIRTUAL_WIDTH - half_w:
            p.position = Vec2(Configs.VIRTUAL_WIDTH - half_w, p.position.y)
            p.velocity = Vec2(-p.velocity.x, p.velocity.y)
    
    def step(self, direction: int, charging: bool, just_released: bool):
        p = self.player
        
        if not p.on_ground:
            p.velocity = Vec2(p.velocity.x, p.velocity.y + Configs.GRAVITY)
        
        if charging and p.on_ground:
            p.charge = min(p.charge + 1, Configs.MAX_CHARGE)
            p.velocity = Vec2(0, p.velocity.y)
        
        if p.charge == Configs.MAX_CHARGE or (just_released and p.charge > 0 and p.on_ground):
            jump_power = p.charge / Configs.MAX_CHARGE
            p.velocity = Vec2(
                direction * Configs.WALK_SPEED * 2 * jump_power,
                -Configs.MAX_CHARGE * jump_power * 0.8
            )
            p.charge = 0
            p.on_ground = False
        
        if not charging and p.on_ground and p.charge == 0:
            p.velocity = Vec2(direction * Configs.WALK_SPEED, p.velocity.y)
        
        if not p.on_ground and not charging:
            new_vx = p.velocity.x + direction * 0.3
            new_vx = max(-10, min(10, new_vx))
            p.velocity = Vec2(new_vx, p.velocity.y)
        
        p.position = Vec2(p.position.x + p.velocity.x, p.position.y + p.velocity.y)
        
        self.collide_with_lines()
        self.collide_with_walls()
    
    def render_debug(self):
        p = self.player
        debug_lines = [
            f"pos: ({p.position.x:.1f}, {p.position.y:.1f})",
            f"vel: ({p.velocity.x:.2f}, {p.velocity.y:.2f})",
            f"on_ground: {p.on_ground}",
            f"charge: {p.charge}/{Configs.MAX_CHARGE}",
            f"lines: {len(self.lines)}",
        ]
        
        y = 10
        for line in debug_lines:
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, y))
            y += 20
    
    def render(self):
        self.screen.fill((40, 40, 40))
        
        for line in self.lines:
            color = (200, 100, 100) if self.is_slope(line) else (100, 200, 100)
            pygame.draw.line(
                self.screen,
                color,
                (line.p1.x, line.p1.y),
                (line.p2.x, line.p2.y),
                3
            )
        
        p = self.player
        half_w = Configs.PLAYER_WIDTH / 2
        draw_x = p.position.x - half_w
        draw_y = p.position.y - Configs.PLAYER_HEIGHT
        pygame.draw.rect(
            self.screen,
            (100, 150, 255),
            (draw_x, draw_y, Configs.PLAYER_WIDTH, Configs.PLAYER_HEIGHT)
        )
        
        if p.charge > 0:
            bar_width = (p.charge / Configs.MAX_CHARGE) * Configs.PLAYER_WIDTH
            pygame.draw.rect(
                self.screen,
                (255, 80, 80),
                (draw_x, draw_y - 10, bar_width, 5)
            )
        
        pygame.draw.circle(self.screen, (255, 255, 0), (int(p.position.x), int(p.position.y)), 4)
        
        self.render_debug()
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            direction, charging, just_released = self.get_input()
            self.step(direction, charging, just_released)
            self.render()
            self.clock.tick(Configs.FPS)
        
        pygame.quit()

if __name__ == "__main__":
    Game().run()
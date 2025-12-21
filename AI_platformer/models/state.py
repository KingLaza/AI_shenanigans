

from __future__ import annotations
from dataclasses import dataclass
from pygame import Vector2

from .vec2 import Vec2

class Configs:
    MIN_CHARGE = 1
    MAX_CHARGE = 60
    MAX_JUMP_HEIGHT = 100       #not really used
    WALK_SPEED = 4  # could be more
    MOVE_COUNT = 20
    FPS = 60
    GRAVITY = (9.81 / FPS) * 4      #it's originally * 3
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    VIRTUAL_WIDTH = 800
    VIRTUAL_HEIGHT = 600
    START_FULLSCREEN = False
    CURRENT_MAP = "00"
    LINE_WIDTH = 2
    PLAYER_WIDTH = 32
    PLAYER_HEIGHT = 48

@dataclass(slots=True)
class PlayerState:
    position: Vec2
    velocity: Vec2
    on_ground: bool
    charge: int
    
    @property
    def screen_pos(self) -> Vector2:
        return Vector2(self.position.x, self.position.y % Configs.VIRTUAL_HEIGHT)
    
    def copy(self) -> "PlayerState":
        return PlayerState(
            position=self.position,
            velocity=self.velocity,
            on_ground=self.on_ground,
            charge=self.charge
        )

@dataclass(slots=True)
class GameState:
    players: dict[int, PlayerState]
    tick: int
    
    def copy(self) -> GameState:
        return GameState(
            players={k: v.copy() for k, v in self.players.items()},
            tick=self.tick
        )
    
@dataclass
class Line:
    p1: Vec2
    p2: Vec2
    
    def y_at_x(self, x: float) -> float | None:
        """Get Y position on this line at given X (if within segment bounds)"""
        if self.p2.x == self.p1.x:  
            return None
        if not (min(self.p1.x, self.p2.x) <= x <= max(self.p1.x, self.p2.x)):
            return None
        t = (x - self.p1.x) / (self.p2.x - self.p1.x)
        return self.p1.y + t * (self.p2.y - self.p1.y)
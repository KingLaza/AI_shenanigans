from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float
    
    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)
    
    def scale(self, s: float) -> Vec2:
        return Vec2(self.x * s, self.y * s)
    
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self) -> Vec2:
        l = self.length()
        if l < 0.0001:
            return Vec2(0, 0)
        return Vec2(self.x / l, self.y / l)
    
    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)
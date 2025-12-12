#from pygame.examples.moveit import WIDTH, HEIGHT
from .configs import Configs

WIDTH, HEIGHT = Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT

print(WIDTH, HEIGHT)
LINES = [
    # (30, HEIGHT-60, WIDTH-30, HEIGHT-60),
    # (30, 60, WIDTH-30, 60),
    # (30, HEIGHT-60, 30, 60),
    # (WIDTH-30, HEIGHT-60, WIDTH-30, 60)
]

TYPE_PRIORITY = {
    "vertical": 2,
    "diagonal": 1,
    "horizontal": 0
}

_BASE_W = 800
_BASE_H = 600

_RAW_POINTS = [
    {"x": 400, "y": 480},
    {"x": 150,  "y": 70},
    {"x": 500, "y": 280},
    {"x": 300, "y": 300},
    {"x": 280, "y": 30},
    {"x": 600, "y": 70},
    {"x": 100, "y": 100},
    {"x": 100, "y": 100},
    {"x": 100, "y": 100},
]

POINTS = [
    (p["x"] / _BASE_W * WIDTH, p["y"] / _BASE_H * HEIGHT)
    for p in _RAW_POINTS
]
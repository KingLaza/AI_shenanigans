#from pygame.examples.moveit import WIDTH, HEIGHT
from .configs import Configs

WIDTH, HEIGHT = Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT

print(WIDTH, HEIGHT)
LINES = [
    (30, HEIGHT-60, WIDTH-30, HEIGHT-60),
    (30, 60, WIDTH-30, 60),
    #(30, 180, WIDTH+100, 180),          #and this
    (30, HEIGHT-60, 30, 60),
    #(150, HEIGHT+60, 150, 60),              #added now
    (WIDTH-30, HEIGHT-60, WIDTH-30, 60)
]

TYPE_PRIORITY = {
    "vertical": 2,
    "diagonal": 1,
    "horizontal": 0
}
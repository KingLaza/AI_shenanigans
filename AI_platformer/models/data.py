from pygame.examples.moveit import WIDTH, HEIGHT
print(WIDTH, HEIGHT)
LINES = [
    (30, HEIGHT+60, WIDTH+100, HEIGHT+60),
    (30, 60, WIDTH+100, 60),
    #(30, 180, WIDTH+100, 180),          #and this
    (30, HEIGHT+60, 30, 60),
    #(150, HEIGHT+60, 150, 60),              #added now
    (WIDTH+100, HEIGHT+60, WIDTH+100, 60)
]
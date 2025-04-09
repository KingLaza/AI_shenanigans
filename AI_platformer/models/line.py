from pygame import Vector2


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.start = Vector2(self.x1, self.y1)
        self.end = Vector2(self.x2, self.y2)
        self.type = self.get_type(x1,y1, x2,y2)

    def get_type(self, x1, y1, x2, y2):
        if y1 == y2:
            return "horizontal"
        if x1 == x2:
            return "vertical"
        if y1 == y1 and x2 == x1:
            return "ERROR CAN NOT HAPPEN FIX IT OR GAME WILL EXPLODE"
        else:
            return "diagonal"

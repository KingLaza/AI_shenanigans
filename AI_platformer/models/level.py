from .line import Line
from .configs import Configs
import pygame
import json

class Level:
    def __init__(self, level_number):
        self.level_number = level_number
        self.lines = []
        self.bg_picture = pygame.image      # I guess
        self.bg_picture_scaled = pygame.image

    def load_assets(self):
        self.load_lines()
        self.load_picture()

    def load_lines(self):
        try:
            with open("lines/lines_"+Configs.CURRENT_MAP+"_"+self.level_number+".json", "r") as f:
                lines_data = json.load(f)
                for data in lines_data:
                    line = Line(data["x1"], data["y1"], data["x2"], data["y2"])
                    self.lines.append(line)
                print("lines found: ", len(self.lines))
        except FileNotFoundError:
            print("No saved lines found. Or wrong file_name bro")

    def load_picture(self):
        self.bg_picture = pygame.image.load("pictures/jk_"+ Configs.CURRENT_MAP+"_"+ self.level_number + ".png").convert()  # replace with your image
        self.bg_picture_scaled = pygame.transform.scale(self.bg_picture, (Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT))

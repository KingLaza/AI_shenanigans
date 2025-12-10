from pygame import Vector2

from models import Game
from models import Configs
from models import Player

if __name__ == '__main__':

    WIDTH, HEIGHT = Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT
    player_start_position = Vector2(WIDTH // 2, HEIGHT - 120)
    game = Game()

    #game.add_player(Player(position=Vector2(WIDTH // 2, HEIGHT - 120)))

    #testing out the new stuff:
    # Quick test - put this in a test file:
    from genetic_algorithm import Genome, Population

    genome = Genome.random()
    print(f"Created genome with {len(genome.moves)} moves!")
    


    game.add_cpu_players(5, player_start_position)
    #game.run(False)
    game.runTest()          #this game.runTest() is made for debugging and collision checking etc.
from pygame import Vector2

from models import Game

if __name__ == '__main__':
    WIDTH, HEIGHT = 800, 600
    player_start_position = Vector2(WIDTH // 2, HEIGHT - 120)
    game = Game()

    #game.add_player(Player(position=Vector2(WIDTH // 2, HEIGHT - 60)))

    game.add_cpu_players(100, player_start_position)
    game.run()
    #game.runTest()          #this game.runTest() is made for debugging and collision checking etc.






# import pygame
#
# # Initialize PyGame
# pygame.init()
#
# # Screen settings
# WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Jump King Prototype")
#
# # Colors
# WHITE = (255, 255, 255)
# BLUE = (0, 0, 255)
# BLACK = (0, 0, 0)
#
# # Player settings
# player_width, player_height = 30, 60
# player_x = WIDTH // 2
# player_y = HEIGHT - 60  # Above ground
# velocity_x = 0
# velocity_y = 0
# gravity = 0.5
# max_jump_strength = -15  # Max power
# min_jump_strength = -5  # Minimum jump power
# jump_charge_speed = 0.3  # Charge increase per frame
# jump_charge = 0
# jumping = False
# on_ground = False
# charging_jump = False
# jump_direction = "up"  # Default jump direction
#
# # Movement settings
# walk_speed = 3
#
# # Define collision lines (currently just the bottom ground line)
# collision_lines = [((0, HEIGHT - 30), (WIDTH, HEIGHT - 30))]
#
# # Game loop
# running = True
# clock = pygame.time.Clock()
#
# while running:
#     screen.fill(WHITE)  # Clear screen
#
#     # Event handling
#     keys = pygame.key.get_pressed()
#
#     # Walking (only allowed if NOT charging a jump)
#     if not charging_jump and on_ground:
#         if keys[pygame.K_LEFT]:
#             player_x -= walk_speed
#         if keys[pygame.K_RIGHT]:
#             player_x += walk_speed
#
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#
#         # Start charging jump
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_SPACE and on_ground:
#                 jump_charge = 0  # Start charging
#                 charging_jump = True
#                 jump_direction = "up"  # Reset to straight jump
#
#         # Release jump
#         if event.type == pygame.KEYUP:
#             if event.key == pygame.K_SPACE and charging_jump:
#                 # Determine jump direction only at the moment of release
#                 if keys[pygame.K_LEFT]:
#                     jump_direction = "left"
#                 elif keys[pygame.K_RIGHT]:
#                     jump_direction = "right"
#
#                 # Calculate jump force
#                 jump_force = min_jump_strength + (jump_charge / abs(max_jump_strength)) * (max_jump_strength - min_jump_strength)
#                 velocity_y = jump_force
#
#                 if jump_direction == "left":
#                     velocity_x = -5
#                 elif jump_direction == "right":
#                     velocity_x = 5
#                 else:
#                     velocity_x = 0  # Jump straight up
#
#                 charging_jump = False
#                 on_ground = False  # Player leaves the ground
#
#     # Charge jump if space is held
#     if charging_jump:
#         jump_charge += jump_charge_speed
#         if jump_charge >= abs(max_jump_strength):  # Auto-jump when fully charged
#             if keys[pygame.K_LEFT]:
#                 jump_direction = "left"
#             elif keys[pygame.K_RIGHT]:
#                 jump_direction = "right"
#
#             velocity_y = max_jump_strength
#             if jump_direction == "left":
#                 velocity_x = -5
#             elif jump_direction == "right":
#                 velocity_x = 5
#             else:
#                 velocity_x = 0
#
#             charging_jump = False
#             on_ground = False
#
#     # Apply gravity
#     velocity_y += gravity
#     player_y += velocity_y
#     player_x += velocity_x
#
#     # Collision detection with bottom line
#     ground_y = HEIGHT - 30 - player_height  # Adjust for player height
#     if player_y >= ground_y:
#         player_y = ground_y
#         velocity_y = 0
#         velocity_x = 0  # Stop horizontal movement on landing
#         on_ground = True  # Player lands
#
#     # Draw ground line
#     for line in collision_lines:
#         pygame.draw.line(screen, BLACK, line[0], line[1], 3)
#
#     # Draw player
#     pygame.draw.rect(screen, BLUE, (player_x, player_y, player_width, player_height))
#
#     pygame.display.update()
#     clock.tick(60)
#
# pygame.quit()

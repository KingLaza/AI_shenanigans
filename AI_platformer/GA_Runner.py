"""
Simple Jump King Genetic Algorithm

This works directly with your existing game and player classes.
No overhead, no bridges, just pure evolution using what you already built.

TWO GAME MODES:
===============
1. Level Solver: Find best moveset for each level individually
2. Full Evolution: Evolve over many generations to reach maximum height
"""

import random
import json
import time
from typing import List, Dict, Tuple
from pygame.math import Vector2

# Import your existing game components
from models import Game, Player, Configs, Move


class SimpleGA:
    """
    Simple Genetic Algorithm that works directly with your existing game.

    Uses your Player class, your Game class, everything you already built.
    No unnecessary abstractions!
    """

    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.game = Game()  # Use your existing game
        self.best_movesets = {}  # Store best movesets per level

    def create_random_moveset(self) -> List[Move]:
        """Create a random moveset using your existing Move class."""
        return [Move() for _ in range(Configs.MOVE_COUNT)]

    def load_best_result(self, filename: str = 'best_evolution_result.json') -> List[Move]:
        """
        Load the best moveset from a previous evolution.

        Returns:
            Best moveset if file exists, None otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Convert back to Move objects
            best_moveset = []
            for move_data in data['moveset']:
                move = Move()
                move.move_type = move_data['move_type']
                move.strength = move_data['strength']
                move.initial_strength = move_data['strength']
                best_moveset.append(move)

            print(f"📂 Loaded best result: Height {data['best_height']:.1f}, Fitness {data['best_fitness']:.1f}")
            return best_moveset

        except FileNotFoundError:
            print(f"📂 No previous results found in {filename}")
            return None
        except Exception as e:
            print(f"📂 Error loading {filename}: {e}")
            return None

    def create_seeded_population(self, population_size: int, seed_moveset: List[Move] = None,
                               seed_percentage: float = 0.2) -> List[List[Move]]:
        """
        Create a population with some individuals seeded from a good moveset.

        Args:
            population_size: Total population size
            seed_moveset: Best moveset from previous evolution
            seed_percentage: What fraction of population should be based on seed (0.0-1.0)

        Returns:
            List of movesets with some seeded and others random
        """
        population = []

        if seed_moveset is not None:
            # Calculate how many to seed
            num_seeded = max(1, int(population_size * seed_percentage))

            print(f"🌱 Seeding {num_seeded}/{population_size} individuals from previous best")

            # Add the original best
            population.append([self._copy_move(move) for move in seed_moveset])

            # Add mutated versions of the best
            for _ in range(num_seeded - 1):
                mutated = self.mutate_moveset(seed_moveset, mutation_rate=0.3)  # Higher mutation for diversity
                population.append(mutated)

        # Fill rest with random individuals
        while len(population) < population_size:
            population.append(self.create_random_moveset())

        return population

    def _copy_move(self, original_move: Move) -> Move:
        """Create a deep copy of a Move object."""
        new_move = Move()
        new_move.move_type = original_move.move_type
        new_move.strength = original_move.strength
        new_move.initial_strength = original_move.initial_strength
        return new_move

    def mutate_moveset(self, moves: List[Move], mutation_rate: float = 0.1) -> List[Move]:
        """Mutate a moveset by randomly changing some moves."""
        mutated = []
        for move in moves:
            if random.random() < mutation_rate:
                # Create new random move
                mutated.append(Move())
            else:
                # Keep existing move
                new_move = Move()
                new_move.move_type = move.move_type
                new_move.strength = move.strength
                new_move.initial_strength = move.initial_strength
                mutated.append(new_move)
        return mutated

    def crossover_movesets(self, parent1: List[Move], parent2: List[Move]) -> Tuple[List[Move], List[Move]]:
        """Combine two movesets to create offspring."""
        if len(parent1) != len(parent2):
            return self.create_random_moveset(), self.create_random_moveset()

        # Two-point crossover
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1, len(parent1) - 1)

        # Create offspring
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        # Copy the moves properly
        child1_moves = []
        child2_moves = []

        for move in child1:
            new_move = Move()
            new_move.move_type = move.move_type
            new_move.strength = move.strength
            new_move.initial_strength = move.initial_strength
            child1_moves.append(new_move)

        for move in child2:
            new_move = Move()
            new_move.move_type = move.move_type
            new_move.strength = move.strength
            new_move.initial_strength = move.initial_strength
            child2_moves.append(new_move)

        return child1_moves, child2_moves

    def evaluate_moveset(self, moves: List[Move], target_level: int = None) -> Dict:
        """
        Evaluate a single moveset quickly (no graphics).
        Used by Level Solver mode.
        """
        # Clear any existing players
        self.game.players.clear()

        # Create player with this moveset at starting position
        start_pos = Vector2(Configs.VIRTUAL_WIDTH // 2, Configs.VIRTUAL_HEIGHT - 120)
        player = Player(start_pos)
        player.moves = moves
        player.curr_move = player.moves[0] if moves else None
        player.current_charge = player.curr_move.initial_strength if player.curr_move else 0

        # Add to game
        self.game.add_player(player)
        self.game.current_level = 0

        # Run simulation (fast, no graphics)
        max_steps = 1800
        start_time = time.time()

        for step in range(max_steps):
            # Update game
            for p in self.game.players:
                p.play_move()
                p.apply_gravity()
                p.update_position()

            self.game.collision_handler()

            # Check if player finished all moves
            if player.curr_move_count >= Configs.MOVE_COUNT:
                break

            # Check if player is stuck
            if step > 600 and player.position.y < 50:
                break

        end_time = time.time()

        # Calculate performance metrics
        performance = {
            'max_height': player.position.y,
            'final_height': player.position.y,
            'max_level': player.current_level,
            'moves_used': player.curr_move_count,
            'simulation_time': end_time - start_time,
            'completed_all_moves': player.curr_move_count >= Configs.MOVE_COUNT
        }

        # Calculate fitness score
        fitness = self.calculate_fitness(performance, target_level)
        performance['fitness'] = fitness

        # Clean up
        self.game.players.clear()

        return performance

    def evaluate_population(self, population_movesets: List[List[Move]], show_visual: bool = False) -> List[Dict]:
        """
        Evaluate an entire population of movesets.

        Args:
            population_movesets: List of movesets to evaluate
            show_visual: If True, show all AI players playing simultaneously

        Returns:
            List of performance data for each moveset
        """
        if show_visual:
            return self._evaluate_population_visual(population_movesets)
        else:
            return self._evaluate_population_fast(population_movesets)

    def _evaluate_population_fast(self, population_movesets: List[List[Move]]) -> List[Dict]:
        """Fast evaluation - test each AI individually (no graphics)"""
        performances = []

        for moveset in population_movesets:
            performance = self.evaluate_moveset(moveset, show_visual=False)
            performances.append(performance)

        return performances

    def _evaluate_population_visual(self, population_movesets: List[List[Move]]) -> List[Dict]:
        """
        Visual evaluation - all AI players play simultaneously!

        This is the cool part - you see the entire population playing together.
        """
        print(f"   🎮 Visual simulation: {len(population_movesets)} AI players playing together!")

        # Clear any existing players
        self.game.players.clear()

        # Create ALL AI players and add them to the game at once
        players = []
        start_pos = Vector2(Configs.VIRTUAL_WIDTH // 2, Configs.VIRTUAL_HEIGHT - 120)

        for i, moveset in enumerate(population_movesets):
            # Spread players out in a grid pattern for better visibility
            row = i // 10  # 10 players per row
            col = i % 10
            offset_x = (col - 4.5) * 15  # Center around start position, 15 pixels apart
            offset_y = row * 20  # 20 pixels between rows

            player_pos = Vector2(
                start_pos.x + offset_x,
                start_pos.y + offset_y
            )

            # Create player with this moveset
            player = Player(player_pos)
            player.moves = moveset
            player.curr_move = player.moves[0] if moveset else None
            player.current_charge = player.curr_move.initial_strength if player.curr_move else 0

            # Add to game and track
            self.game.add_player(player)
            players.append(player)

        self.game.current_level = 0

        # Run simulation with ALL players
        max_steps = 1800  # 30 seconds at 60 FPS
        start_time = time.time()

        print(f"   👀 Showing {len(players)} AI players (Press ESC to skip, SPACE to pause)")

        paused = False
        for step in range(max_steps):
            # Handle events
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Calculate final results quickly and return
                    return self._calculate_final_performances(players, start_time)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Skip this simulation
                        return self._calculate_final_performances(players, start_time)
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print("   ⏸️ Paused" if paused else "   ▶️ Resumed")

            if not paused:
                # Update ALL players simultaneously
                for player in players:
                    player.play_move()
                    player.apply_gravity()
                    player.update_position()

                # Handle collisions for all players
                self.game.collision_handler()

            # Render the game with all players visible
            self.game.render()
            self.game.clock.tick(Configs.FPS)  # Run at normal game speed

            # Check if all players finished their moves
            all_finished = all(p.curr_move_count >= Configs.MOVE_COUNT for p in players)
            if all_finished:
                print(f"   ✅ All AI players finished their moves!")
                break

        end_time = time.time()

        # Calculate individual performance for each player
        performances = self._calculate_final_performances(players, start_time)

        # Clean up
        self.game.players.clear()

        return performances

    def _calculate_final_performances(self, players: List, start_time: float) -> List[Dict]:
        """Calculate performance metrics for all players after simulation"""
        end_time = time.time()
        performances = []

        for player in players:
            performance = {
                'max_height': player.position.y,
                'final_height': player.position.y,
                'max_level': player.current_level,
                'moves_used': player.curr_move_count,
                'simulation_time': end_time - start_time,
                'completed_all_moves': player.curr_move_count >= Configs.MOVE_COUNT
            }

            # Calculate fitness (no target level for population evaluation)
            fitness = self.calculate_fitness(performance, target_level=None)
            performance['fitness'] = fitness

            performances.append(performance)

        return performances

    def calculate_fitness(self, performance: Dict, target_level: int = None) -> float:
        """
        Simple fitness calculation focusing on what matters:
        1. Height reached (most important)
        2. Time efficiency (bonus for reaching height faster)
        """
        height = performance['max_height']
        moves_used = performance['moves_used']
        max_level = performance['max_level']

        # Primary score: height reached
        height_score = height

        # Bonus for reaching target level (if specified)
        level_bonus = 0
        if target_level is not None and max_level >= target_level:
            level_bonus = 500  # Big bonus for reaching target

        # Small efficiency bonus (use fewer moves = better)
        efficiency_bonus = max(0, (Configs.MOVE_COUNT - moves_used) * 2)

        fitness = height_score + level_bonus + efficiency_bonus

        return fitness


class LevelSolverMode:
    """
    Game Mode 1: Find the best moveset for each level individually.

    This runs simulations for each level and saves the best moveset for that level.
    """

    def __init__(self, ga: SimpleGA):
        self.ga = ga
        self.level_solutions = {}

    def solve_level(self, level_number: int, num_simulations: int = 100, show_simulations: int = 0) -> Dict:
        """
        Find the best moveset for a specific level.

        Args:
            level_number: Which level to solve (0, 1, 2, ...)
            num_simulations: How many attempts to make
            show_simulations: How many simulations to show visually (0 = none)

        Returns:
            Best performance data and moveset for this level
        """
        print(f"🎯 Solving Level {level_number} with {num_simulations} simulations...")
        if show_simulations > 0:
            print(f"   👀 Will show {show_simulations} simulations visually")

        # Calculate which simulations to show
        show_indices = set()
        if show_simulations > 0:
            if show_simulations >= num_simulations:
                show_indices = set(range(num_simulations))  # Show all
            else:
                # Show evenly spaced simulations (like 0, 50, 100 from user's example)
                if show_simulations == 1:
                    show_indices = {0}  # Just show first simulation
                elif show_simulations == 2:
                    show_indices = {0, num_simulations - 1}  # First and last
                else:
                    # Evenly space them: first, middle(s), last
                    step = (num_simulations - 1) / (show_simulations - 1)
                    show_indices = set(round(i * step) for i in range(show_simulations))

        best_performance = None
        best_moveset = None

        # Generate and test movesets
        for sim in range(num_simulations):
            should_show = sim in show_indices

            if sim % 20 == 0 or should_show:
                status = " 👀 VISUAL" if should_show else ""
                print(f"   Simulation {sim + 1}/{num_simulations}{status}")

            # Create random moveset
            moveset = self.ga.create_random_moveset()

            # Test it (with or without visuals)
            if should_show:
                print(f"   👀 Visual simulation {sim + 1}")
                # For individual visual simulation, we use a temporary population of 1
                performance = self.ga.evaluate_population([moveset], show_visual=True)[0]
            else:
                performance = self.ga.evaluate_moveset(moveset, target_level=level_number)

            # Keep track of best
            if best_performance is None or performance['fitness'] > best_performance['fitness']:
                best_performance = performance
                best_moveset = moveset.copy()

                print(f"   🏆 New best! Height: {performance['max_height']:.1f}, "
                      f"Level: {performance['max_level']}, Fitness: {performance['fitness']:.1f}")

        # Save solution
        self.level_solutions[level_number] = {
            'moveset': best_moveset,
            'performance': best_performance
        }

        return best_performance

    def solve_all_levels(self, max_level: int = 3, simulations_per_level: int = 100, show_simulations: int = 0):
        """Solve all levels from 0 to max_level."""
        print(f"🚀 Solving levels 0 to {max_level}")

        for level in range(max_level + 1):
            self.solve_level(level, simulations_per_level, show_simulations)

        print(f"\n✅ All levels solved!")
        self.save_solutions()

    def save_solutions(self, filename: str = "level_solutions.json"):
        """Save all level solutions to file."""
        solutions_data = {}

        for level, solution in self.level_solutions.items():
            # Convert moveset to saveable format
            moveset_data = []
            for move in solution['moveset']:
                moveset_data.append({
                    'move_type': move.move_type,
                    'strength': move.strength
                })

            solutions_data[str(level)] = {
                'moveset': moveset_data,
                'performance': solution['performance']
            }

        with open(filename, 'w') as f:
            json.dump(solutions_data, f, indent=2)

        print(f"💾 Solutions saved to {filename}")


class FullEvolutionMode:
    """
    Game Mode 2: Full genetic algorithm evolution over many generations.

    This evolves a population over time to find the absolute best performers.
    """

    def __init__(self, ga: SimpleGA):
        self.ga = ga
        self.generation = 0
        self.best_ever = None

    def evolve(self, num_generations: int = 20, mutation_rate: float = 0.1, show_simulations: int = 0,
               use_previous_best: bool = False, seed_percentage: float = 0.2) -> Dict:
        """
        Run full evolution for specified generations.

        Args:
            num_generations: How many generations to evolve
            mutation_rate: Probability of mutating each move
            show_simulations: How many simulations to show visually per generation (0 = none)
            use_previous_best: Whether to seed population with previous best result
            seed_percentage: What fraction of population to seed with previous best

        Returns:
            Best performance data found
        """
        print(f"🧬 Starting evolution for {num_generations} generations...")
        print(f"   Population size: {self.ga.population_size}")
        print(f"   Mutation rate: {mutation_rate}")
        if show_simulations > 0:
            print(f"   👀 Will show {show_simulations} POPULATION simulations per generation")
            print(f"   (Each visual shows all {self.ga.population_size} AI players playing together!)")

        # Generate initial population (with optional seeding)
        if use_previous_best:
            seed_moveset = self.ga.load_best_result()
            population = self.ga.create_seeded_population(
                self.ga.population_size, seed_moveset, seed_percentage
            )
        else:
            print(f"🎲 Creating random initial population...")
            population = []
            for _ in range(self.ga.population_size):
                population.append(self.ga.create_random_moveset())

        for gen in range(num_generations):
            print(f"\n📊 Generation {gen + 1}/{num_generations}")

            # Determine which simulations to show visually
            show_indices = set()
            if show_simulations > 0:
                if show_simulations >= num_generations:
                    # Show all generations
                    show_indices = set(range(num_generations))
                else:
                    # Show evenly spaced generations (like 0, 50, 100 from user's example)
                    if show_simulations == 1:
                        show_indices = {0}  # Just show first generation
                    elif show_simulations == 2:
                        show_indices = {0, num_generations - 1}  # First and last
                    else:
                        # Evenly space them: first, middle(s), last
                        step = (num_generations - 1) / (show_simulations - 1)
                        show_indices = set(round(i * step) for i in range(show_simulations))

            # Should we show this generation?
            should_show = gen in show_indices

            # Evaluate entire population (either visually or fast)
            if should_show:
                print(f"   🎮 VISUAL SIMULATION: All {self.ga.population_size} AI players together!")
                performances = self.ga.evaluate_population(population, show_visual=True)
            else:
                print(f"   ⚡ Fast evaluation of {self.ga.population_size} AI players...")
                performances = self.ga.evaluate_population(population, show_visual=False)

            # Extract fitness scores
            fitness_scores = [p['fitness'] for p in performances]

            # Track best ever
            for i, performance in enumerate(performances):
                if self.best_ever is None or performance['fitness'] > self.best_ever['fitness']:
                    self.best_ever = performance.copy()
                    self.best_ever['moveset'] = population[i].copy()
                    print(f"   🏆 NEW RECORD! Height: {performance['max_height']:.1f}, "
                          f"Level: {performance['max_level']}, Fitness: {performance['fitness']:.1f}")

            # Show generation stats
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
            max_fitness = max(fitness_scores) if fitness_scores else 0
            print(f"   📊 Avg fitness: {avg_fitness:.1f}, Max: {max_fitness:.1f}")

            # Selection and reproduction for next generation
            new_population = []

            # Keep best 10% (elitism)
            elite_count = max(1, self.ga.population_size // 10)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]

            for idx in elite_indices:
                new_population.append(population[idx])

            # Fill rest with offspring
            while len(new_population) < self.ga.population_size:
                # Tournament selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child1, child2 = self.ga.crossover_movesets(parent1, parent2)

                # Mutation
                child1 = self.ga.mutate_moveset(child1, mutation_rate)
                child2 = self.ga.mutate_moveset(child2, mutation_rate)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.ga.population_size]

        print(f"\n🎉 Evolution complete!")
        print(f"Best fitness achieved: {self.best_ever['fitness']:.1f}")
        print(f"Best height reached: {self.best_ever['max_height']:.1f}")

        return self.best_ever

    def tournament_selection(self, population: List, fitness_scores: List, tournament_size: int = 3):
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]


# Easy-to-use runner functions
def run_level_solver(max_level: int = 2, simulations_per_level: int = 100, show_simulations: int = 0):
    """
    Game Mode 1: Solve each level individually.

    Args:
        max_level: Highest level to solve (0, 1, 2, ...)
        simulations_per_level: How many attempts per level
        show_simulations: How many simulations to show visually per level (0 = none)
    """
    print("🎯 LEVEL SOLVER MODE")
    print("=" * 40)

    ga = SimpleGA(population_size=50)
    solver = LevelSolverMode(ga)
    solver.solve_all_levels(max_level, simulations_per_level, show_simulations)


def run_full_evolution(generations: int = 20, population_size: int = 50, show_simulations: int = 0,
                      use_previous_best: bool = False, seed_percentage: float = 0.2):
    """
    Game Mode 2: Full evolution over many generations.

    Args:
        generations: Number of generations to evolve
        population_size: How many AI players per generation
        show_simulations: How many simulations to show visually per generation (0 = none)
        use_previous_best: Whether to seed with previous best result
        seed_percentage: Fraction of population to seed with previous best
    """
    print("🧬 FULL EVOLUTION MODE")
    print("=" * 40)

    ga = SimpleGA(population_size=population_size)
    evolution = FullEvolutionMode(ga)
    best = evolution.evolve(generations, show_simulations=show_simulations,
                          use_previous_best=use_previous_best, seed_percentage=seed_percentage)

    # Save best result
    with open('best_evolution_result.json', 'w') as f:
        result_data = {
            'best_fitness': best['fitness'],
            'best_height': best['max_height'],
            'best_level': best['max_level'],
            'moveset': [{'move_type': m.move_type, 'strength': m.strength} for m in best['moveset']]
        }
        json.dump(result_data, f, indent=2)

    print(f"💾 Best result saved to best_evolution_result.json")


if __name__ == "__main__":
    print("🤖 Jump King Simple GA")
    print("Choose mode:")
    print("1. Level Solver (find best moveset per level)")
    print("2. Full Evolution (evolve over generations)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Level solver mode (individual visual simulations)
        show_sims = input("How many simulations to show visually? (0 for none): ").strip()
        try:
            show_simulations = int(show_sims)
        except:
            show_simulations = 0

        max_level = input("Max level to solve (default 2): ").strip()
        try:
            max_level = int(max_level)
        except:
            max_level = 2

        simulations = input("Simulations per level (default 100): ").strip()
        try:
            simulations = int(simulations)
        except:
            simulations = 100

        print(f"\n🚀 Starting Level Solver:")
        print(f"   Max level: {max_level}")
        print(f"   Simulations per level: {simulations}")
        print(f"   Visual simulations: {show_simulations}")

        run_level_solver(max_level, simulations, show_simulations)

    elif choice == "2":
        # Full evolution mode (population visual simulations)
        print("\n🎮 In Full Evolution mode, visual simulations show ALL AI players together!")
        show_sims = input("How many GENERATION simulations to show visually? (0 for none): ").strip()
        try:
            show_simulations = int(show_sims)
        except:
            show_simulations = 0

        generations = input("Number of generations (default 20): ").strip()
        try:
            generations = int(generations)
        except:
            generations = 20

        pop_size = input("Population size (default 50): ").strip()
        try:
            pop_size = int(pop_size)
        except:
            pop_size = 50

        # Ask about using previous best results
        use_prev = input("Start from previous best result? (y/n, default n): ").strip().lower()
        use_previous_best = use_prev in ['y', 'yes', '1', 'true']

        seed_percentage = 0.2  # Default
        if use_previous_best:
            seed_input = input("What % of population to seed with previous best? (default 20): ").strip()
            try:
                seed_percentage = float(seed_input) / 100.0
                seed_percentage = max(0.0, min(1.0, seed_percentage))  # Clamp to 0-1
            except:
                seed_percentage = 0.2

        print(f"\n🚀 Starting Full Evolution:")
        print(f"   Generations: {generations}")
        print(f"   Population size: {pop_size}")
        if use_previous_best:
            print(f"   🌱 Will seed {seed_percentage*100:.0f}% of population from previous best")
        else:
            print(f"   🎲 Starting with completely random population")
        if show_simulations > 0:
            print(f"   👀 Will show {show_simulations} generations visually")
            print(f"   🎮 Each visual shows all {pop_size} AI players playing together!")
        else:
            print(f"   ⚡ All simulations will run at maximum speed (no visuals)")

        run_full_evolution(generations, pop_size, show_simulations, use_previous_best, seed_percentage)

    else:
        print("Invalid choice, running full evolution with defaults...")
        run_full_evolution()
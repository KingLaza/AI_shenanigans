"""
Fitness Function: Defining Success for AI

The fitness function is THE most important component of a genetic algorithm.
It defines what "good" means and shapes the entire evolutionary process.

KEY AI CONCEPT: OBJECTIVE FUNCTIONS
==================================
An objective function mathematically defines what we're trying to optimize.
For Jump King AI, we want to maximize:
- Height reached (primary goal)
- Level progression (reaching higher levels)
- Survival time (staying alive longer)
- Efficient movement (reaching goals with fewer moves)

And minimize:
- Deaths/falls
- Wasted moves
- Getting stuck in loops

The fitness function is where domain knowledge meets algorithmic optimization.
"""

from typing import Dict, Any, List, Callable, Optional
import math
from abc import ABC, abstractmethod
from enum import Enum


class FitnessMetric(Enum):
    """Different aspects of performance we can measure"""
    HEIGHT_REACHED = "height_reached"
    LEVEL_PROGRESSION = "level_progression"
    SURVIVAL_TIME = "survival_time"
    MOVEMENT_EFFICIENCY = "movement_efficiency"
    DEATH_PENALTY = "death_penalty"
    EXPLORATION_BONUS = "exploration_bonus"
    CONSISTENCY = "consistency"


class FitnessFunction(ABC):
    """
    Abstract base class for fitness evaluation strategies.

    AI CONCEPT: STRATEGY PATTERN
    ===========================
    Different fitness functions emphasize different behaviors. By making
    this abstract, we can experiment with various approaches:
    - Height-focused (reach maximum height)
    - Explorer (visit many areas)
    - Efficient (reach goals with minimal moves)
    - Consistent (reliable performance)
    """

    def __init__(self, name: str):
        self.name = name
        self.evaluation_count = 0
        self.best_fitness = float('-inf')
        self.fitness_history: List[float] = []

    @abstractmethod
    def evaluate(self, performance_data: Dict[str, Any]) -> float:
        """
        Calculate fitness score from performance data.

        Args:
            performance_data: Dictionary containing player performance metrics

        Returns:
            Fitness score (higher = better)
        """
        pass

    def __call__(self, performance_data: Dict[str, Any]) -> float:
        """Make the fitness function callable"""
        fitness = self.evaluate(performance_data)

        # Track statistics
        self.evaluation_count += 1
        self.fitness_history.append(fitness)
        self.best_fitness = max(self.best_fitness, fitness)

        return fitness

    def reset_statistics(self):
        """Reset tracking statistics"""
        self.evaluation_count = 0
        self.best_fitness = float('-inf')
        self.fitness_history.clear()


class JumpKingFitness(FitnessFunction):
    """
    Comprehensive fitness function for Jump King AI.

    AI CONCEPT: MULTI-OBJECTIVE OPTIMIZATION
    ========================================
    Real-world problems rarely have a single objective. Jump King success
    involves multiple competing goals:
    - Go high (main objective)
    - Don't die (survival)
    - Be efficient (don't waste moves)
    - Explore (find new paths)

    We combine these into a weighted sum, but other approaches exist:
    - Pareto optimization (find trade-offs)
    - Lexicographic ordering (priority-based)
    - Goal programming (satisfy constraints)
    """

    def __init__(self,
                 height_weight: float = 1.0,
                 level_weight: float = 2.0,
                 survival_weight: float = 0.5,
                 efficiency_weight: float = 0.3,
                 death_penalty: float = -10.0,
                 exploration_bonus: float = 0.2):
        """
        Initialize the fitness function with customizable weights.

        Args:
            height_weight: How much to reward absolute height
            level_weight: How much to reward reaching higher levels
            survival_weight: How much to reward staying alive longer
            efficiency_weight: How much to reward efficient movement
            death_penalty: Penalty for dying/falling
            exploration_bonus: Bonus for visiting new areas
        """
        super().__init__("JumpKingFitness")

        # Weights for different objectives
        self.height_weight = height_weight
        self.level_weight = level_weight
        self.survival_weight = survival_weight
        self.efficiency_weight = efficiency_weight
        self.death_penalty = death_penalty
        self.exploration_bonus = exploration_bonus

        # Normalization constants (adjust based on your game scale)
        self.max_expected_height = 1000.0  # Adjust to your level heights
        self.max_expected_level = 10  # Number of levels in your game
        self.max_simulation_time = 1200  # Max simulation steps (20 moves * 60 FPS)

    def evaluate(self, performance_data: Dict[str, Any]) -> float:
        """
        Calculate comprehensive fitness score.

        Performance data should contain:
        - max_height_reached: Highest Y position achieved
        - final_height: Height at end of simulation
        - max_level_reached: Highest level number achieved
        - survival_time: Number of steps survived
        - total_moves_used: Number of moves actually executed
        - deaths: Number of times player died/fell
        - positions_visited: Set of unique positions (for exploration)
        - level_completion_times: Time to reach each level
        """

        # Extract metrics with defaults
        max_height = performance_data.get('max_height_reached', 0)
        final_height = performance_data.get('final_height', 0)
        max_level = performance_data.get('max_level_reached', 0)
        survival_time = performance_data.get('survival_time', 0)
        moves_used = performance_data.get('total_moves_used', 20)
        deaths = performance_data.get('deaths', 0)
        positions_visited = performance_data.get('positions_visited', set())

        # Component 1: Height Achievement
        # Reward both maximum height reached and final height
        height_score = self._calculate_height_score(max_height, final_height)

        # Component 2: Level Progression
        # Exponential rewards for reaching higher levels
        level_score = self._calculate_level_score(max_level, performance_data)

        # Component 3: Survival and Efficiency
        survival_score = self._calculate_survival_score(survival_time, moves_used)

        # Component 4: Movement Efficiency
        efficiency_score = self._calculate_efficiency_score(
            max_height, moves_used, survival_time
        )

        # Component 5: Death Penalty
        death_score = self.death_penalty * deaths

        # Component 6: Exploration Bonus
        exploration_score = self._calculate_exploration_score(positions_visited)

        # Combine all components
        total_fitness = (
                self.height_weight * height_score +
                self.level_weight * level_score +
                self.survival_weight * survival_score +
                self.efficiency_weight * efficiency_score +
                death_score +
                self.exploration_bonus * exploration_score
        )

        # Store detailed breakdown for analysis
        performance_data['fitness_breakdown'] = {
            'height_score': height_score,
            'level_score': level_score,
            'survival_score': survival_score,
            'efficiency_score': efficiency_score,
            'death_score': death_score,
            'exploration_score': exploration_score,
            'total_fitness': total_fitness
        }

        return total_fitness

    def _calculate_height_score(self, max_height: float, final_height: float) -> float:
        """
        Calculate height-based fitness.

        AI CONCEPT: REWARD SHAPING
        =========================
        We reward both peak achievement (max height) and sustained performance
        (final height). This encourages both exploration and stability.
        """
        # Normalize heights to 0-1 range
        normalized_max = min(max_height / self.max_expected_height, 1.0)
        normalized_final = min(final_height / self.max_expected_height, 1.0)

        # Weighted combination: peak achievement + sustained performance
        return 0.7 * normalized_max + 0.3 * normalized_final

    def _calculate_level_score(self, max_level: int, performance_data: Dict) -> float:
        """
        Calculate level progression fitness.

        Exponential rewards for reaching higher levels - each level is
        significantly harder than the last.
        """
        if max_level == 0:
            return 0.0

        # Exponential scaling: level 1 = 1 point, level 2 = 2 points, etc.
        base_score = (2 ** max_level - 1) / (2 ** self.max_expected_level - 1)

        # Bonus for speed: faster level completion = higher score
        completion_times = performance_data.get('level_completion_times', {})
        speed_bonus = 0.0

        for level, time in completion_times.items():
            if time > 0:
                # Bonus decreases with time (faster = better)
                time_factor = max(0.1, 1.0 - (time / self.max_simulation_time))
                speed_bonus += time_factor * 0.1

        return base_score + speed_bonus

    def _calculate_survival_score(self, survival_time: int, moves_used: int) -> float:
        """
        Reward staying alive and using moves effectively.
        """
        # Basic survival score
        survival_ratio = min(survival_time / self.max_simulation_time, 1.0)

        # Bonus for using moves (don't reward just standing still)
        move_usage_ratio = moves_used / 20.0  # 20 is max moves

        return survival_ratio * move_usage_ratio

    def _calculate_efficiency_score(self, height: float, moves_used: int, time: int) -> float:
        """
        Reward efficient movement - getting high with fewer moves and time.

        AI CONCEPT: EFFICIENCY METRICS
        =============================
        We want solutions that achieve goals with minimal resources.
        This prevents the AI from developing wasteful behaviors.
        """
        if moves_used == 0 or time == 0:
            return 0.0

        # Height per move used
        height_per_move = height / moves_used

        # Height per time unit
        height_per_time = height / time

        # Normalize and combine
        normalized_move_efficiency = min(height_per_move / 50.0, 1.0)  # Adjust scale
        normalized_time_efficiency = min(height_per_time / 1.0, 1.0)  # Adjust scale

        return (normalized_move_efficiency + normalized_time_efficiency) / 2

    def _calculate_exploration_score(self, positions_visited: set) -> float:
        """
        Reward exploring different areas of the level.

        This prevents the AI from just finding one path and repeating it.
        Exploration can lead to discovering better routes.
        """
        if not positions_visited:
            return 0.0

        # More unique positions = higher score
        exploration_factor = len(positions_visited)

        # Normalize based on expected exploration (adjust based on level size)
        max_expected_positions = 100  # Rough estimate
        return min(exploration_factor / max_expected_positions, 1.0)


class HeightOnlyFitness(FitnessFunction):
    """
    Simple fitness function that only cares about maximum height.

    Use this for:
    - Baseline comparisons
    - Simple experiments
    - When you want pure height optimization
    """

    def __init__(self):
        super().__init__("HeightOnlyFitness")

    def evaluate(self, performance_data: Dict[str, Any]) -> float:
        return performance_data.get('max_height_reached', 0)


class SurvivalFitness(FitnessFunction):
    """
    Fitness function that prioritizes staying alive.

    Good for:
    - Learning basic movement
    - Avoiding immediate failures
    - Conservative strategies
    """

    def __init__(self):
        super().__init__("SurvivalFitness")

    def evaluate(self, performance_data: Dict[str, Any]) -> float:
        survival_time = performance_data.get('survival_time', 0)
        deaths = performance_data.get('deaths', 0)

        # High survival time, low deaths
        return survival_time - (deaths * 100)


class ExplorationFitness(FitnessFunction):
    """
    Fitness function that rewards exploration and discovery.

    Good for:
    - Finding new paths
    - Discovering level secrets
    - Diverse strategies
    """

    def __init__(self):
        super().__init__("ExplorationFitness")

    def evaluate(self, performance_data: Dict[str, Any]) -> float:
        positions = performance_data.get('positions_visited', set())
        height = performance_data.get('max_height_reached', 0)

        # Reward both exploration and some height progress
        exploration_score = len(positions) * 2
        height_score = height * 0.1

        return exploration_score + height_score


class FitnessEvaluator:
    """
    Manages fitness evaluation for populations.

    AI CONCEPT: EVALUATION STRATEGIES
    ===============================
    Different evaluation approaches:
    - Individual: Evaluate each genome separately
    - Competitive: Genomes compete against each other
    - Collaborative: Genomes work together
    - Dynamic: Fitness function changes over time
    """

    def __init__(self, fitness_function: FitnessFunction):
        self.fitness_function = fitness_function
        self.evaluation_cache: Dict[str, float] = {}  # Cache results

    def evaluate_genome(self, genome, performance_data: Dict[str, Any]) -> float:
        """
        Evaluate a single genome's fitness.

        Args:
            genome: The genome to evaluate
            performance_data: Performance metrics from simulation

        Returns:
            Fitness score
        """
        # Check cache first (for identical genomes)
        genome_key = self._get_genome_key(genome)
        if genome_key in self.evaluation_cache:
            cached_fitness = self.evaluation_cache[genome_key]
            genome.fitness = cached_fitness
            return cached_fitness

        # Calculate fitness
        fitness = self.fitness_function(performance_data)

        # Store in genome and cache
        genome.fitness = fitness
        genome.performance_data = performance_data.copy()
        self.evaluation_cache[genome_key] = fitness

        return fitness

    def evaluate_population(self, population, simulation_results: List[Dict]) -> List[float]:
        """
        Evaluate entire population fitness.

        Args:
            population: Population object
            simulation_results: List of performance data for each genome

        Returns:
            List of fitness scores
        """
        fitness_scores = []

        for genome, performance_data in zip(population, simulation_results):
            fitness = self.evaluate_genome(genome, performance_data)
            fitness_scores.append(fitness)

        return fitness_scores

    def _get_genome_key(self, genome) -> str:
        """Generate cache key for genome"""
        # Simple hash based on move sequence
        move_tuples = [(m.move_type, m.strength) for m in genome.moves]
        return str(hash(tuple(move_tuples)))

    def clear_cache(self):
        """Clear the evaluation cache"""
        self.evaluation_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            'total_evaluations': self.fitness_function.evaluation_count,
            'best_fitness': self.fitness_function.best_fitness,
            'cache_size': len(self.evaluation_cache),
            'function_name': self.fitness_function.name
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Fitness Functions...")

    # Sample performance data
    sample_performance = {
        'max_height_reached': 150.0,
        'final_height': 120.0,
        'max_level_reached': 2,
        'survival_time': 800,
        'total_moves_used': 15,
        'deaths': 1,
        'positions_visited': set(range(50)),  # Visited 50 positions
        'level_completion_times': {1: 400, 2: 800}
    }

    # Test different fitness functions
    fitness_functions = [
        JumpKingFitness(),
        HeightOnlyFitness(),
        SurvivalFitness(),
        ExplorationFitness()
    ]

    for fitness_fn in fitness_functions:
        score = fitness_fn.evaluate(sample_performance)
        print(f"{fitness_fn.name}: {score:.3f}")

    # Test detailed breakdown
    jk_fitness = JumpKingFitness()
    score = jk_fitness.evaluate(sample_performance)

    print(f"\nDetailed breakdown:")
    breakdown = sample_performance['fitness_breakdown']
    for component, value in breakdown.items():
        print(f"  {component}: {value:.3f}")

    print("All tests passed!")
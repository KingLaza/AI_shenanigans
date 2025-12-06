"""
Selection Strategies: Driving Evolutionary Pressure

Selection determines which genomes get to reproduce, directly controlling
the direction of evolution. Different selection methods create different
evolutionary pressures and lead to different behaviors.

KEY AI CONCEPT: SELECTION PRESSURE
=================================
Selection pressure is how "picky" we are about who gets to reproduce:
- High pressure: Only the very best reproduce (fast convergence, risk of local optima)
- Low pressure: Everyone has a chance (slow convergence, maintains diversity)
- Variable pressure: Adjust based on population state

This is one of the most important design decisions in genetic algorithms.
"""

from typing import List, Tuple, Optional, Dict, Any
import random
import math
from abc import ABC, abstractmethod
from .genome import Genome


class SelectionStrategy(ABC):
    """
    Abstract base class for selection strategies.

    AI CONCEPT: STRATEGY PATTERN FOR ALGORITHMS
    ==========================================
    Different selection methods embody different philosophies about evolution:
    - Tournament: Local competition
    - Roulette: Proportional to fitness
    - Rank: Position-based selection
    - Elitist: Always preserve the best

    By making this pluggable, we can experiment and find what works best.
    """

    def __init__(self, name: str):
        self.name = name
        self.selection_count = 0
        self.selection_history: List[Dict] = []

    @abstractmethod
    def select_parents(self, population: List[Genome], num_pairs: int) -> List[Tuple[Genome, Genome]]:
        """
        Select parent pairs for reproduction.

        Args:
            population: List of genomes to select from
            num_pairs: Number of parent pairs to select

        Returns:
            List of (parent1, parent2) tuples
        """
        pass

    def select_survivors(self, population: List[Genome], num_survivors: int) -> List[Genome]:
        """
        Select which genomes survive to the next generation.

        Default implementation: select best by fitness
        """
        # Filter evaluated genomes and sort by fitness
        evaluated = [g for g in population if g.fitness is not None]
        evaluated.sort(key=lambda g: g.fitness, reverse=True)

        return evaluated[:num_survivors]

    def record_selection(self, selected_genomes: List[Genome], population: List[Genome]):
        """Record statistics about this selection event"""
        if not selected_genomes:
            return

        selected_fitness = [g.fitness for g in selected_genomes if g.fitness is not None]
        population_fitness = [g.fitness for g in population if g.fitness is not None]

        if selected_fitness and population_fitness:
            record = {
                'selection_count': self.selection_count,
                'selected_mean': sum(selected_fitness) / len(selected_fitness),
                'population_mean': sum(population_fitness) / len(population_fitness),
                'selected_best': max(selected_fitness),
                'population_best': max(population_fitness),
                'selection_pressure': len(selected_genomes) / len(population)
            }
            self.selection_history.append(record)

        self.selection_count += 1


class TournamentSelection(SelectionStrategy):
    """
    Tournament Selection: Local competition determines winners.

    AI CONCEPT: TOURNAMENT SELECTION
    ===============================
    How it works:
    1. Pick k random genomes (tournament size)
    2. The best one wins and gets selected
    3. Repeat until you have enough parents

    Benefits:
    - Easy to implement and understand
    - Natural selection pressure control (larger tournaments = more pressure)
    - Works well even with negative fitness values
    - Maintains good diversity

    Tournament size effects:
    - Size 1: Random selection (no pressure)
    - Size 2: Moderate pressure, good balance
    - Size 5+: High pressure, fast convergence
    - Size = population: Always selects the best (elitist)
    """

    def __init__(self, tournament_size: int = 3, allow_duplicates: bool = True):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of genomes competing in each tournament
            allow_duplicates: Whether same genome can be selected multiple times
        """
        super().__init__(f"Tournament(size={tournament_size})")
        self.tournament_size = tournament_size
        self.allow_duplicates = allow_duplicates

        if tournament_size < 1:
            raise ValueError("Tournament size must be at least 1")

    def select_parents(self, population: List[Genome], num_pairs: int) -> List[Tuple[Genome, Genome]]:
        """Select parent pairs using tournament selection"""
        if not population:
            return []

        # Filter to only evaluated genomes
        candidates = [g for g in population if g.fitness is not None]
        if len(candidates) < 2:
            return []

        parent_pairs = []
        used_genomes = set() if not self.allow_duplicates else None

        for _ in range(num_pairs):
            # Select first parent
            parent1 = self._run_tournament(candidates, used_genomes)
            if parent1 is None:
                break

            # Select second parent (different from first)
            excluded = {parent1} if used_genomes is None else used_genomes | {parent1}
            parent2 = self._run_tournament(candidates, excluded)
            if parent2 is None:
                # If we can't find a different parent, use the first one anyway
                # (happens with small populations)
                parent2 = parent1

            parent_pairs.append((parent1, parent2))

            if not self.allow_duplicates:
                used_genomes.add(parent1)
                used_genomes.add(parent2)

                # If we've used too many genomes, stop
                if len(used_genomes) >= len(candidates) - 1:
                    break

        # Record selection statistics
        all_selected = [p for pair in parent_pairs for p in pair]
        self.record_selection(all_selected, population)

        return parent_pairs

    def _run_tournament(self, candidates: List[Genome], excluded: Optional[set] = None) -> Optional[Genome]:
        """Run a single tournament and return the winner"""
        if excluded is None:
            excluded = set()

        # Get available candidates
        available = [g for g in candidates if g not in excluded]
        if not available:
            return None

        # Select tournament participants
        tournament_size = min(self.tournament_size, len(available))
        participants = random.sample(available, tournament_size)

        # Find the winner (best fitness)
        winner = max(participants, key=lambda g: g.fitness or float('-inf'))
        return winner


class RouletteWheelSelection(SelectionStrategy):
    """
    Roulette Wheel Selection: Probability proportional to fitness.

    AI CONCEPT: FITNESS-PROPORTIONAL SELECTION
    =========================================
    Imagine a roulette wheel where each genome gets a slice proportional
    to its fitness. Better genomes get bigger slices, so they're more
    likely to be selected, but everyone has some chance.

    Benefits:
    - Intuitive: better fitness = higher selection chance
    - Preserves diversity: weak genomes still have some chance
    - Smooth selection pressure

    Problems:
    - Doesn't work with negative fitness values
    - Can have premature convergence if one genome is much better
    - Sensitive to fitness function scaling
    """

    def __init__(self, scaling_method: str = "linear"):
        """
        Initialize roulette wheel selection.

        Args:
            scaling_method: How to handle fitness values
                - "linear": Use raw fitness values
                - "rank": Convert to ranks first
                - "power": Apply power transformation
        """
        super().__init__(f"RouletteWheel({scaling_method})")
        self.scaling_method = scaling_method

    def select_parents(self, population: List[Genome], num_pairs: int) -> List[Tuple[Genome, Genome]]:
        """Select parent pairs using roulette wheel selection"""
        # Filter to evaluated genomes
        candidates = [g for g in population if g.fitness is not None]
        if len(candidates) < 2:
            return []

        # Calculate selection probabilities
        probabilities = self._calculate_probabilities(candidates)

        parent_pairs = []
        for _ in range(num_pairs):
            parent1 = self._spin_wheel(candidates, probabilities)
            parent2 = self._spin_wheel(candidates, probabilities)

            # Ensure parents are different (if possible)
            attempts = 0
            while parent2 == parent1 and len(candidates) > 1 and attempts < 10:
                parent2 = self._spin_wheel(candidates, probabilities)
                attempts += 1

            parent_pairs.append((parent1, parent2))

        # Record statistics
        all_selected = [p for pair in parent_pairs for p in pair]
        self.record_selection(all_selected, population)

        return parent_pairs

    def _calculate_probabilities(self, candidates: List[Genome]) -> List[float]:
        """Calculate selection probabilities for each genome"""
        fitness_values = [g.fitness for g in candidates]

        if self.scaling_method == "linear":
            # Handle negative fitness values by shifting
            min_fitness = min(fitness_values)
            if min_fitness < 0:
                adjusted_fitness = [f - min_fitness + 1 for f in fitness_values]
            else:
                adjusted_fitness = fitness_values

        elif self.scaling_method == "rank":
            # Convert to ranks (1 = worst, n = best)
            sorted_indices = sorted(range(len(fitness_values)),
                                    key=lambda i: fitness_values[i])
            adjusted_fitness = [0] * len(fitness_values)
            for rank, idx in enumerate(sorted_indices):
                adjusted_fitness[idx] = rank + 1

        elif self.scaling_method == "power":
            # Apply power transformation to increase selection pressure
            min_fitness = min(fitness_values)
            shifted = [f - min_fitness + 1 for f in fitness_values]
            adjusted_fitness = [f ** 2 for f in shifted]  # Square for more pressure

        # Convert to probabilities
        total = sum(adjusted_fitness)
        if total == 0:
            # All fitness values are equal, use uniform probabilities
            return [1.0 / len(candidates)] * len(candidates)

        return [f / total for f in adjusted_fitness]

    def _spin_wheel(self, candidates: List[Genome], probabilities: List[float]) -> Genome:
        """Spin the roulette wheel and return selected genome"""
        r = random.random()
        cumulative_prob = 0.0

        for genome, prob in zip(candidates, probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return genome

        # Fallback (shouldn't happen due to floating point errors)
        return candidates[-1]


class RankSelection(SelectionStrategy):
    """
    Rank-based Selection: Selection based on fitness ranking, not raw values.

    AI CONCEPT: RANK-BASED SELECTION
    ===============================
    Instead of using raw fitness values, we rank genomes from worst to best
    and use ranks for selection. This solves several problems:
    - Works with negative fitness values
    - Prevents domination by super-fit individuals
    - Consistent selection pressure regardless of fitness scale

    Ranking strategies:
    - Linear: P(rank) = rank / sum(ranks)
    - Exponential: P(rank) = exp(rank) / sum(exp(ranks))
    """

    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize rank selection.

        Args:
            selection_pressure: Controls how much better ranks are favored
                - 1.0: Uniform selection (no pressure)
                - 1.5: Moderate pressure (recommended)
                - 2.0+: High pressure
        """
        super().__init__(f"Rank(pressure={selection_pressure})")
        self.selection_pressure = selection_pressure

        if selection_pressure < 1.0:
            raise ValueError("Selection pressure must be >= 1.0")

    def select_parents(self, population: List[Genome], num_pairs: int) -> List[Tuple[Genome, Genome]]:
        """Select parent pairs using rank-based selection"""
        # Filter and rank genomes
        candidates = [g for g in population if g.fitness is not None]
        if len(candidates) < 2:
            return []

        # Sort by fitness (worst to best)
        ranked_genomes = sorted(candidates, key=lambda g: g.fitness)

        # Calculate selection probabilities based on rank
        probabilities = self._calculate_rank_probabilities(len(ranked_genomes))

        parent_pairs = []
        for _ in range(num_pairs):
            parent1_idx = self._select_by_rank_probability(probabilities)
            parent2_idx = self._select_by_rank_probability(probabilities)

            # Ensure different parents if possible
            attempts = 0
            while parent2_idx == parent1_idx and len(ranked_genomes) > 1 and attempts < 10:
                parent2_idx = self._select_by_rank_probability(probabilities)
                attempts += 1

            parent1 = ranked_genomes[parent1_idx]
            parent2 = ranked_genomes[parent2_idx]
            parent_pairs.append((parent1, parent2))

        # Record statistics
        all_selected = [p for pair in parent_pairs for p in pair]
        self.record_selection(all_selected, population)

        return parent_pairs

    def _calculate_rank_probabilities(self, population_size: int) -> List[float]:
        """Calculate selection probability for each rank"""
        # Linear ranking: P(rank) = (2 - sp + 2*(sp-1)*(rank-1)/(n-1)) / n
        # Where sp = selection_pressure, rank goes from 1 (worst) to n (best)

        if population_size == 1:
            return [1.0]

        probabilities = []
        for rank in range(1, population_size + 1):  # 1-indexed ranks
            prob = (2 - self.selection_pressure +
                    2 * (self.selection_pressure - 1) * (rank - 1) / (population_size - 1))
            prob /= population_size  # Normalize
            probabilities.append(prob)

        return probabilities

    def _select_by_rank_probability(self, probabilities: List[float]) -> int:
        """Select index based on rank probabilities"""
        r = random.random()
        cumulative_prob = 0.0

        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return i

        return len(probabilities) - 1  # Fallback


class ElitistSelection(SelectionStrategy):
    """
    Elitist Selection: Always preserve the best genomes.

    AI CONCEPT: ELITISM
    ==================
    Elitism guarantees that the best solutions are never lost due to
    random chance in crossover or mutation. This ensures monotonic
    improvement in the best fitness.

    Strategies:
    - Pure elitism: Only the best reproduce
    - Mixed elitism: Some elite + some diverse selections
    - Adaptive elitism: Adjust elite percentage based on progress
    """

    def __init__(self, elite_percentage: float = 0.2, diversity_percentage: float = 0.3):
        """
        Initialize elitist selection.

        Args:
            elite_percentage: Fraction of population that are elite
            diversity_percentage: Fraction selected for diversity
        """
        super().__init__(f"Elitist(elite={elite_percentage:.1f})")
        self.elite_percentage = elite_percentage
        self.diversity_percentage = diversity_percentage

        if elite_percentage + diversity_percentage > 1.0:
            raise ValueError("Elite + diversity percentages cannot exceed 1.0")

    def select_parents(self, population: List[Genome], num_pairs: int) -> List[Tuple[Genome, Genome]]:
        """Select parent pairs with elitist strategy"""
        # Filter to evaluated genomes
        candidates = [g for g in population if g.fitness is not None]
        if len(candidates) < 2:
            return []

        # Sort by fitness (best first)
        sorted_genomes = sorted(candidates, key=lambda g: g.fitness, reverse=True)

        # Determine selection pools
        num_elite = max(1, int(len(sorted_genomes) * self.elite_percentage))
        num_diverse = int(len(sorted_genomes) * self.diversity_percentage)

        elite_pool = sorted_genomes[:num_elite]
        diverse_pool = sorted_genomes[num_elite:num_elite + num_diverse] if num_diverse > 0 else []
        random_pool = sorted_genomes[num_elite + num_diverse:] if len(sorted_genomes) > num_elite + num_diverse else []

        parent_pairs = []
        for _ in range(num_pairs):
            # Select parents from different pools
            parent1 = self._select_from_pools(elite_pool, diverse_pool, random_pool)
            parent2 = self._select_from_pools(elite_pool, diverse_pool, random_pool)

            # Ensure different parents if possible
            attempts = 0
            while parent2 == parent1 and len(candidates) > 1 and attempts < 10:
                parent2 = self._select_from_pools(elite_pool, diverse_pool, random_pool)
                attempts += 1

            parent_pairs.append((parent1, parent2))

        # Record statistics
        all_selected = [p for pair in parent_pairs for p in pair]
        self.record_selection(all_selected, population)

        return parent_pairs

    def _select_from_pools(self, elite_pool: List[Genome], diverse_pool: List[Genome],
                           random_pool: List[Genome]) -> Genome:
        """Select from one of the pools based on probabilities"""
        # Weighted selection from pools
        pool_weights = [0.6, 0.3, 0.1]  # Favor elite, then diverse, then random
        available_pools = []

        if elite_pool:
            available_pools.append((elite_pool, pool_weights[0]))
        if diverse_pool:
            available_pools.append((diverse_pool, pool_weights[1]))
        if random_pool:
            available_pools.append((random_pool, pool_weights[2]))

        if not available_pools:
            return elite_pool[0] if elite_pool else diverse_pool[0]

        # Select pool
        r = random.random()
        cumulative_weight = 0.0
        total_weight = sum(weight for _, weight in available_pools)

        for pool, weight in available_pools:
            cumulative_weight += weight / total_weight
            if r <= cumulative_weight:
                return random.choice(pool)

        # Fallback
        return available_pools[0][0][0]


class SelectionManager:
    """
    Manages selection strategies and provides analysis tools.

    Allows easy switching between strategies and comparison of their effects.
    """

    def __init__(self):
        self.strategies: Dict[str, SelectionStrategy] = {}
        self.current_strategy: Optional[SelectionStrategy] = None

    def register_strategy(self, strategy: SelectionStrategy):
        """Register a selection strategy"""
        self.strategies[strategy.name] = strategy

    def set_strategy(self, name: str):
        """Set the active selection strategy"""
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy: {name}")
        self.current_strategy = self.strategies[name]

    def select_parents(self, population: List[Genome], num_pairs: int) -> List[Tuple[Genome, Genome]]:
        """Select parents using current strategy"""
        if self.current_strategy is None:
            raise ValueError("No selection strategy set")
        return self.current_strategy.select_parents(population, num_pairs)

    def compare_strategies(self, population: List[Genome], num_trials: int = 100) -> Dict[str, Dict]:
        """Compare different selection strategies on the same population"""
        results = {}

        for name, strategy in self.strategies.items():
            trial_results = []

            for _ in range(num_trials):
                parents = strategy.select_parents(population, len(population) // 4)
                if parents:
                    selected_fitness = [p.fitness for pair in parents for p in pair if p.fitness is not None]
                    if selected_fitness:
                        trial_results.append({
                            'mean_fitness': sum(selected_fitness) / len(selected_fitness),
                            'max_fitness': max(selected_fitness),
                            'diversity': len(set(p.id for pair in parents for p in pair))
                        })

            if trial_results:
                results[name] = {
                    'avg_mean_fitness': sum(r['mean_fitness'] for r in trial_results) / len(trial_results),
                    'avg_max_fitness': sum(r['max_fitness'] for r in trial_results) / len(trial_results),
                    'avg_diversity': sum(r['diversity'] for r in trial_results) / len(trial_results)
                }

        return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing Selection Strategies...")

    # Create test population
    from .genome import Genome

    test_population = []
    for i in range(20):
        genome = Genome.random()
        genome.fitness = random.uniform(0, 100) + i * 2  # Increasing fitness trend
        test_population.append(genome)

    # Test different selection strategies
    strategies = [
        TournamentSelection(tournament_size=3),
        RouletteWheelSelection(),
        RankSelection(selection_pressure=1.5),
        ElitistSelection(elite_percentage=0.3)
    ]

    for strategy in strategies:
        print(f"\nTesting {strategy.name}:")
        parent_pairs = strategy.select_parents(test_population, 5)

        for i, (p1, p2) in enumerate(parent_pairs):
            print(f"  Pair {i + 1}: {p1.fitness:.1f} + {p2.fitness:.1f}")

    print("\nAll tests passed!")
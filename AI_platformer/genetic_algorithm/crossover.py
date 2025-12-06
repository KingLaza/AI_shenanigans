"""
Crossover Operators: Sexual Reproduction in AI

Crossover combines genetic material from two parents to create offspring.
This is the "sexual reproduction" of genetic algorithms - it allows successful
traits to be combined in new ways, potentially creating superior solutions.

KEY AI CONCEPT: RECOMBINATION & BUILDING BLOCKS
==============================================
The Building Block Hypothesis suggests that genetic algorithms work by
discovering, preserving, and combining "building blocks" - short sequences
of genes that contribute to fitness. Good crossover operators:

1. Preserve good building blocks from parents
2. Create novel combinations of building blocks
3. Maintain genetic diversity
4. Respect problem structure

For Jump King, building blocks might be:
- Successful jump sequences for specific obstacles
- Walking patterns for horizontal movement
- Timing patterns for complex maneuvers
"""

from typing import List, Tuple, Optional, Dict, Any
import random
import copy
from abc import ABC, abstractmethod
from .genome import Genome, Move


class CrossoverOperator(ABC):
    """
    Abstract base class for crossover operators.

    AI CONCEPT: CROSSOVER TAXONOMY
    =============================
    Different crossover methods emphasize different aspects:
    - Single-point: Simple, preserves large building blocks
    - Two-point: Balanced, good for medium-sized blocks
    - Uniform: High disruption, good for fine-grained mixing
    - Order-based: Preserves sequence relationships
    - Adaptive: Changes behavior based on population state
    """

    def __init__(self, name: str, crossover_rate: float = 0.8):
        """
        Initialize crossover operator.

        Args:
            name: Name of the crossover method
            crossover_rate: Probability of crossover occurring (vs copying parents)
        """
        self.name = name
        self.crossover_rate = crossover_rate
        self.crossover_count = 0
        self.offspring_created = 0

        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")

    @abstractmethod
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """
        Perform crossover between two parents.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            Tuple of (child1, child2) offspring genomes
        """
        pass

    def apply_crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """
        Apply crossover with probability, otherwise copy parents.

        This wrapper handles the crossover rate - sometimes we skip crossover
        and just copy the parents to maintain diversity.
        """
        self.crossover_count += 1

        if random.random() < self.crossover_rate:
            # Perform crossover
            child1, child2 = self.crossover(parent1, parent2)
            self.offspring_created += 2
        else:
            # Skip crossover, copy parents (with slight mutation to avoid exact copies)
            child1 = Genome(
                copy.deepcopy(parent1.moves),
                max(parent1.generation, parent2.generation) + 1
            )
            child2 = Genome(
                copy.deepcopy(parent2.moves),
                max(parent1.generation, parent2.generation) + 1
            )
            child1.parent_ids = (parent1.id, None)
            child2.parent_ids = (parent2.id, None)

        return child1, child2

    def get_statistics(self) -> Dict[str, Any]:
        """Get crossover statistics"""
        return {
            'name': self.name,
            'crossover_rate': self.crossover_rate,
            'total_attempts': self.crossover_count,
            'offspring_created': self.offspring_created,
            'actual_crossover_rate': (self.offspring_created / max(1, self.crossover_count * 2))
        }


class SinglePointCrossover(CrossoverOperator):
    """
    Single-Point Crossover: Split at one position and swap tails.

    AI CONCEPT: SINGLE-POINT CROSSOVER
    ==================================
    How it works:
    1. Choose random cut point between 1 and length-1
    2. Child1 = Parent1[0:cut] + Parent2[cut:end]
    3. Child2 = Parent2[0:cut] + Parent1[cut:end]

    Example with move sequences:
    Parent1: [A, B, C, D, E, F]
    Parent2: [X, Y, Z, W, V, U]
    Cut point: 3
    Child1:  [A, B, C, W, V, U]
    Child2:  [X, Y, Z, D, E, F]

    Benefits:
    - Simple and fast
    - Preserves large building blocks
    - Low disruption to successful patterns

    Best for:
    - Problems where position in sequence matters
    - When you want to preserve long successful patterns
    - Early generations when exploring large-scale structure
    """

    def __init__(self, crossover_rate: float = 0.8):
        super().__init__("SinglePoint", crossover_rate)

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform single-point crossover"""
        if len(parent1.moves) != len(parent2.moves):
            raise ValueError("Parents must have same sequence length")

        sequence_length = len(parent1.moves)

        # Choose crossover point (between 1 and length-1)
        if sequence_length <= 2:
            crossover_point = 1
        else:
            crossover_point = random.randint(1, sequence_length - 1)

        # Create offspring by swapping segments
        child1_moves = (
                copy.deepcopy(parent1.moves[:crossover_point]) +
                copy.deepcopy(parent2.moves[crossover_point:])
        )

        child2_moves = (
                copy.deepcopy(parent2.moves[:crossover_point]) +
                copy.deepcopy(parent1.moves[crossover_point:])
        )

        # Create child genomes
        child1 = Genome(child1_moves, max(parent1.generation, parent2.generation) + 1)
        child2 = Genome(child2_moves, max(parent1.generation, parent2.generation) + 1)

        # Track lineage
        child1.parent_ids = (parent1.id, parent2.id)
        child2.parent_ids = (parent1.id, parent2.id)

        return child1, child2


class TwoPointCrossover(CrossoverOperator):
    """
    Two-Point Crossover: Swap a middle segment between parents.

    AI CONCEPT: TWO-POINT CROSSOVER
    ===============================
    How it works:
    1. Choose two random cut points
    2. Swap the middle segment between parents
    3. Keep the head and tail from each parent

    Example:
    Parent1: [A, B, C, D, E, F, G, H]
    Parent2: [X, Y, Z, W, V, U, T, S]
    Cut points: 2, 6
    Child1:  [A, B, W, V, U, T, G, H]
    Child2:  [X, Y, C, D, E, F, T, S]

    Benefits:
    - Preserves building blocks at both ends
    - More conservative than uniform crossover
    - Good balance of preservation and exploration
    - Works well for many types of problems

    Best for:
    - General-purpose crossover
    - When you want moderate disruption
    - Problems with multiple important regions
    """

    def __init__(self, crossover_rate: float = 0.8):
        super().__init__("TwoPoint", crossover_rate)

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform two-point crossover"""
        if len(parent1.moves) != len(parent2.moves):
            raise ValueError("Parents must have same sequence length")

        sequence_length = len(parent1.moves)

        if sequence_length <= 2:
            # Fall back to single-point for very short sequences
            point = 1
            child1_moves = (
                    copy.deepcopy(parent1.moves[:point]) +
                    copy.deepcopy(parent2.moves[point:])
            )
            child2_moves = (
                    copy.deepcopy(parent2.moves[:point]) +
                    copy.deepcopy(parent1.moves[point:])
            )
        else:
            # Choose two crossover points
            point1 = random.randint(1, sequence_length - 2)
            point2 = random.randint(point1 + 1, sequence_length - 1)

            # Create offspring by swapping middle segments
            child1_moves = (
                    copy.deepcopy(parent1.moves[:point1]) +
                    copy.deepcopy(parent2.moves[point1:point2]) +
                    copy.deepcopy(parent1.moves[point2:])
            )

            child2_moves = (
                    copy.deepcopy(parent2.moves[:point1]) +
                    copy.deepcopy(parent1.moves[point1:point2]) +
                    copy.deepcopy(parent2.moves[point2:])
            )

        # Create child genomes
        child1 = Genome(child1_moves, max(parent1.generation, parent2.generation) + 1)
        child2 = Genome(child2_moves, max(parent1.generation, parent2.generation) + 1)

        # Track lineage
        child1.parent_ids = (parent1.id, parent2.id)
        child2.parent_ids = (parent1.id, parent2.id)

        return child1, child2


class UniformCrossover(CrossoverOperator):
    """
    Uniform Crossover: Randomly choose each gene from either parent.

    AI CONCEPT: UNIFORM CROSSOVER
    ============================
    How it works:
    1. For each position, flip a coin
    2. If heads: take gene from parent1
    3. If tails: take gene from parent2
    4. Do this independently for each child

    Example:
    Parent1: [A, B, C, D, E, F]
    Parent2: [X, Y, Z, W, V, U]
    Random:  [1, 2, 1, 2, 1, 2] (1=parent1, 2=parent2)
    Child1:  [A, Y, C, W, E, U]
    Child2:  [X, B, Z, D, V, F]

    Benefits:
    - Maximum mixing of genetic material
    - Good for fine-grained exploration
    - No bias toward position in sequence
    - High genetic diversity in offspring

    Drawbacks:
    - Can destroy large building blocks
    - High disruption may lose good patterns

    Best for:
    - Later generations when fine-tuning
    - Problems where position doesn't matter much
    - When population lacks diversity
    """

    def __init__(self, crossover_rate: float = 0.8, gene_swap_probability: float = 0.5):
        """
        Initialize uniform crossover.

        Args:
            crossover_rate: Probability of performing crossover
            gene_swap_probability: Probability of choosing each parent for each gene
        """
        super().__init__("Uniform", crossover_rate)
        self.gene_swap_probability = gene_swap_probability

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform uniform crossover"""
        if len(parent1.moves) != len(parent2.moves):
            raise ValueError("Parents must have same sequence length")

        sequence_length = len(parent1.moves)

        child1_moves = []
        child2_moves = []

        for i in range(sequence_length):
            if random.random() < self.gene_swap_probability:
                # Child1 gets gene from parent1, child2 gets gene from parent2
                child1_moves.append(copy.deepcopy(parent1.moves[i]))
                child2_moves.append(copy.deepcopy(parent2.moves[i]))
            else:
                # Child1 gets gene from parent2, child2 gets gene from parent1
                child1_moves.append(copy.deepcopy(parent2.moves[i]))
                child2_moves.append(copy.deepcopy(parent1.moves[i]))

        # Create child genomes
        child1 = Genome(child1_moves, max(parent1.generation, parent2.generation) + 1)
        child2 = Genome(child2_moves, max(parent1.generation, parent2.generation) + 1)

        # Track lineage
        child1.parent_ids = (parent1.id, parent2.id)
        child2.parent_ids = (parent1.id, parent2.id)

        return child1, child2


class OrderCrossover(CrossoverOperator):
    """
    Order Crossover: Preserves relative order while mixing sequences.

    AI CONCEPT: ORDER-PRESERVING CROSSOVER
    ======================================
    For problems where the ORDER of elements matters more than their
    absolute positions. This is common in scheduling, path-finding,
    and sequence optimization problems.

    How it works:
    1. Choose a segment from parent1
    2. Copy remaining elements from parent2 in the order they appear
    3. This preserves relative ordering while introducing new combinations

    For Jump King, this could help preserve successful timing patterns
    while introducing new move combinations.
    """

    def __init__(self, crossover_rate: float = 0.8):
        super().__init__("Order", crossover_rate)

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform order crossover adapted for move sequences"""
        if len(parent1.moves) != len(parent2.moves):
            raise ValueError("Parents must have same sequence length")

        sequence_length = len(parent1.moves)

        # For move sequences, we'll use a modified approach:
        # Select a segment and try to preserve move type patterns

        if sequence_length <= 2:
            # Fall back to single point for short sequences
            return SinglePointCrossover().crossover(parent1, parent2)

        # Choose segment to preserve
        start = random.randint(0, sequence_length - 2)
        end = random.randint(start + 1, sequence_length)

        # Create children by preserving segments and filling with ordered elements
        child1_moves = self._create_ordered_child(parent1, parent2, start, end)
        child2_moves = self._create_ordered_child(parent2, parent1, start, end)

        # Create child genomes
        child1 = Genome(child1_moves, max(parent1.generation, parent2.generation) + 1)
        child2 = Genome(child2_moves, max(parent1.generation, parent2.generation) + 1)

        # Track lineage
        child1.parent_ids = (parent1.id, parent2.id)
        child2.parent_ids = (parent1.id, parent2.id)

        return child1, child2

    def _create_ordered_child(self, primary_parent: Genome, secondary_parent: Genome,
                              start: int, end: int) -> List[Move]:
        """Create child preserving order from parents"""
        sequence_length = len(primary_parent.moves)
        child_moves = [None] * sequence_length

        # Copy segment from primary parent
        for i in range(start, end):
            child_moves[i] = copy.deepcopy(primary_parent.moves[i])

        # Fill remaining positions with moves from secondary parent
        # Try to preserve move type patterns where possible
        secondary_idx = 0
        for i in range(sequence_length):
            if child_moves[i] is None:  # Position needs to be filled
                # Find next suitable move from secondary parent
                while secondary_idx < len(secondary_parent.moves):
                    candidate = secondary_parent.moves[secondary_idx]
                    secondary_idx += 1

                    # Use this move (could add more sophisticated matching logic)
                    child_moves[i] = copy.deepcopy(candidate)
                    break
                else:
                    # Fallback: create random move if we run out
                    child_moves[i] = Move.random()

        return child_moves


class AdaptiveCrossover(CrossoverOperator):
    """
    Adaptive Crossover: Changes behavior based on population state.

    AI CONCEPT: ADAPTIVE GENETIC OPERATORS
    =====================================
    Static operators use the same strategy throughout evolution. Adaptive
    operators modify their behavior based on:
    - Population diversity
    - Fitness improvement rate
    - Generation number
    - Problem-specific metrics

    This allows the algorithm to:
    - Use disruptive crossover when stuck in local optima
    - Use conservative crossover when making good progress
    - Balance exploration and exploitation automatically
    """

    def __init__(self, crossover_rate: float = 0.8):
        super().__init__("Adaptive", crossover_rate)

        # Available crossover strategies
        self.strategies = {
            'conservative': TwoPointCrossover(1.0),
            'balanced': TwoPointCrossover(1.0),
            'disruptive': UniformCrossover(1.0, 0.5),
            'exploratory': UniformCrossover(1.0, 0.3)
        }

        # Adaptation history
        self.strategy_history = []
        self.diversity_threshold = 0.1
        self.stagnation_threshold = 5

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform adaptive crossover"""
        # Choose strategy based on current state
        strategy_name = self._choose_strategy(parent1, parent2)
        strategy = self.strategies[strategy_name]

        # Record choice
        self.strategy_history.append(strategy_name)

        # Perform crossover with chosen strategy
        return strategy.crossover(parent1, parent2)

    def _choose_strategy(self, parent1: Genome, parent2: Genome) -> str:
        """Choose crossover strategy based on adaptive criteria"""

        # Calculate parent diversity
        parent_diversity = parent1.diversity_score([parent2])

        # Check recent strategy effectiveness (simplified)
        recent_history = self.strategy_history[-10:] if len(self.strategy_history) >= 10 else self.strategy_history

        # Decision logic
        if parent_diversity < self.diversity_threshold:
            # Parents are very similar, use disruptive crossover
            return 'disruptive'
        elif parent_diversity > 0.5:
            # Parents are very different, use conservative crossover
            return 'conservative'
        elif len(recent_history) >= 5 and recent_history.count('conservative') > 3:
            # Been using conservative too much, try something different
            return 'exploratory'
        else:
            # Default balanced approach
            return 'balanced'

    def get_strategy_distribution(self) -> Dict[str, float]:
        """Get distribution of strategies used"""
        if not self.strategy_history:
            return {}

        total = len(self.strategy_history)
        distribution = {}

        for strategy in self.strategies.keys():
            count = self.strategy_history.count(strategy)
            distribution[strategy] = count / total

        return distribution


class CrossoverManager:
    """
    Manages crossover operators and provides comparison tools.

    Allows easy switching between operators and analysis of their performance.
    """

    def __init__(self):
        self.operators: Dict[str, CrossoverOperator] = {}
        self.current_operator: Optional[CrossoverOperator] = None

    def register_operator(self, operator: CrossoverOperator):
        """Register a crossover operator"""
        self.operators[operator.name] = operator

    def set_operator(self, name: str):
        """Set the active crossover operator"""
        if name not in self.operators:
            raise ValueError(f"Unknown operator: {name}")
        self.current_operator = self.operators[name]

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform crossover using current operator"""
        if self.current_operator is None:
            raise ValueError("No crossover operator set")
        return self.current_operator.apply_crossover(parent1, parent2)

    def compare_operators(self, test_parents: List[Tuple[Genome, Genome]],
                          fitness_function) -> Dict[str, Dict]:
        """Compare different crossover operators"""
        results = {}

        for name, operator in self.operators.items():
            offspring_fitness = []

            for parent1, parent2 in test_parents:
                child1, child2 = operator.crossover(parent1, parent2)

                # Would need actual fitness evaluation here
                # This is a placeholder for testing
                offspring_fitness.extend([
                    random.uniform(0, 100),  # Placeholder fitness
                    random.uniform(0, 100)
                ])

            if offspring_fitness:
                results[name] = {
                    'mean_offspring_fitness': sum(offspring_fitness) / len(offspring_fitness),
                    'best_offspring_fitness': max(offspring_fitness),
                    'fitness_variance': sum((f - sum(offspring_fitness) / len(offspring_fitness)) ** 2
                                            for f in offspring_fitness) / len(offspring_fitness)
                }

        return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing Crossover Operators...")

    # Create test parents
    parent1 = Genome.random()
    parent2 = Genome.random()

    print(f"Parent 1: {[f'{m.move_type}-{m.strength}' for m in parent1.moves[:5]]}...")
    print(f"Parent 2: {[f'{m.move_type}-{m.strength}' for m in parent2.moves[:5]]}...")

    # Test different crossover operators
    operators = [
        SinglePointCrossover(),
        TwoPointCrossover(),
        UniformCrossover(),
        OrderCrossover(),
        AdaptiveCrossover()
    ]

    for operator in operators:
        print(f"\n{operator.name} Crossover:")
        child1, child2 = operator.apply_crossover(parent1, parent2)

        print(f"  Child 1: {[f'{m.move_type}-{m.strength}' for m in child1.moves[:5]]}...")
        print(f"  Child 2: {[f'{m.move_type}-{m.strength}' for m in child2.moves[:5]]}...")

    print("\nAll tests passed!")
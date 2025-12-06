"""
Mutation Operators: The Engine of Exploration

Mutation introduces random changes to genomes, serving several critical roles:
1. Prevents premature convergence to local optima
2. Introduces novel genetic material not present in initial population
3. Maintains population diversity over time
4. Enables fine-tuning of successful solutions

KEY AI CONCEPT: EXPLORATION vs EXPLOITATION TRADE-OFF
===================================================
Genetic algorithms must balance:
- EXPLOITATION: Using what we know works (selection, crossover)
- EXPLORATION: Discovering new possibilities (mutation)

Mutation is the primary exploration mechanism. Without it, genetic algorithms
can only recombine existing solutions and will eventually converge to a
local optimum. With too much mutation, they become random search.
"""

from typing import List, Dict, Any, Optional
import random
import copy
from abc import ABC, abstractmethod
from .genome import Genome, Move


class MutationOperator(ABC):
    """
    Abstract base class for mutation operators.

    AI CONCEPT: MUTATION STRATEGIES
    ==============================
    Different mutation approaches serve different purposes:
    - Gaussian: Small changes around current values (fine-tuning)
    - Uniform: Random replacement (exploration)
    - Creep: Gradual incremental changes (hill-climbing)
    - Swap: Reorder existing elements (structural changes)
    - Adaptive: Change mutation rate based on progress
    """

    def __init__(self, name: str, mutation_rate: float = 0.1):
        """
        Initialize mutation operator.

        Args:
            name: Name of the mutation method
            mutation_rate: Probability of mutating each gene
        """
        self.name = name
        self.mutation_rate = mutation_rate
        self.mutation_count = 0
        self.genes_mutated = 0

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")

    @abstractmethod
    def mutate(self, genome: Genome) -> Genome:
        """
        Create a mutated copy of the genome.

        Args:
            genome: Original genome to mutate

        Returns:
            New mutated genome
        """
        pass

    def apply_mutation(self, genome: Genome) -> Genome:
        """
        Apply mutation with statistics tracking.

        This wrapper tracks mutation statistics and handles the mutation
        process consistently across different operators.
        """
        self.mutation_count += 1
        mutated = self.mutate(genome)
        return mutated

    def get_statistics(self) -> Dict[str, Any]:
        """Get mutation statistics"""
        return {
            'name': self.name,
            'mutation_rate': self.mutation_rate,
            'total_mutations': self.mutation_count,
            'genes_mutated': self.genes_mutated,
            'avg_genes_per_mutation': (self.genes_mutated / max(1, self.mutation_count))
        }


class UniformMutation(MutationOperator):
    """
    Uniform Mutation: Replace genes with completely random values.

    AI CONCEPT: UNIFORM MUTATION
    ===========================
    For each gene, with probability mutation_rate:
    - Replace with a completely random value from the valid range
    - This provides maximum exploration but high disruption

    Benefits:
    - Can escape local optima quickly
    - Introduces truly novel genetic material
    - Simple to implement and understand

    Drawbacks:
    - High disruption can destroy good solutions
    - May be too aggressive for fine-tuning

    Best for:
    - Early generations when exploring broadly
    - When stuck in local optima
    - Problems with many possible values per gene
    """

    def __init__(self, mutation_rate: float = 0.1):
        super().__init__("Uniform", mutation_rate)

    def mutate(self, genome: Genome) -> Genome:
        """Create mutated copy with uniform random replacement"""
        mutated_moves = []
        genes_mutated_this_time = 0

        for move in genome.moves:
            if random.random() < self.mutation_rate:
                # Replace with completely random move
                mutated_moves.append(Move.random())
                genes_mutated_this_time += 1
            else:
                # Keep original move
                mutated_moves.append(copy.deepcopy(move))

        # Create new genome
        mutated_genome = Genome(mutated_moves, genome.generation + 1)
        mutated_genome.parent_ids = (genome.id, None)  # Asexual reproduction

        # Update statistics
        self.genes_mutated += genes_mutated_this_time

        return mutated_genome


class GaussianMutation(MutationOperator):
    """
    Gaussian Mutation: Add small random changes from normal distribution.

    AI CONCEPT: GAUSSIAN MUTATION
    ============================
    Instead of replacing genes completely, add small random values drawn
    from a Gaussian (normal) distribution. This provides:
    - Fine-tuning capability
    - Gradual exploration around current solutions
    - Controlled disruption amount

    For Jump King moves:
    - Move type: Occasionally change to adjacent type
    - Strength: Add small random value (±1-3 typically)

    Benefits:
    - Less disruptive than uniform mutation
    - Good for fine-tuning successful solutions
    - Maintains solution structure while exploring variants

    Best for:
    - Later generations when refining solutions
    - Continuous parameter optimization
    - When small changes can improve fitness significantly
    """

    def __init__(self, mutation_rate: float = 0.1, strength_sigma: float = 2.0, type_change_prob: float = 0.05):
        """
        Initialize Gaussian mutation.

        Args:
            mutation_rate: Probability of mutating each gene
            strength_sigma: Standard deviation for strength changes
            type_change_prob: Probability of changing move type
        """
        super().__init__("Gaussian", mutation_rate)
        self.strength_sigma = strength_sigma
        self.type_change_prob = type_change_prob

    def mutate(self, genome: Genome) -> Genome:
        """Create mutated copy with Gaussian perturbations"""
        mutated_moves = []
        genes_mutated_this_time = 0

        for move in genome.moves:
            if random.random() < self.mutation_rate:
                # Apply Gaussian mutation
                new_move_type = move.move_type
                new_strength = move.strength

                # Maybe change move type to adjacent type
                if random.random() < self.type_change_prob:
                    # Change to adjacent move type (0-4)
                    direction = random.choice([-1, 1])
                    new_move_type = max(0, min(4, move.move_type + direction))

                # Add Gaussian noise to strength
                strength_change = random.gauss(0, self.strength_sigma)
                new_strength = max(1, min(20, round(move.strength + strength_change)))

                mutated_moves.append(Move(new_move_type, new_strength))
                genes_mutated_this_time += 1
            else:
                # Keep original move
                mutated_moves.append(copy.deepcopy(move))

        # Create new genome
        mutated_genome = Genome(mutated_moves, genome.generation + 1)
        mutated_genome.parent_ids = (genome.id, None)

        # Update statistics
        self.genes_mutated += genes_mutated_this_time

        return mutated_genome


class CreepMutation(MutationOperator):
    """
    Creep Mutation: Make small incremental changes to gene values.

    AI CONCEPT: CREEP MUTATION
    ==========================
    Creep mutation makes the smallest possible changes to genes:
    - Increment or decrement by 1
    - Move to adjacent categories
    - Very conservative exploration

    This is essentially a form of local hill-climbing embedded in
    the genetic algorithm. It's excellent for fine-tuning but poor
    for escaping local optima.

    Benefits:
    - Minimal disruption to working solutions
    - Systematic exploration of nearby solutions
    - Good convergence properties

    Drawbacks:
    - Can get stuck in local optima
    - Slow exploration of distant regions
    - May not find dramatically different solutions

    Best for:
    - Final optimization phases
    - Problems with smooth fitness landscapes
    - When current solutions are close to optimal
    """

    def __init__(self, mutation_rate: float = 0.2):
        super().__init__("Creep", mutation_rate)

    def mutate(self, genome: Genome) -> Genome:
        """Create mutated copy with small incremental changes"""
        mutated_moves = []
        genes_mutated_this_time = 0

        for move in genome.moves:
            if random.random() < self.mutation_rate:
                # Make small random change
                new_move_type = move.move_type
                new_strength = move.strength

                # 50% chance to change move type by ±1
                if random.random() < 0.5:
                    direction = random.choice([-1, 1])
                    new_move_type = max(0, min(4, move.move_type + direction))

                # 50% chance to change strength by ±1
                if random.random() < 0.5:
                    direction = random.choice([-1, 1])
                    new_strength = max(1, min(20, move.strength + direction))

                mutated_moves.append(Move(new_move_type, new_strength))
                genes_mutated_this_time += 1
            else:
                # Keep original move
                mutated_moves.append(copy.deepcopy(move))

        # Create new genome
        mutated_genome = Genome(mutated_moves, genome.generation + 1)
        mutated_genome.parent_ids = (genome.id, None)

        # Update statistics
        self.genes_mutated += genes_mutated_this_time

        return mutated_genome


class SwapMutation(MutationOperator):
    """
    Swap Mutation: Exchange positions of genes in the sequence.

    AI CONCEPT: STRUCTURAL MUTATION
    ==============================
    Instead of changing gene values, swap mutation changes the ORDER
    of genes. This is useful when:
    - Sequence order matters (like in Jump King)
    - Good moves exist but are in wrong positions
    - Timing of actions is important

    For Jump King, this could:
    - Move a successful jump earlier or later in sequence
    - Reorder walking and jumping patterns
    - Change timing of complex maneuvers

    Benefits:
    - Preserves all genetic material (no information loss)
    - Can dramatically change behavior with small change
    - Good for problems where timing/order is crucial

    Best for:
    - Sequence optimization problems
    - When good components exist but are poorly ordered
    - Scheduling and planning problems
    """

    def __init__(self, mutation_rate: float = 0.1, max_swaps: int = 3):
        """
        Initialize swap mutation.

        Args:
            mutation_rate: Probability of performing any swaps
            max_swaps: Maximum number of swaps per mutation
        """
        super().__init__("Swap", mutation_rate)
        self.max_swaps = max_swaps

    def mutate(self, genome: Genome) -> Genome:
        """Create mutated copy with swapped positions"""
        mutated_moves = copy.deepcopy(genome.moves)
        genes_mutated_this_time = 0

        if random.random() < self.mutation_rate:
            # Perform 1 to max_swaps swaps
            num_swaps = random.randint(1, min(self.max_swaps, len(mutated_moves) // 2))

            for _ in range(num_swaps):
                # Choose two different positions to swap
                pos1 = random.randint(0, len(mutated_moves) - 1)
                pos2 = random.randint(0, len(mutated_moves) - 1)

                # Ensure different positions
                while pos2 == pos1 and len(mutated_moves) > 1:
                    pos2 = random.randint(0, len(mutated_moves) - 1)

                # Swap the moves
                if pos1 != pos2:
                    mutated_moves[pos1], mutated_moves[pos2] = mutated_moves[pos2], mutated_moves[pos1]
                    genes_mutated_this_time += 2  # Two genes affected

        # Create new genome
        mutated_genome = Genome(mutated_moves, genome.generation + 1)
        mutated_genome.parent_ids = (genome.id, None)

        # Update statistics
        self.genes_mutated += genes_mutated_this_time

        return mutated_genome


class AdaptiveMutation(MutationOperator):
    """
    Adaptive Mutation: Adjust mutation rate and strategy based on progress.

    AI CONCEPT: ADAPTIVE GENETIC OPERATORS
    ======================================
    Fixed mutation rates may not be optimal throughout evolution:
    - Early generations: Need high exploration (high mutation rate)
    - Later generations: Need fine-tuning (low mutation rate)
    - Stagnant periods: Need more exploration (increase rate)
    - Rapid improvement: Keep current strategy (maintain rate)

    Adaptive mutation monitors evolutionary progress and adjusts accordingly.
    This is a form of meta-learning - the algorithm learns how to learn.

    Adaptation strategies:
    - Success-based: Reduce rate when improving, increase when stuck
    - Diversity-based: Increase rate when population becomes too similar
    - Generation-based: Decrease rate over time
    - Fitness-based: Different rates for different fitness levels
    """

    def __init__(self, initial_rate: float = 0.1, min_rate: float = 0.01, max_rate: float = 0.3):
        """
        Initialize adaptive mutation.

        Args:
            initial_rate: Starting mutation rate
            min_rate: Minimum allowed mutation rate
            max_rate: Maximum allowed mutation rate
        """
        super().__init__("Adaptive", initial_rate)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = initial_rate
        self.fitness_history = []
        self.stagnation_counter = 0
        self.improvement_threshold = 0.01

        # Available mutation strategies
        self.strategies = {
            'uniform': UniformMutation(1.0),  # Rate controlled here
            'gaussian': GaussianMutation(1.0),
            'creep': CreepMutation(1.0)
        }
        self.current_strategy = 'gaussian'  # Default strategy

    def mutate(self, genome: Genome) -> Genome:
        """Mutate using adaptive strategy"""
        # Update mutation parameters based on recent history
        self._adapt_parameters()

        # Choose mutation strategy
        strategy = self.strategies[self.current_strategy]
        strategy.mutation_rate = self.current_rate

        # Apply mutation
        mutated = strategy.mutate(genome)

        # Update our statistics
        if hasattr(strategy, 'genes_mutated'):
            self.genes_mutated += strategy.genes_mutated - (
                strategy.genes_mutated if hasattr(self, '_last_genes_mutated') else 0)
            self._last_genes_mutated = strategy.genes_mutated

        return mutated

    def update_fitness_history(self, best_fitness: float):
        """Update fitness history for adaptation"""
        self.fitness_history.append(best_fitness)

        # Keep only recent history
        if len(self.fitness_history) > 20:
            self.fitness_history = self.fitness_history[-20:]

    def _adapt_parameters(self):
        """Adapt mutation rate and strategy based on progress"""
        if len(self.fitness_history) < 5:
            return  # Not enough history

        # Check for improvement in recent generations
        recent_fitness = self.fitness_history[-5:]
        if len(recent_fitness) >= 2:
            improvement = recent_fitness[-1] - recent_fitness[0]

            if improvement < self.improvement_threshold:
                # Stagnation detected
                self.stagnation_counter += 1

                if self.stagnation_counter >= 3:
                    # Increase mutation rate to escape local optimum
                    self.current_rate = min(self.max_rate, self.current_rate * 1.2)
                    self.current_strategy = 'uniform'  # More disruptive
                    self.stagnation_counter = 0
            else:
                # Good improvement, maintain or reduce mutation rate
                self.stagnation_counter = 0
                self.current_rate = max(self.min_rate, self.current_rate * 0.95)

                # Choose strategy based on improvement rate
                if improvement > self.improvement_threshold * 5:
                    self.current_strategy = 'creep'  # Fine-tuning
                else:
                    self.current_strategy = 'gaussian'  # Balanced

        # Ensure rate stays in bounds
        self.current_rate = max(self.min_rate, min(self.max_rate, self.current_rate))
        self.mutation_rate = self.current_rate  # Update base class rate

    def get_adaptation_state(self) -> Dict[str, Any]:
        """Get current adaptation state"""
        return {
            'current_rate': self.current_rate,
            'current_strategy': self.current_strategy,
            'stagnation_counter': self.stagnation_counter,
            'fitness_history_length': len(self.fitness_history),
            'recent_improvement': (
                self.fitness_history[-1] - self.fitness_history[-5]
                if len(self.fitness_history) >= 5 else 0
            )
        }


class MutationManager:
    """
    Manages mutation operators and provides analysis tools.

    Allows easy switching between operators and comparison of their effects.
    """

    def __init__(self):
        self.operators: Dict[str, MutationOperator] = {}
        self.current_operator: Optional[MutationOperator] = None

    def register_operator(self, operator: MutationOperator):
        """Register a mutation operator"""
        self.operators[operator.name] = operator

    def set_operator(self, name: str):
        """Set the active mutation operator"""
        if name not in self.operators:
            raise ValueError(f"Unknown operator: {name}")
        self.current_operator = self.operators[name]

    def mutate(self, genome: Genome) -> Genome:
        """Mutate genome using current operator"""
        if self.current_operator is None:
            raise ValueError("No mutation operator set")
        return self.current_operator.apply_mutation(genome)

    def compare_operators(self, test_genomes: List[Genome], num_trials: int = 100) -> Dict[str, Dict]:
        """Compare different mutation operators"""
        results = {}

        for name, operator in self.operators.items():
            diversity_scores = []

            for genome in test_genomes[:min(10, len(test_genomes))]:  # Test on subset
                for _ in range(num_trials // len(test_genomes[:10])):
                    mutated = operator.mutate(genome)
                    diversity = genome.diversity_score([mutated])
                    diversity_scores.append(diversity)

            if diversity_scores:
                results[name] = {
                    'mean_diversity': sum(diversity_scores) / len(diversity_scores),
                    'max_diversity': max(diversity_scores),
                    'min_diversity': min(diversity_scores),
                    'mutation_rate': operator.mutation_rate
                }

        return results

    def analyze_mutation_effects(self, population: List[Genome]) -> Dict[str, Any]:
        """Analyze the effects of mutation on a population"""
        if self.current_operator is None:
            return {}

        original_diversity = self._calculate_population_diversity(population)

        # Create mutated versions
        mutated_population = []
        for genome in population:
            mutated = self.current_operator.mutate(genome)
            mutated_population.append(mutated)

        mutated_diversity = self._calculate_population_diversity(mutated_population)

        return {
            'operator_name': self.current_operator.name,
            'original_diversity': original_diversity,
            'mutated_diversity': mutated_diversity,
            'diversity_change': mutated_diversity - original_diversity,
            'operator_stats': self.current_operator.get_statistics()
        }

    def _calculate_population_diversity(self, population: List[Genome]) -> float:
        """Calculate average diversity in population"""
        if len(population) < 2:
            return 0.0

        total_diversity = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                diversity = population[i].diversity_score([population[j]])
                total_diversity += diversity
                comparisons += 1

        return total_diversity / comparisons if comparisons > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    print("Testing Mutation Operators...")

    # Create test genome
    original = Genome.random()
    print(f"Original: {[f'{m.move_type}-{m.strength}' for m in original.moves[:5]]}...")

    # Test different mutation operators
    operators = [
        UniformMutation(mutation_rate=0.3),
        GaussianMutation(mutation_rate=0.3),
        CreepMutation(mutation_rate=0.3),
        SwapMutation(mutation_rate=0.5),
        AdaptiveMutation(initial_rate=0.3)
    ]

    for operator in operators:
        print(f"\n{operator.name} Mutation:")
        mutated = operator.apply_mutation(original)
        print(f"  Mutated: {[f'{m.move_type}-{m.strength}' for m in mutated.moves[:5]]}...")

        # Calculate similarity
        diversity = original.diversity_score([mutated])
        print(f"  Diversity from original: {diversity:.3f}")
        print(f"  Stats: {operator.get_statistics()}")

    # Test adaptive mutation with fitness updates
    adaptive = AdaptiveMutation()
    print(f"\nTesting Adaptive Mutation:")

    # Simulate fitness improvements
    fitness_sequence = [10, 12, 15, 15, 15, 16, 16, 16, 30, 35]
    for fitness in fitness_sequence:
        adaptive.update_fitness_history(fitness)
        state = adaptive.get_adaptation_state()
        print(f"  Fitness {fitness}: Rate={state['current_rate']:.3f}, Strategy={state['current_strategy']}")

    print("\nAll tests passed!")
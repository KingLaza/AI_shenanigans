"""
Population: Managing Collections of Solutions

The Population class handles groups of genomes and implements population-level
operations. This is where we see emergent behavior from individual solutions
working together.

KEY AI CONCEPT: POPULATION DYNAMICS
==================================
Individual solutions are interesting, but populations show complex behaviors:
- Diversity maintenance
- Convergence patterns
- Selection pressure effects
- Generational improvements

In nature, populations adapt to environments over time. Our artificial
population adapts to the Jump King challenge.
"""

from typing import List, Dict, Optional, Callable, Tuple
import random
import statistics
import copy
from .genome import Genome


class PopulationStatistics:
    """
    Tracks important population metrics over time.

    AI CONCEPT: PERFORMANCE MONITORING
    =================================
    Measuring progress is crucial in AI. Without good metrics, you can't
    tell if your algorithm is working or how to improve it.
    """

    def __init__(self):
        self.generation = 0
        self.best_fitness = float('-inf')
        self.average_fitness = 0.0
        self.worst_fitness = float('inf')
        self.diversity_score = 0.0
        self.convergence_rate = 0.0
        self.improvement_rate = 0.0

        # Historical data
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def update(self, population: List[Genome]):
        """Update statistics from current population"""
        if not population:
            return

        # Get fitness values (only from evaluated genomes)
        fitness_values = [g.fitness for g in population if g.fitness is not None]

        if fitness_values:
            self.best_fitness = max(fitness_values)
            self.average_fitness = statistics.mean(fitness_values)
            self.worst_fitness = min(fitness_values)
            self.fitness_history.append(self.average_fitness)

        # Calculate diversity (how different genomes are from each other)
        self.diversity_score = self._calculate_diversity(population)
        self.diversity_history.append(self.diversity_score)

        # Calculate improvement rate
        if len(self.fitness_history) >= 2:
            self.improvement_rate = (
                    self.fitness_history[-1] - self.fitness_history[-2]
            )

        self.generation += 1

    def _calculate_diversity(self, population: List[Genome]) -> float:
        """
        Measure how diverse the population is.

        High diversity = good exploration
        Low diversity = risk of premature convergence
        """
        if len(population) < 2:
            return 1.0

        total_distance = 0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Compare each pair of genomes
                genome1, genome2 = population[i], population[j]
                move_distances = [
                    move1.distance_to(move2)
                    for move1, move2 in zip(genome1.moves, genome2.moves)
                ]
                total_distance += sum(move_distances) / len(move_distances)
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def is_converged(self, threshold: float = 0.01) -> bool:
        """Check if population has converged (stopped improving)"""
        if len(self.fitness_history) < 10:
            return False

        recent_improvements = self.fitness_history[-10:]
        variation = statistics.pstdev(recent_improvements) if len(recent_improvements) > 1 else float('inf')
        return variation < threshold

    def __str__(self) -> str:
        return f"Gen {self.generation}: Best={self.best_fitness:.2f}, Avg={self.average_fitness:.2f}, Diversity={self.diversity_score:.3f}"


class Population:
    """
    Manages a collection of genomes and implements population-level operations.

    AI CONCEPT: COLLECTIVE INTELLIGENCE
    ==================================
    The population is more than the sum of its parts. Through interaction
    (crossover) and competition (selection), the group develops capabilities
    that no individual genome could achieve alone.

    Key responsibilities:
    - Maintain genetic diversity
    - Coordinate reproduction
    - Track evolutionary progress
    - Prevent premature convergence
    """

    def __init__(self, size: int = 100, initial_generation: int = 0):
        """
        Initialize a population.

        Args:
            size: Number of genomes in the population
            initial_generation: Starting generation number
        """
        self.size = size
        self.current_generation = initial_generation
        self.genomes: List[Genome] = []
        self.statistics = PopulationStatistics()

        # Performance tracking
        self.hall_of_fame: List[Genome] = []  # Best genomes ever found
        self.hall_of_fame_size = 10

        # Initialize with random genomes
        self.initialize_random()

    def initialize_random(self):
        """
        Create initial population with random genomes.

        AI CONCEPT: INITIALIZATION STRATEGIES
        ===================================
        Starting conditions matter! Random initialization provides maximum
        diversity, giving evolution the best chance to find good solutions.

        Alternative strategies:
        - Seeded: Start with some known good solutions
        - Hybrid: Mix random + heuristic solutions
        - Problem-specific: Use domain knowledge
        """
        self.genomes = [
            Genome.random(generation=self.current_generation)
            for _ in range(self.size)
        ]
        print(f"Initialized population of {self.size} random genomes")

    def add_genome(self, genome: Genome):
        """Add a genome to the population"""
        if len(self.genomes) < self.size:
            self.genomes.append(genome)
        else:
            raise ValueError(f"Population is full (size {self.size})")

    def get_best(self, n: int = 1) -> List[Genome]:
        """
        Get the n best genomes by fitness.

        AI CONCEPT: ELITISM
        ==================
        Always preserving the best solutions ensures we never lose progress.
        This is called "elitist selection" - the elite always survive.
        """
        # Filter out genomes that haven't been evaluated
        evaluated = [g for g in self.genomes if g.fitness is not None]

        if not evaluated:
            return []

        # Sort by fitness (descending) and return top n
        evaluated.sort(key=lambda g: g.fitness, reverse=True)
        return evaluated[:n]

    def get_worst(self, n: int = 1) -> List[Genome]:
        """Get the n worst genomes by fitness"""
        evaluated = [g for g in self.genomes if g.fitness is not None]

        if not evaluated:
            return []

        evaluated.sort(key=lambda g: g.fitness)
        return evaluated[:n]

    def update_hall_of_fame(self):
        """
        Update the hall of fame with best performers.

        AI CONCEPT: MEMORY & LEARNING
        ============================
        The hall of fame preserves the best solutions across ALL generations.
        This prevents losing exceptional individuals due to genetic drift.
        """
        current_best = self.get_best(self.size)

        # Add new genomes to hall of fame
        all_candidates = self.hall_of_fame + current_best

        # Remove duplicates and sort by fitness
        unique_candidates = {g.id: g for g in all_candidates}.values()
        sorted_candidates = sorted(
            unique_candidates,
            key=lambda g: g.fitness or float('-inf'),
            reverse=True
        )

        # Keep only top performers
        self.hall_of_fame = sorted_candidates[:self.hall_of_fame_size]

    def calculate_diversity_pressure(self) -> Dict[str, float]:
        """
        Calculate metrics related to population diversity.

        AI CONCEPT: DIVERSITY MAINTENANCE
        ===============================
        Genetic algorithms can suffer from premature convergence - when
        everyone becomes too similar too quickly. Monitoring diversity
        helps us detect and prevent this problem.

        Returns:
            Dictionary with diversity metrics
        """
        if len(self.genomes) < 2:
            return {'average_diversity': 1.0, 'min_diversity': 1.0}

        diversities = []
        for genome in self.genomes:
            others = [g for g in self.genomes if g.id != genome.id]
            diversity = genome.diversity_score(others)
            diversities.append(diversity)

        return {
            'average_diversity': statistics.mean(diversities),
            'min_diversity': min(diversities),
            'max_diversity': max(diversities),
            'diversity_std': statistics.pstdev(diversities) if len(diversities) > 1 else 0.0
        }

    def apply_diversity_pressure(self, min_diversity: float = 0.1):
        """
        Remove genomes that are too similar to others.

        AI CONCEPT: NICHING & SPECIATION
        ===============================
        Sometimes we need to actively maintain diversity by removing
        overly similar individuals. This prevents the population from
        collapsing to a single solution.
        """
        if len(self.genomes) <= 2:
            return

        to_remove = []
        diversity_metrics = self.calculate_diversity_pressure()

        if diversity_metrics['average_diversity'] < min_diversity:
            # Find most similar pairs and remove the worse performer
            for i, genome1 in enumerate(self.genomes):
                for j, genome2 in enumerate(self.genomes[i + 1:], i + 1):
                    similarity = 1.0 - genome1.diversity_score([genome2])

                    if similarity > (1.0 - min_diversity):
                        # These genomes are too similar
                        worse_genome = (
                            genome1 if (genome1.fitness or 0) < (genome2.fitness or 0)
                            else genome2
                        )
                        if worse_genome not in to_remove:
                            to_remove.append(worse_genome)

        # Remove overly similar genomes
        for genome in to_remove[:len(self.genomes) // 4]:  # Don't remove more than 25%
            if genome in self.genomes:
                self.genomes.remove(genome)

        # Replace with random genomes to maintain population size
        while len(self.genomes) < self.size:
            self.genomes.append(Genome.random(self.current_generation))

    def next_generation(self):
        """
        Advance to the next generation.

        This is typically called after selection, crossover, and mutation
        have created a new set of genomes.
        """
        self.current_generation += 1

        # Update generation numbers for all genomes
        for genome in self.genomes:
            if genome.generation < self.current_generation:
                genome.generation = self.current_generation

        # Update statistics
        self.statistics.update(self.genomes)
        self.update_hall_of_fame()

    def get_fitness_distribution(self) -> Dict[str, float]:
        """Get statistical summary of population fitness"""
        fitness_values = [g.fitness for g in self.genomes if g.fitness is not None]

        if not fitness_values:
            return {'count': 0}

        return {
            'count': len(fitness_values),
            'mean': statistics.mean(fitness_values),
            'median': statistics.median(fitness_values),
            'std_dev': statistics.pstdev(fitness_values) if len(fitness_values) > 1 else 0.0,
            'min': min(fitness_values),
            'max': max(fitness_values),
            'range': max(fitness_values) - min(fitness_values)
        }

    def export_population(self) -> List[Dict]:
        """Export population data for analysis"""
        return [genome.to_dict() for genome in self.genomes]

    def import_population(self, data: List[Dict]):
        """Import population from exported data"""
        self.genomes = [Genome.from_dict(genome_data) for genome_data in data]
        self.size = len(self.genomes)
        if self.genomes:
            self.current_generation = max(g.generation for g in self.genomes)

    def reset(self, keep_hall_of_fame: bool = True):
        """
        Reset population to random state.

        Useful for:
        - Restarting evolution with different parameters
        - Escaping local optima
        - Comparing different initialization strategies
        """
        if not keep_hall_of_fame:
            self.hall_of_fame.clear()

        self.current_generation = 0
        self.statistics = PopulationStatistics()
        self.initialize_random()

    def __len__(self) -> int:
        """Return population size"""
        return len(self.genomes)

    def __iter__(self):
        """Allow iteration over genomes"""
        return iter(self.genomes)

    def __getitem__(self, index) -> Genome:
        """Allow indexing into population"""
        return self.genomes[index]

    def __str__(self) -> str:
        """Human-readable representation"""
        return f"Population(size={len(self.genomes)}, generation={self.current_generation})"

    def summary(self) -> str:
        """Detailed population summary"""
        fitness_dist = self.get_fitness_distribution()
        diversity_metrics = self.calculate_diversity_pressure()

        lines = [
            f"Population Summary (Generation {self.current_generation})",
            f"  Size: {len(self.genomes)}",
            f"  Evaluated: {fitness_dist.get('count', 0)}",
        ]

        if fitness_dist.get('count', 0) > 0:
            lines.extend([
                f"  Best Fitness: {fitness_dist['max']:.3f}",
                f"  Average Fitness: {fitness_dist['mean']:.3f}",
                f"  Worst Fitness: {fitness_dist['min']:.3f}",
                f"  Fitness Std Dev: {fitness_dist['std_dev']:.3f}",
            ])

        lines.extend([
            f"  Average Diversity: {diversity_metrics['average_diversity']:.3f}",
            f"  Hall of Fame: {len(self.hall_of_fame)} genomes",
        ])

        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Population class...")

    # Create a population
    pop = Population(size=50)
    print(f"Created: {pop}")
    print(pop.summary())

    # Simulate some fitness evaluations
    for genome in pop:
        genome.fitness = random.uniform(0, 100)

    pop.next_generation()
    print("\nAfter fitness evaluation:")
    print(pop.summary())

    # Test best/worst selection
    best_genomes = pop.get_best(5)
    print(f"\nTop 5 genomes:")
    for i, genome in enumerate(best_genomes, 1):
        print(f"  {i}. {genome}")

    print("All tests passed!")
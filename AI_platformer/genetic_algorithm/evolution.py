"""
Evolution Engine: The Orchestrator of Artificial Evolution

The Evolution Engine brings together all genetic algorithm components to
create a complete evolutionary system. It implements the main GA loop and
manages the evolutionary process from start to finish.

KEY AI CONCEPT: THE GENETIC ALGORITHM LOOP
==========================================
The canonical genetic algorithm follows this pattern:

1. Initialize: Create random population
2. Evaluate: Measure fitness of all individuals
3. Select: Choose parents based on fitness
4. Reproduce: Create offspring via crossover and mutation
5. Replace: Form new generation from offspring and/or parents
6. Repeat: Continue until termination criteria met

This simple loop, when repeated many times, can solve complex optimization
problems that would be intractable with traditional methods.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import copy
from dataclasses import dataclass, field
from enum import Enum

from .genome import Genome
from .population import Population
from .fitness import FitnessFunction, FitnessEvaluator
from .selection import SelectionStrategy, TournamentSelection
from .crossover import CrossoverOperator, TwoPointCrossover
from .mutation import MutationOperator, GaussianMutation


class TerminationReason(Enum):
    """Reasons why evolution might stop"""
    MAX_GENERATIONS = "max_generations_reached"
    FITNESS_THRESHOLD = "fitness_threshold_reached"
    CONVERGENCE = "population_converged"
    STAGNATION = "fitness_stagnated"
    TIME_LIMIT = "time_limit_exceeded"
    USER_STOPPED = "user_intervention"
    ERROR = "error_occurred"


@dataclass
class EvolutionConfig:
    """
    Configuration for evolutionary runs.

    AI CONCEPT: HYPERPARAMETER TUNING
    =================================
    These parameters control how evolution behaves. Finding good values
    is an art and science - they need to be tuned for each problem:

    - population_size: Larger = more diversity, slower convergence
    - max_generations: More = better solutions, longer runtime
    - elite_size: Preserves best solutions across generations
    - mutation_rate: Higher = more exploration, risk of disruption
    - crossover_rate: Higher = more exploitation of good combinations
    """

    # Population parameters
    population_size: int = 100
    max_generations: int = 50
    elite_size: int = 5

    # Genetic operator rates
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # Termination criteria
    fitness_threshold: Optional[float] = None
    convergence_threshold: float = 0.001
    stagnation_generations: int = 10
    max_runtime_seconds: Optional[float] = None

    # Diversity maintenance
    maintain_diversity: bool = True
    min_diversity: float = 0.1
    diversity_check_interval: int = 5

    # Replacement strategy
    replacement_strategy: str = "generational"  # "generational", "steady_state", "elitist"

    # Logging and analysis
    verbose: bool = True
    log_interval: int = 5
    track_lineage: bool = False
    save_checkpoints: bool = False


class EvolutionEngine:
    """
    The main genetic algorithm engine that orchestrates evolution.

    AI CONCEPT: EVOLUTIONARY COMPUTATION ARCHITECTURE
    ===============================================
    The engine follows the "Strategy Pattern" - it uses pluggable components:
    - Any fitness function implementing the FitnessFunction interface
    - Any selection strategy implementing SelectionStrategy
    - Any crossover operator implementing CrossoverOperator
    - Any mutation operator implementing MutationOperator

    This design allows easy experimentation with different algorithms.
    """

    def __init__(self,
                 fitness_function: FitnessFunction,
                 selection_strategy: Optional[SelectionStrategy] = None,
                 crossover_operator: Optional[CrossoverOperator] = None,
                 mutation_operator: Optional[MutationOperator] = None,
                 config: Optional[EvolutionConfig] = None):
        """
        Initialize the evolution engine.

        Args:
            fitness_function: How to evaluate genome quality
            selection_strategy: How to choose parents (default: Tournament)
            crossover_operator: How to create offspring (default: TwoPoint)
            mutation_operator: How to introduce variation (default: Gaussian)
            config: Evolution parameters (default: EvolutionConfig())
        """

        # Core components
        self.fitness_function = fitness_function
        self.fitness_evaluator = FitnessEvaluator(fitness_function)

        # Genetic operators (with sensible defaults)
        self.selection_strategy = selection_strategy or TournamentSelection(tournament_size=3)
        self.crossover_operator = crossover_operator or TwoPointCrossover()
        self.mutation_operator = mutation_operator or GaussianMutation()

        # Configuration
        self.config = config or EvolutionConfig()

        # Update operator rates from config
        self.crossover_operator.crossover_rate = self.config.crossover_rate
        self.mutation_operator.mutation_rate = self.config.mutation_rate

        # Evolution state
        self.population: Optional[Population] = None
        self.current_generation = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.best_ever: Optional[Genome] = None
        self.start_time: Optional[float] = None
        self.is_running = False

        # Callbacks for external monitoring
        self.generation_callbacks: List[Callable] = []
        self.termination_callbacks: List[Callable] = []

    def add_generation_callback(self, callback: Callable):
        """Add callback function called after each generation"""
        self.generation_callbacks.append(callback)

    def add_termination_callback(self, callback: Callable):
        """Add callback function called when evolution terminates"""
        self.termination_callbacks.append(callback)

    def evolve(self, simulation_function: Callable[[List[Genome]], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run the complete evolutionary process.

        AI CONCEPT: SIMULATION-BASED OPTIMIZATION
        =======================================
        This method implements simulation-based optimization - we can't
        calculate fitness analytically, so we run simulations to evaluate
        each genome's performance.

        Args:
            simulation_function: Function that takes genomes and returns performance data
                                Function signature: List[Genome] -> List[Dict[str, Any]]

        Returns:
            Dictionary with evolution results and statistics
        """

        self.start_time = time.time()
        self.is_running = True
        termination_reason = None

        try:
            # Initialize population
            self._initialize_population()

            if self.config.verbose:
                print(f"🧬 Starting evolution with {self.config.population_size} genomes")
                print(f"📊 Target: {self.config.max_generations} generations")
                print("=" * 60)

            # Main evolution loop
            for generation in range(self.config.max_generations):
                if not self.is_running:
                    termination_reason = TerminationReason.USER_STOPPED
                    break

                self.current_generation = generation

                # Evaluate population
                self._evaluate_population(simulation_function)

                # Check termination criteria
                termination_reason = self._check_termination()
                if termination_reason:
                    break

                # Create next generation
                self._create_next_generation()

                # Update population generation
                self.population.next_generation()

                # Log progress
                if generation % self.config.log_interval == 0 or generation == self.config.max_generations - 1:
                    self._log_generation()

                # Call generation callbacks
                for callback in self.generation_callbacks:
                    try:
                        callback(self, generation)
                    except Exception as e:
                        if self.config.verbose:
                            print(f"⚠️  Generation callback error: {e}")

                # Maintain diversity if needed
                if (self.config.maintain_diversity and
                        generation % self.config.diversity_check_interval == 0):
                    self._maintain_diversity()

            # Evolution completed
            if not termination_reason:
                termination_reason = TerminationReason.MAX_GENERATIONS

        except Exception as e:
            termination_reason = TerminationReason.ERROR
            if self.config.verbose:
                print(f"❌ Evolution error: {e}")

        finally:
            self.is_running = False

        # Final evaluation and results
        results = self._compile_results(termination_reason)

        # Call termination callbacks
        for callback in self.termination_callbacks:
            try:
                callback(self, results)
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Termination callback error: {e}")

        return results

    def _initialize_population(self):
        """Initialize the population for evolution"""
        self.population = Population(
            size=self.config.population_size,
            initial_generation=0
        )

        self.current_generation = 0
        self.evolution_history.clear()
        self.best_ever = None

        if self.config.verbose:
            print(f"🎲 Initialized population of {len(self.population)} random genomes")

    def _evaluate_population(self, simulation_function: Callable):
        """Evaluate fitness for all genomes in population"""

        if self.config.verbose and self.current_generation % self.config.log_interval == 0:
            print(f"🔬 Evaluating generation {self.current_generation}...")

        # Run simulations for all genomes
        performance_results = simulation_function(list(self.population.genomes))

        if len(performance_results) != len(self.population.genomes):
            raise ValueError(
                f"Simulation returned {len(performance_results)} results for {len(self.population.genomes)} genomes")

        # Calculate fitness for each genome
        fitness_scores = self.fitness_evaluator.evaluate_population(
            self.population, performance_results
        )

        # Update best ever
        current_best = self.population.get_best(1)
        if current_best and (self.best_ever is None or current_best[0].fitness > self.best_ever.fitness):
            self.best_ever = copy.deepcopy(current_best[0])

    def _check_termination(self) -> Optional[TerminationReason]:
        """Check if evolution should terminate"""

        # Time limit check
        if self.config.max_runtime_seconds:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.max_runtime_seconds:
                return TerminationReason.TIME_LIMIT

        # Fitness threshold check
        if self.config.fitness_threshold and self.best_ever:
            if self.best_ever.fitness >= self.config.fitness_threshold:
                return TerminationReason.FITNESS_THRESHOLD

        # Convergence check
        if self.population.statistics.is_converged(self.config.convergence_threshold):
            return TerminationReason.CONVERGENCE

        # Stagnation check
        if len(self.evolution_history) >= self.config.stagnation_generations:
            recent_best = [h['best_fitness'] for h in self.evolution_history[-self.config.stagnation_generations:]]
            if max(recent_best) - min(recent_best) < self.config.convergence_threshold:
                return TerminationReason.STAGNATION

        return None

    def _create_next_generation(self):
        """Create the next generation using genetic operators"""

        if self.config.replacement_strategy == "generational":
            self._generational_replacement()
        elif self.config.replacement_strategy == "steady_state":
            self._steady_state_replacement()
        elif self.config.replacement_strategy == "elitist":
            self._elitist_replacement()
        else:
            raise ValueError(f"Unknown replacement strategy: {self.config.replacement_strategy}")

    def _generational_replacement(self):
        """Replace entire population each generation"""

        current_genomes = list(self.population.genomes)

        # Preserve elite
        elite = self.population.get_best(self.config.elite_size)

        # Calculate number of offspring needed
        num_offspring_pairs = (self.config.population_size - len(elite)) // 2

        # Select parents and create offspring
        parent_pairs = self.selection_strategy.select_parents(current_genomes, num_offspring_pairs)

        offspring = []
        for parent1, parent2 in parent_pairs:
            # Crossover
            child1, child2 = self.crossover_operator.apply_crossover(parent1, parent2)

            # Mutation
            child1 = self.mutation_operator.apply_mutation(child1)
            child2 = self.mutation_operator.apply_mutation(child2)

            offspring.extend([child1, child2])

        # Fill any remaining slots with mutated elite
        while len(offspring) + len(elite) < self.config.population_size:
            parent = elite[len(offspring) % len(elite)] if elite else current_genomes[0]
            mutated = self.mutation_operator.apply_mutation(parent)
            offspring.append(mutated)

        # Create new population
        new_genomes = elite + offspring[:self.config.population_size - len(elite)]
        self.population.genomes = new_genomes

    def _steady_state_replacement(self):
        """Replace only worst individuals each generation"""

        current_genomes = list(self.population.genomes)

        # Select a few parent pairs
        num_pairs = min(5, len(current_genomes) // 4)
        parent_pairs = self.selection_strategy.select_parents(current_genomes, num_pairs)

        offspring = []
        for parent1, parent2 in parent_pairs:
            child1, child2 = self.crossover_operator.apply_crossover(parent1, parent2)
            child1 = self.mutation_operator.apply_mutation(child1)
            child2 = self.mutation_operator.apply_mutation(child2)
            offspring.extend([child1, child2])

        # Replace worst individuals with offspring
        worst = self.population.get_worst(len(offspring))
        for i, worst_genome in enumerate(worst):
            if i < len(offspring):
                idx = self.population.genomes.index(worst_genome)
                self.population.genomes[idx] = offspring[i]

    def _elitist_replacement(self):
        """Always preserve best, replace rest"""
        elite_percentage = 0.2  # Keep top 20%
        elite_size = max(1, int(self.config.population_size * elite_percentage))

        current_genomes = list(self.population.genomes)
        elite = self.population.get_best(elite_size)

        # Create offspring from elite and selected parents
        num_offspring_pairs = (self.config.population_size - elite_size) // 2
        parent_pairs = self.selection_strategy.select_parents(current_genomes, num_offspring_pairs)

        offspring = []
        for parent1, parent2 in parent_pairs:
            child1, child2 = self.crossover_operator.apply_crossover(parent1, parent2)
            child1 = self.mutation_operator.apply_mutation(child1)
            child2 = self.mutation_operator.apply_mutation(child2)
            offspring.extend([child1, child2])

        # Combine elite and offspring
        new_genomes = elite + offspring[:self.config.population_size - elite_size]
        self.population.genomes = new_genomes

    def _maintain_diversity(self):
        """Apply diversity maintenance if needed"""
        diversity_metrics = self.population.calculate_diversity_pressure()

        if diversity_metrics['average_diversity'] < self.config.min_diversity:
            if self.config.verbose:
                print(f"🌱 Applying diversity pressure (current: {diversity_metrics['average_diversity']:.3f})")

            self.population.apply_diversity_pressure(self.config.min_diversity)

    def _log_generation(self):
        """Log current generation statistics"""
        stats = self.population.statistics
        fitness_dist = self.population.get_fitness_distribution()

        # Store in history
        generation_data = {
            'generation': self.current_generation,
            'best_fitness': stats.best_fitness,
            'average_fitness': stats.average_fitness,
            'worst_fitness': stats.worst_fitness,
            'diversity': stats.diversity_score,
            'improvement_rate': stats.improvement_rate,
            'evaluation_count': self.fitness_function.evaluation_count
        }
        self.evolution_history.append(generation_data)

        if self.config.verbose:
            elapsed = time.time() - self.start_time
            print(f"Gen {self.current_generation:3d} | "
                  f"Best: {stats.best_fitness:7.2f} | "
                  f"Avg: {stats.average_fitness:7.2f} | "
                  f"Div: {stats.diversity_score:5.3f} | "
                  f"Time: {elapsed:6.1f}s")

    def _compile_results(self, termination_reason: TerminationReason) -> Dict[str, Any]:
        """Compile final evolution results"""

        end_time = time.time()
        total_runtime = end_time - self.start_time if self.start_time else 0

        results = {
            # Basic info
            'termination_reason': termination_reason.value,
            'generations_completed': self.current_generation,
            'total_runtime': total_runtime,
            'evaluations_performed': self.fitness_function.evaluation_count,

            # Best solutions
            'best_genome': self.best_ever,
            'best_fitness': self.best_ever.fitness if self.best_ever else None,
            'final_population': copy.deepcopy(self.population.genomes),

            # Evolution history
            'evolution_history': self.evolution_history,
            'final_population_stats': self.population.get_fitness_distribution(),
            'diversity_metrics': self.population.calculate_diversity_pressure(),

            # Configuration used
            'config': self.config,

            # Component statistics
            'fitness_function_stats': self.fitness_evaluator.get_statistics(),
            'selection_stats': getattr(self.selection_strategy, 'selection_history', []),
            'crossover_stats': self.crossover_operator.get_statistics(),
            'mutation_stats': self.mutation_operator.get_statistics()
        }

        if self.config.verbose:
            print("\n" + "=" * 60)
            print(f"🏁 Evolution completed: {termination_reason.value}")
            print(f"⏱️  Runtime: {total_runtime:.1f} seconds")
            print(f"🧬 Generations: {self.current_generation}")
            print(f"🔬 Evaluations: {self.fitness_function.evaluation_count}")
            if self.best_ever:
                print(f"🏆 Best fitness: {self.best_ever.fitness:.3f}")
                print(f"🆔 Best genome ID: {self.best_ever.id}")

        return results

    def stop_evolution(self):
        """Stop evolution gracefully"""
        self.is_running = False

    def get_current_best(self) -> Optional[Genome]:
        """Get current best genome"""
        if self.population:
            best = self.population.get_best(1)
            return best[0] if best else None
        return None

    def export_state(self) -> Dict[str, Any]:
        """Export current evolution state for checkpointing"""
        return {
            'current_generation': self.current_generation,
            'population': self.population.export_population() if self.population else None,
            'evolution_history': self.evolution_history,
            'best_ever': self.best_ever.to_dict() if self.best_ever else None,
            'config': self.config
        }

    def import_state(self, state: Dict[str, Any]):
        """Import evolution state from checkpoint"""
        self.current_generation = state['current_generation']
        self.evolution_history = state['evolution_history']

        if state['population']:
            self.population = Population(size=len(state['population']))
            self.population.import_population(state['population'])

        if state['best_ever']:
            self.best_ever = Genome.from_dict(state['best_ever'])

        if state['config']:
            self.config = state['config']


# Example usage and testing
if __name__ == "__main__":
    print("Testing Evolution Engine...")

    # Import required components
    from .fitness import HeightOnlyFitness


    # Simple simulation function for testing
    def dummy_simulation(genomes: List[Genome]) -> List[Dict[str, Any]]:
        """Dummy simulation that gives random fitness"""
        import random
        results = []
        for genome in genomes:
            # Simulate some performance metrics
            max_height = random.uniform(0, 100) + sum(m.strength for m in genome.moves[:5])
            results.append({
                'max_height_reached': max_height,
                'final_height': max_height * 0.8,
                'survival_time': random.randint(100, 1000),
                'total_moves_used': random.randint(5, 20),
                'deaths': random.randint(0, 3)
            })
        return results


    # Create evolution engine
    fitness_fn = HeightOnlyFitness()
    config = EvolutionConfig(
        population_size=20,
        max_generations=10,
        verbose=True,
        log_interval=2
    )

    engine = EvolutionEngine(
        fitness_function=fitness_fn,
        config=config
    )

    # Run evolution
    print("🧪 Running test evolution...")
    results = engine.evolve(dummy_simulation)

    print(f"\n✅ Test completed!")
    print(f"Best fitness achieved: {results['best_fitness']:.2f}")
    print(f"Generations completed: {results['generations_completed']}")
    print("Evolution engine test passed!")
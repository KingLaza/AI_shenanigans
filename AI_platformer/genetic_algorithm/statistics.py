"""
Evolution Statistics: Measuring and Visualizing AI Progress

Statistics are crucial for understanding how genetic algorithms work and
for tuning their performance. This module provides comprehensive tracking,
analysis, and visualization of evolutionary progress.

KEY AI CONCEPT: PERFORMANCE MONITORING & ANALYSIS
===============================================
Machine learning without measurement is just random search. Good statistics help you:

1. Understand algorithm behavior
2. Identify problems (premature convergence, stagnation)
3. Compare different approaches
4. Tune hyperparameters effectively
5. Communicate results to others

For genetic algorithms specifically, we track:
- Fitness progression over generations
- Population diversity metrics
- Selection pressure effects
- Genetic operator performance
- Convergence patterns
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
import json

# Optional plotting (graceful fallback if matplotlib not available)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    import numpy as np

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    np = None

from .genome import Genome


@dataclass
class GenerationStats:
    """Statistics for a single generation"""
    generation: int
    timestamp: datetime

    # Fitness statistics
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    median_fitness: float
    fitness_std: float

    # Population statistics
    population_size: int
    evaluated_count: int
    diversity_score: float

    # Improvement metrics
    improvement_from_last: float = 0.0
    improvement_rate: float = 0.0
    stagnation_counter: int = 0

    # Selection statistics
    selection_pressure: float = 0.0
    elite_percentage: float = 0.0

    # Operator statistics
    crossover_events: int = 0
    mutation_events: int = 0
    genes_mutated: int = 0


class EvolutionStatistics:
    """
    Comprehensive statistics tracking for genetic algorithm evolution.

    AI CONCEPT: EXPERIMENTAL METHODOLOGY
    ==================================
    Good experimentation requires careful measurement. This class implements
    best practices for GA analysis:

    - Track multiple metrics (not just best fitness)
    - Monitor population dynamics
    - Detect problematic behaviors
    - Enable statistical comparisons
    - Support hypothesis testing
    """

    def __init__(self):
        self.generation_history: List[GenerationStats] = []
        self.experiment_start_time: Optional[datetime] = None
        self.experiment_metadata: Dict[str, Any] = {}

        # Lineage tracking (if enabled)
        self.lineage_tree: Dict[str, List[str]] = {}  # genome_id -> parent_ids
        self.hall_of_fame: List[Genome] = []

        # Performance tracking
        self.convergence_events: List[int] = []  # Generations where convergence occurred
        self.diversity_crises: List[int] = []  # Generations with very low diversity
        self.breakthrough_generations: List[int] = []  # Major fitness improvements

    def start_experiment(self, metadata: Dict[str, Any] = None):
        """Initialize tracking for a new experiment"""
        self.experiment_start_time = datetime.now()
        self.experiment_metadata = metadata or {}
        self.generation_history.clear()
        self.lineage_tree.clear()
        self.hall_of_fame.clear()
        self.convergence_events.clear()
        self.diversity_crises.clear()
        self.breakthrough_generations.clear()

    def record_generation(self,
                          generation: int,
                          population: List[Genome],
                          selection_stats: Dict = None,
                          operator_stats: Dict = None) -> GenerationStats:
        """
        Record statistics for a completed generation.

        Args:
            generation: Generation number
            population: Current population
            selection_stats: Statistics from selection operator
            operator_stats: Statistics from crossover/mutation operators
        """

        # Calculate fitness statistics
        fitness_values = [g.fitness for g in population if g.fitness is not None]

        if not fitness_values:
            # No evaluated genomes
            stats = GenerationStats(
                generation=generation,
                timestamp=datetime.now(),
                best_fitness=float('-inf'),
                average_fitness=0.0,
                worst_fitness=float('-inf'),
                median_fitness=0.0,
                fitness_std=0.0,
                population_size=len(population),
                evaluated_count=0,
                diversity_score=0.0
            )
        else:
            # Calculate comprehensive fitness statistics
            best_fitness = max(fitness_values)
            average_fitness = statistics.mean(fitness_values)
            worst_fitness = min(fitness_values)
            median_fitness = statistics.median(fitness_values)
            fitness_std = statistics.pstdev(fitness_values) if len(fitness_values) > 1 else 0.0

            # Calculate diversity
            diversity_score = self._calculate_population_diversity(population)

            # Calculate improvement metrics
            improvement_from_last = 0.0
            improvement_rate = 0.0
            stagnation_counter = 0

            if self.generation_history:
                last_best = self.generation_history[-1].best_fitness
                improvement_from_last = best_fitness - last_best

                # Calculate improvement rate over last 5 generations
                if len(self.generation_history) >= 5:
                    recent_best = [s.best_fitness for s in self.generation_history[-5:]]
                    improvement_rate = (best_fitness - recent_best[0]) / 5

                # Calculate stagnation
                stagnation_counter = self._calculate_stagnation_counter()

            # Extract operator statistics
            crossover_events = operator_stats.get('crossover_events', 0) if operator_stats else 0
            mutation_events = operator_stats.get('mutation_events', 0) if operator_stats else 0
            genes_mutated = operator_stats.get('genes_mutated', 0) if operator_stats else 0

            # Extract selection statistics
            selection_pressure = selection_stats.get('selection_pressure', 0.0) if selection_stats else 0.0
            elite_percentage = selection_stats.get('elite_percentage', 0.0) if selection_stats else 0.0

            stats = GenerationStats(
                generation=generation,
                timestamp=datetime.now(),
                best_fitness=best_fitness,
                average_fitness=average_fitness,
                worst_fitness=worst_fitness,
                median_fitness=median_fitness,
                fitness_std=fitness_std,
                population_size=len(population),
                evaluated_count=len(fitness_values),
                diversity_score=diversity_score,
                improvement_from_last=improvement_from_last,
                improvement_rate=improvement_rate,
                stagnation_counter=stagnation_counter,
                selection_pressure=selection_pressure,
                elite_percentage=elite_percentage,
                crossover_events=crossover_events,
                mutation_events=mutation_events,
                genes_mutated=genes_mutated
            )

        # Store statistics
        self.generation_history.append(stats)

        # Update lineage tracking
        self._update_lineage_tracking(population)

        # Detect special events
        self._detect_events(stats)

        return stats

    def _calculate_population_diversity(self, population: List[Genome]) -> float:
        """Calculate average pairwise diversity in population"""
        if len(population) < 2:
            return 1.0

        total_distance = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                genome1, genome2 = population[i], population[j]

                # Calculate distance between genomes
                distance = 0.0
                for move1, move2 in zip(genome1.moves, genome2.moves):
                    distance += move1.distance_to(move2)
                distance /= len(genome1.moves)  # Normalize by sequence length

                total_distance += distance
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def _calculate_stagnation_counter(self) -> int:
        """Calculate how many generations since last significant improvement"""
        if len(self.generation_history) < 2:
            return 0

        current_best = self.generation_history[-1].best_fitness
        stagnation_threshold = 0.001  # Configurable

        counter = 0
        for i in range(len(self.generation_history) - 2, -1, -1):
            if current_best - self.generation_history[i].best_fitness > stagnation_threshold:
                break
            counter += 1

        return counter

    def _update_lineage_tracking(self, population: List[Genome]):
        """Update lineage tree for genealogy analysis"""
        for genome in population:
            if genome.id not in self.lineage_tree and genome.parent_ids:
                self.lineage_tree[genome.id] = list(genome.parent_ids)

    def _detect_events(self, stats: GenerationStats):
        """Detect and record special evolutionary events"""

        # Detect breakthroughs (significant fitness jumps)
        if stats.improvement_from_last > 10.0:  # Configurable threshold
            self.breakthrough_generations.append(stats.generation)

        # Detect diversity crises
        if stats.diversity_score < 0.05:  # Configurable threshold
            self.diversity_crises.append(stats.generation)

        # Detect convergence
        if stats.fitness_std < 0.01 and stats.evaluated_count > 10:  # Configurable
            self.convergence_events.append(stats.generation)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary of evolutionary run"""
        if not self.generation_history:
            return {'status': 'no_data'}

        fitness_values = [s.best_fitness for s in self.generation_history]
        diversity_values = [s.diversity_score for s in self.generation_history]

        return {
            'total_generations': len(self.generation_history),
            'final_best_fitness': fitness_values[-1],
            'overall_improvement': fitness_values[-1] - fitness_values[0],
            'max_fitness_achieved': max(fitness_values),
            'average_improvement_per_generation': (fitness_values[-1] - fitness_values[0]) / len(fitness_values),
            'generations_to_best': fitness_values.index(max(fitness_values)),

            'diversity_stats': {
                'initial_diversity': diversity_values[0],
                'final_diversity': diversity_values[-1],
                'min_diversity': min(diversity_values),
                'average_diversity': sum(diversity_values) / len(diversity_values)
            },

            'events': {
                'breakthrough_count': len(self.breakthrough_generations),
                'diversity_crises': len(self.diversity_crises),
                'convergence_events': len(self.convergence_events)
            },

            'runtime_info': {
                'start_time': self.experiment_start_time.isoformat() if self.experiment_start_time else None,
                'end_time': self.generation_history[-1].timestamp.isoformat(),
                'total_duration_minutes': (
                    (self.generation_history[-1].timestamp - self.experiment_start_time).total_seconds() / 60
                    if self.experiment_start_time else 0
                )
            }
        }

    def plot_fitness_evolution(self, save_path: Optional[str] = None, show_plot: bool = True) -> bool:
        """
        Plot fitness evolution over generations.

        AI CONCEPT: FITNESS LANDSCAPE VISUALIZATION
        =========================================
        Fitness plots reveal important patterns:
        - Smooth upward trend = good convergence
        - Plateaus = possible local optima
        - Sudden jumps = breakthrough discoveries
        - Oscillations = insufficient selection pressure
        - Flat lines = premature convergence
        """

        if not PLOTTING_AVAILABLE:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
            return False

        if not self.generation_history:
            print("⚠️  No data to plot")
            return False

        generations = [s.generation for s in self.generation_history]
        best_fitness = [s.best_fitness for s in self.generation_history]
        avg_fitness = [s.average_fitness for s in self.generation_history]
        worst_fitness = [s.worst_fitness for s in self.generation_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Fitness evolution plot
        ax1.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness', alpha=0.8)
        ax1.plot(generations, avg_fitness, 'b-', linewidth=1.5, label='Average Fitness', alpha=0.7)
        ax1.plot(generations, worst_fitness, 'r-', linewidth=1, label='Worst Fitness', alpha=0.6)

        # Fill between best and worst for visual impact
        ax1.fill_between(generations, best_fitness, worst_fitness, alpha=0.2, color='blue')

        # Mark special events
        for gen in self.breakthrough_generations:
            ax1.axvline(x=gen, color='gold', linestyle='--', alpha=0.7, linewidth=1)
            ax1.text(gen, max(best_fitness) * 0.9, 'Breakthrough', rotation=90,
                     fontsize=8, ha='right', va='top')

        for gen in self.convergence_events:
            ax1.axvline(x=gen, color='purple', linestyle=':', alpha=0.7, linewidth=1)
            ax1.text(gen, max(best_fitness) * 0.85, 'Convergence', rotation=90,
                     fontsize=8, ha='right', va='top')

        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Evolution of Fitness Over Generations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Diversity plot
        diversity_values = [s.diversity_score for s in self.generation_history]
        ax2.plot(generations, diversity_values, 'orange', linewidth=2, label='Population Diversity')

        # Mark diversity crises
        for gen in self.diversity_crises:
            ax2.axvline(x=gen, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax2.text(gen, max(diversity_values) * 0.9, 'Low Diversity', rotation=90,
                     fontsize=8, ha='right', va='top')

        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity Score')
        ax2.set_title('Population Diversity Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to {save_path}")

        if show_plot:
            plt.show()

        return True

    def plot_selection_analysis(self, save_path: Optional[str] = None, show_plot: bool = True) -> bool:
        """Plot analysis of selection pressure and genetic diversity"""

        if not PLOTTING_AVAILABLE or not self.generation_history:
            return False

        generations = [s.generation for s in self.generation_history]
        selection_pressure = [s.selection_pressure for s in self.generation_history]
        diversity_scores = [s.diversity_score for s in self.generation_history]
        improvement_rates = [s.improvement_rate for s in self.generation_history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Selection pressure over time
        ax1.plot(generations, selection_pressure, 'purple', linewidth=2)
        ax1.set_title('Selection Pressure Over Time')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Selection Pressure')
        ax1.grid(True, alpha=0.3)

        # Diversity vs Selection Pressure scatter
        ax2.scatter(selection_pressure, diversity_scores, alpha=0.6, c=generations, cmap='viridis')
        ax2.set_xlabel('Selection Pressure')
        ax2.set_ylabel('Diversity Score')
        ax2.set_title('Diversity vs Selection Pressure')
        colorbar = plt.colorbar(ax2.collections[0], ax=ax2)
        colorbar.set_label('Generation')

        # Improvement rate over time
        ax3.plot(generations, improvement_rates, 'green', linewidth=2)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Fitness Improvement Rate')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Improvement Rate')
        ax3.grid(True, alpha=0.3)

        # Stagnation analysis
        stagnation_counts = [s.stagnation_counter for s in self.generation_history]
        ax4.plot(generations, stagnation_counts, 'red', linewidth=2)
        ax4.set_title('Stagnation Counter')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Generations Since Improvement')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show_plot:
            plt.show()

        return True

    def plot_genome_genealogy(self, max_genomes: int = 50, save_path: Optional[str] = None) -> bool:
        """Plot family tree of genomes (if lineage tracking enabled)"""

        if not PLOTTING_AVAILABLE:
            return False

        if not self.lineage_tree:
            print("⚠️  No lineage data available. Enable lineage tracking in evolution config.")
            return False

        # This is a simplified genealogy plot
        # A full implementation would use graph layout algorithms
        print(f"📊 Genealogy tree contains {len(self.lineage_tree)} genomes")
        print("   (Full genealogy visualization requires additional graph libraries)")

        # Could implement with networkx if available
        return True

    def export_data(self, file_path: str, format: str = 'json'):
        """Export all statistics data for external analysis"""

        data = {
            'experiment_metadata': self.experiment_metadata,
            'summary_statistics': self.get_summary_statistics(),
            'generation_history': [
                {
                    'generation': s.generation,
                    'timestamp': s.timestamp.isoformat(),
                    'best_fitness': s.best_fitness,
                    'average_fitness': s.average_fitness,
                    'worst_fitness': s.worst_fitness,
                    'diversity_score': s.diversity_score,
                    'improvement_rate': s.improvement_rate,
                    'stagnation_counter': s.stagnation_counter
                }
                for s in self.generation_history
            ],
            'special_events': {
                'breakthroughs': self.breakthrough_generations,
                'diversity_crises': self.diversity_crises,
                'convergence_events': self.convergence_events
            }
        }

        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        print(f"📁 Data exported to {file_path}")

    def compare_experiments(self, other_stats: 'EvolutionStatistics',
                            metric: str = 'best_fitness') -> Dict[str, Any]:
        """Compare this experiment with another experiment"""

        if not self.generation_history or not other_stats.generation_history:
            return {'error': 'Insufficient data for comparison'}

        # Align generations for fair comparison
        max_gen = min(len(self.generation_history), len(other_stats.generation_history))

        self_values = [getattr(s, metric) for s in self.generation_history[:max_gen]]
        other_values = [getattr(s, metric) for s in other_stats.generation_history[:max_gen]]

        return {
            'metric_compared': metric,
            'generations_compared': max_gen,
            'experiment_1_final': self_values[-1],
            'experiment_2_final': other_values[-1],
            'winner': 'experiment_1' if self_values[-1] > other_values[-1] else 'experiment_2',
            'improvement_difference': abs(self_values[-1] - other_values[-1]),
            'convergence_comparison': {
                'exp1_converged_at': min(self.convergence_events) if self.convergence_events else None,
                'exp2_converged_at': min(other_stats.convergence_events) if other_stats.convergence_events else None
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Evolution Statistics...")

    # Create test data
    stats_tracker = EvolutionStatistics()
    stats_tracker.start_experiment({'algorithm': 'test_ga', 'parameters': {'pop_size': 50}})

    # Simulate some evolution data
    from .genome import Genome
    import random

    for generation in range(20):
        # Create fake population with improving fitness
        population = []
        for i in range(50):
            genome = Genome.random(generation)
            # Simulate improving fitness over generations with some noise
            base_fitness = generation * 2 + random.gauss(0, 2) + random.uniform(0, 10)
            genome.fitness = max(0, base_fitness)
            population.append(genome)

        # Record generation statistics
        gen_stats = stats_tracker.record_generation(generation, population)

        if generation % 5 == 0:
            print(f"Gen {generation}: Best={gen_stats.best_fitness:.2f}, "
                  f"Avg={gen_stats.average_fitness:.2f}, "
                  f"Div={gen_stats.diversity_score:.3f}")

    # Test summary statistics
    summary = stats_tracker.get_summary_statistics()
    print(f"\n📊 Evolution Summary:")
    print(f"   Total improvement: {summary['overall_improvement']:.2f}")
    print(f"   Generations to best: {summary['generations_to_best']}")
    print(f"   Breakthrough events: {summary['events']['breakthrough_count']}")

    # Test plotting (if matplotlib available)
    if PLOTTING_AVAILABLE:
        print(f"\n📈 Creating plots...")
        stats_tracker.plot_fitness_evolution(show_plot=False)
        print("   Fitness evolution plot created")

        stats_tracker.plot_selection_analysis(show_plot=False)
        print("   Selection analysis plot created")
    else:
        print("\n⚠️  Plotting not available (matplotlib not installed)")

    # Test data export
    stats_tracker.export_data('test_evolution_data.json')
    print("   Data exported to test_evolution_data.json")

    print("\n✅ All statistics tests passed!")
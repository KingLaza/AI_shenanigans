"""
Genetic Algorithm Framework for Jump King AI

This package implements a complete genetic algorithm system designed to evolve
intelligent behavior for platformer games.

Core Philosophy:
- Modular design for easy experimentation
- Educational focus with extensive documentation
- Clean separation between algorithm and application
- Type hints and docstrings for clarity

Components:
- genome: Individual solution representation
- population: Collection management
- fitness: Evaluation strategies
- selection: Parent selection algorithms
- crossover: Recombination operators
- mutation: Variation operators
- evolution: Main evolutionary engine
- statistics: Performance tracking
"""

from .genome import Genome
from .population import Population
from .fitness import FitnessFunction
from .selection import SelectionStrategy
from .crossover import CrossoverOperator
from .mutation import MutationOperator
from .evolution import EvolutionEngine
from .statistics import EvolutionStatistics

__version__ = "1.0.0"
__author__ = "Your Name"

# Export main classes for easy importing
__all__ = [
    'Genome',
    'Population',
    'FitnessFunction',
    'SelectionStrategy',
    'CrossoverOperator',
    'MutationOperator',
    'EvolutionEngine',
    'EvolutionStatistics'
]

# Default configuration for quick start
DEFAULT_GA_CONFIG = {
    'population_size': 100,
    'generations': 50,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'elite_size': 5,
    'tournament_size': 3
}
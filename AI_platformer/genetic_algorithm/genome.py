"""
Genome: The Building Block of Genetic Algorithms

A genome represents one possible solution to our problem. In biological terms,
it's like the DNA of an organism - it contains all the information needed to
build and evaluate that organism.

KEY AI CONCEPT: PROBLEM REPRESENTATION
=======================================
The most critical decision in AI is HOW you represent your problem. The genome
is your representation choice, and it determines what solutions are possible.

For Jump King AI:
- Problem: Find a sequence of moves that gets the player as high as possible
- Representation: A list of Move objects (type + strength)
- Search Space: 5^20 * 20^20 = astronomically large!

Why this representation works:
- Finite and well-defined
- Can be easily modified (mutated and crossed over)
- Maps directly to game actions
- Captures the temporal nature of the problem
"""

from typing import List, Dict, Any, Tuple
import random
import copy
from dataclasses import dataclass


@dataclass
class Move:
    """
    Represents a single action the AI can take.

    This is your existing Move class, but with additional methods
    for genetic algorithm operations.

    Attributes:
        move_type (int): Action type (0-4)
            0: Jump up
            1: Jump left
            2: Jump right
            3: Walk left
            4: Walk right
        strength (int): Action intensity (1-20)
            For jumps: charge time/force
            For walks: duration
    """
    move_type: int
    strength: int

    def __post_init__(self):
        """Validate move parameters"""
        if not 0 <= self.move_type <= 4:
            raise ValueError(f"move_type must be 0-4, got {self.move_type}")
        if not 1 <= self.strength <= 20:
            raise ValueError(f"strength must be 1-20, got {self.strength}")

    @classmethod
    def random(cls) -> 'Move':
        """Create a random move - used for initial population"""
        return cls(
            move_type=random.randint(0, 4),
            strength=random.randint(1, 20)
        )

    def mutate(self, mutation_rate: float = 0.1) -> 'Move':
        """
        Create a slightly modified version of this move.

        AI CONCEPT: MUTATION
        ====================
        Mutation introduces random changes to prevent the algorithm from
        getting stuck. Without mutation, we'd only recombine existing
        solutions and never discover truly novel approaches.

        Args:
            mutation_rate: Probability of changing each component

        Returns:
            A new Move object with possible modifications
        """
        new_move_type = self.move_type
        new_strength = self.strength

        # Mutate move type with given probability
        if random.random() < mutation_rate:
            new_move_type = random.randint(0, 4)

        # Mutate strength with given probability
        if random.random() < mutation_rate:
            # Small random change around current value
            delta = random.randint(-3, 3)
            new_strength = max(1, min(20, self.strength + delta))

        return Move(new_move_type, new_strength)

    def distance_to(self, other: 'Move') -> float:
        """
        Calculate similarity between moves.
        Used for diversity measurements.
        """
        type_diff = abs(self.move_type - other.move_type)
        strength_diff = abs(self.strength - other.strength)
        return type_diff + (strength_diff / 20.0)  # Normalize strength


class Genome:
    """
    A complete solution to the Jump King problem.

    AI CONCEPT: GENOTYPE vs PHENOTYPE
    =================================
    - Genotype: The encoded representation (this class)
    - Phenotype: The actual behavior when executed (player performance)

    This separation lets us manipulate solutions mathematically while
    testing them in the real environment.

    Attributes:
        moves: Sequence of actions for the player to execute
        fitness: Cached performance score (None until evaluated)
        generation: Which generation this genome was born in
        parent_ids: Track lineage for analysis
    """

    def __init__(self, moves: List[Move] = None, generation: int = 0):
        """
        Initialize a genome.

        Args:
            moves: List of Move objects. If None, creates random sequence
            generation: Which evolutionary generation this belongs to
        """
        self.sequence_length = 20  # From your configs

        if moves is None:
            # Create random genome for initial population
            self.moves = [Move.random() for _ in range(self.sequence_length)]
        else:
            if len(moves) != self.sequence_length:
                raise ValueError(f"Expected {self.sequence_length} moves, got {len(moves)}")
            self.moves = copy.deepcopy(moves)

        # Performance tracking
        self.fitness: float = None
        self.generation = generation
        self.parent_ids: Tuple[str, str] = None  # For lineage tracking
        self.id = self._generate_id()

        # Detailed performance metrics
        self.performance_data: Dict[str, Any] = {}

    def _generate_id(self) -> str:
        """Generate unique identifier for this genome"""
        import uuid
        return str(uuid.uuid4())[:8]

    @classmethod
    def random(cls, generation: int = 0) -> 'Genome':
        """
        Create a completely random genome.

        This is how we initialize our population - with diverse,
        random solutions that evolution will improve over time.
        """
        return cls(generation=generation)

    def mutate(self, mutation_rate: float = 0.1) -> 'Genome':
        """
        Create a mutated copy of this genome.

        AI CONCEPT: EXPLORATION vs EXPLOITATION
        ======================================
        Mutation provides EXPLORATION - it helps us discover new areas
        of the solution space. Without it, we'd only exploit known good
        solutions and might miss better ones.

        Args:
            mutation_rate: Probability of mutating each move

        Returns:
            New genome with some moves potentially modified
        """
        mutated_moves = [
            move.mutate(mutation_rate) for move in self.moves
        ]

        child = Genome(mutated_moves, self.generation + 1)
        child.parent_ids = (self.id, None)  # Asexual reproduction
        return child

    def crossover(self, other: 'Genome') -> Tuple['Genome', 'Genome']:
        """
        Combine this genome with another to create offspring.

        AI CONCEPT: SEXUAL REPRODUCTION & RECOMBINATION
        =============================================
        Crossover combines successful traits from two parents. This is
        EXPLOITATION - we're building on what we know works.

        Multiple crossover strategies:
        1. Single-point: Cut at one position, swap tails
        2. Two-point: Cut at two positions, swap middle section
        3. Uniform: Randomly choose each gene from either parent

        Args:
            other: The second parent genome

        Returns:
            Tuple of two offspring genomes
        """
        if len(self.moves) != len(other.moves):
            raise ValueError("Cannot cross genomes of different lengths")

        # Two-point crossover (good for preserving building blocks)
        length = len(self.moves)
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)

        # Create offspring by swapping middle sections
        child1_moves = (
                self.moves[:point1] +
                other.moves[point1:point2] +
                self.moves[point2:]
        )

        child2_moves = (
                other.moves[:point1] +
                self.moves[point1:point2] +
                other.moves[point2:]
        )

        # Create the offspring
        child1 = Genome(child1_moves, max(self.generation, other.generation) + 1)
        child2 = Genome(child2_moves, max(self.generation, other.generation) + 1)

        # Track parentage
        child1.parent_ids = (self.id, other.id)
        child2.parent_ids = (self.id, other.id)

        return child1, child2

    def diversity_score(self, population: List['Genome']) -> float:
        """
        Calculate how different this genome is from the population.

        AI CONCEPT: POPULATION DIVERSITY
        ==============================
        Diversity prevents premature convergence. If everyone's too similar,
        evolution stops improving. This metric helps maintain variety.
        """
        if not population:
            return 1.0

        total_distance = 0
        for other in population:
            if other.id != self.id:
                move_distances = [
                    move1.distance_to(move2)
                    for move1, move2 in zip(self.moves, other.moves)
                ]
                total_distance += sum(move_distances) / len(move_distances)

        return total_distance / len(population) if population else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'fitness': self.fitness,
            'moves': [
                {'move_type': move.move_type, 'strength': move.strength}
                for move in self.moves
            ],
            'performance_data': self.performance_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Create genome from dictionary"""
        moves = [
            Move(move_data['move_type'], move_data['strength'])
            for move_data in data['moves']
        ]

        genome = cls(moves, data['generation'])
        genome.id = data['id']
        genome.parent_ids = data.get('parent_ids')
        genome.fitness = data.get('fitness')
        genome.performance_data = data.get('performance_data', {})

        return genome

    def __str__(self) -> str:
        """Human-readable representation"""
        fitness_str = f"{self.fitness:.2f}" if self.fitness else "Not evaluated"
        return f"Genome({self.id}, Gen {self.generation}, Fitness: {fitness_str})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Check equality based on move sequences"""
        if not isinstance(other, Genome):
            return False
        return self.moves == other.moves

    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dicts"""
        return hash(self.id)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Genome class...")

    # Create random genomes
    genome1 = Genome.random(generation=0)
    genome2 = Genome.random(generation=0)

    print(f"Genome 1: {genome1}")
    print(f"Genome 2: {genome2}")

    # Test crossover
    child1, child2 = genome1.crossover(genome2)
    print(f"Child 1: {child1}")
    print(f"Child 2: {child2}")

    # Test mutation
    mutated = genome1.mutate(mutation_rate=0.3)
    print(f"Mutated: {mutated}")

    print("All tests passed!")
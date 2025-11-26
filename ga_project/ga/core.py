# ga/core.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random
import math

@dataclass
class GAParameters:
    """Параметры генетического алгоритма (по UML: GAParameters)."""
    populationSize: int = 50
    crossoverProb: float = 0.8
    mutationProb: float = 0.1
    maxGenerations: int = 100
    randomSeed: Optional[int] = None
    otherParams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Individual:
    """Особь: вектор генов + значение функции (fitness)."""
    genes: List[float]
    fitness: Optional[float] = None  # fitness = значение целевой функции (минимизируем!)


@dataclass
class GenerationStat:
    """Статистика по поколению (GenerationStat из UML)."""
    generationIndex: int
    bestFitness: float
    meanFitness: float


@dataclass
class GAResult:
    """Результат работы ГА (GAResult из UML)."""
    bestIndividual: Individual
    history: List[GenerationStat]


class Population:
    """Популяция (Population из UML)."""

    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @classmethod
    def initialize(cls, problem: "OptimizationProblem", params: GAParameters) -> "Population":
        """Инициализация случайной популяции в пределах допустимых границ задачи."""
        individuals: List[Individual] = []
        for _ in range(params.populationSize):
            genes = [
                random.uniform(lb, ub) for (lb, ub) in problem.bounds
            ]
            individuals.append(Individual(genes=genes))
        return cls(individuals)

    def getBest(self) -> Individual:
        """Лучшая особь (минимальное значение fitness)."""
        return min(self.individuals, key=lambda ind: ind.fitness)


# Чтобы избежать циклических импортов
class OptimizationProblem:
    def evaluate(self, ind: Individual) -> float:
        raise NotImplementedError

    @property
    def bounds(self):
        raise NotImplementedError


def _clone_individual(ind: Individual, reset_fitness: bool = True) -> Individual:
    """Глубокое копирование особи."""
    return Individual(
        genes=list(ind.genes),
        fitness=None if reset_fitness else ind.fitness
    )


class GeneticAlgorithm:
    """
    Реализация GeneticAlgorithm из UML:
    - selectionOperator : SelectionOperator
    - crossoverOperator : CrossoverOperator
    - mutationOperator  : MutationOperator
    - problem           : OptimizationProblem
    - params            : GAParameters
    """

    def __init__(self, problem: OptimizationProblem,
                 selectionOperator: "SelectionOperator",
                 crossoverOperator: "CrossoverOperator",
                 mutationOperator: "MutationOperator",
                 params: GAParameters):
        self.problem = problem
        self.selectionOperator = selectionOperator
        self.crossoverOperator = crossoverOperator
        self.mutationOperator = mutationOperator
        self.params = params

        if self.params.randomSeed is not None:
            random.seed(self.params.randomSeed)

    def _evaluate_population(self, population: Population) -> None:
        """Вычисление значения целевой функции (fitness) для всей популяции."""
        for ind in population.individuals:
            if ind.fitness is None:
                ind.fitness = self.problem.evaluate(ind)

    def run(self) -> GAResult:
        """Запуск ГА, возврат GAResult (как в UML)."""
        population = Population.initialize(self.problem, self.params)
        history: List[GenerationStat] = []

        # начальная оценка
        self._evaluate_population(population)
        best = population.getBest()
        mean_fitness = sum(ind.fitness for ind in population.individuals) / len(population.individuals)
        history.append(GenerationStat(0, bestFitness=best.fitness, meanFitness=mean_fitness))

        for gen in range(1, self.params.maxGenerations + 1):
            # выбор родителей
            parents = self.selectionOperator.select(population, self.params.populationSize)

            # формирование потомков
            children: List[Individual] = []
            i = 0
            while len(children) < self.params.populationSize:
                p1 = parents[i % len(parents)]
                p2 = parents[(i + 1) % len(parents)]
                i += 2

                if random.random() < self.params.crossoverProb:
                    offspring = self.crossoverOperator.crossover(p1, p2)
                else:
                    offspring = [_clone_individual(p1), _clone_individual(p2)]

                for child in offspring:
                    self.mutationOperator.mutate(child, self.params, self.problem)
                    children.append(child)
                    if len(children) >= self.params.populationSize:
                        break

            population = Population(children)
            self._evaluate_population(population)

            best = population.getBest()
            mean_fitness = sum(ind.fitness for ind in population.individuals) / len(population.individuals)
            history.append(GenerationStat(gen, bestFitness=best.fitness, meanFitness=mean_fitness))

        # Лучший из последней популяции (без элитизма, но достаточно для курсовой)
        best = population.getBest()
        return GAResult(bestIndividual=best, history=history)

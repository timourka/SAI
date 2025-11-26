# ga/problems.py
from __future__ import annotations
from typing import List, Tuple
from .core import Individual, OptimizationProblem
import math


class SphereProblem(OptimizationProblem):
    """
    f(x) = sum(x_i^2)
    Классическая тестовая задача, минимум в (0, ..., 0).
    """

    def __init__(self, dimension: int = 10):
        print("f(x) = sum(x_i^2)")
        self.dimension = dimension
        self._bounds: List[Tuple[float, float]] = [(-5.12, 5.12)] * dimension

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return self._bounds

    def evaluate(self, ind: Individual) -> float:
        return sum(x * x for x in ind.genes)


class RastriginProblem(OptimizationProblem):
    """
    f(x) = 10n + sum(x_i^2 - 10*cos(2πx_i))
    Многомодальная задача с большим количеством локальных минимумов.
    """

    def __init__(self, dimension: int = 10):
        print("f(x) = 10n + sum(x_i^2 - 10*cos(2πx_i))")
        self.dimension = dimension
        self._bounds: List[Tuple[float, float]] = [(-5.12, 5.12)] * dimension

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return self._bounds

    def evaluate(self, ind: Individual) -> float:
        n = len(ind.genes)
        return 10 * n + sum(
            x * x - 10 * math.cos(2 * math.pi * x)
            for x in ind.genes
        )


class AnotherTestProblem(OptimizationProblem):
    """
    Пример ещё одной задачи (Розенброк).
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    """

    def __init__(self, dimension: int = 2):
        print("f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)")
        self.dimension = dimension
        self._bounds: List[Tuple[float, float]] = [(-2.0, 2.0)] * dimension

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return self._bounds

    def evaluate(self, ind: Individual) -> float:
        s = 0.0
        for i in range(len(ind.genes) - 1):
            x = ind.genes[i]
            y = ind.genes[i + 1]
            s += 100.0 * (y - x * x) ** 2 + (1.0 - x) ** 2
        return s

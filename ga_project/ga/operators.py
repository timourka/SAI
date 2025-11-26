# ga/operators.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import random
import math

from .core import Individual, Population, GAParameters
from .core import _clone_individual
from .core import OptimizationProblem


# ====== SelectionOperator ======

class SelectionOperator(ABC):
    """Интерфейс SelectionOperator (UML)."""

    @abstractmethod
    def select(self, population: Population, k: int) -> List[Individual]:
        """
        Вернуть k выбранных особей (с копированием генов).
        Чем меньше fitness, тем лучше.
        """


class TournamentSelection(SelectionOperator):
    """Турнирная селекция."""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: Population, k: int) -> List[Individual]:
        selected: List[Individual] = []
        individuals = population.individuals
        n = len(individuals)

        for _ in range(k):
            contestants = random.sample(individuals, self.tournament_size)
            winner = min(contestants, key=lambda ind: ind.fitness)
            selected.append(_clone_individual(winner, reset_fitness=False))
        return selected


class RouletteSelection(SelectionOperator):
    """Рулетка: вероятность ∝ 1 / (1 + fitness)."""

    def select(self, population: Population, k: int) -> List[Individual]:
        selected: List[Individual] = []
        inds = population.individuals

        # для задач с fitness >= 0
        scores = [1.0 / (1.0 + ind.fitness) for ind in inds]
        total = sum(scores)
        if total == 0:
            # fallback: равномерная выборка
            for _ in range(k):
                selected.append(_clone_individual(random.choice(inds), reset_fitness=False))
            return selected

        for _ in range(k):
            r = random.random() * total
            acc = 0.0
            for ind, sc in zip(inds, scores):
                acc += sc
                if acc >= r:
                    selected.append(_clone_individual(ind, reset_fitness=False))
                    break

        return selected


# ====== CrossoverOperator ======

class CrossoverOperator(ABC):
    """Интерфейс CrossoverOperator (UML)."""

    @abstractmethod
    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        """Вернёт список потомков (обычно 2)."""


class OnePointCrossover(CrossoverOperator):
    """Одноточечный кроссинговер."""

    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        n = len(p1.genes)
        if n < 2:
            return [_clone_individual(p1), _clone_individual(p2)]
        point = random.randint(1, n - 1)
        c1_genes = p1.genes[:point] + p2.genes[point:]
        c2_genes = p2.genes[:point] + p1.genes[point:]
        return [Individual(c1_genes), Individual(c2_genes)]


class TwoPointCrossover(CrossoverOperator):
    """Двухточечный кроссинговер."""

    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        n = len(p1.genes)
        if n < 3:
            # fallback к одноточечному
            return OnePointCrossover().crossover(p1, p2)
        p1_idx = random.randint(1, n - 2)
        p2_idx = random.randint(p1_idx + 1, n - 1)
        c1_genes = (
            p1.genes[:p1_idx] +
            p2.genes[p1_idx:p2_idx] +
            p1.genes[p2_idx:]
        )
        c2_genes = (
            p2.genes[:p1_idx] +
            p1.genes[p1_idx:p2_idx] +
            p2.genes[p2_idx:]
        )
        return [Individual(c1_genes), Individual(c2_genes)]


class UniformCrossover(CrossoverOperator):
    """Равномерный кроссинговер (каждый ген выбирается случайно от одного из родителей)."""

    def __init__(self, swap_prob: float = 0.5):
        self.swap_prob = swap_prob

    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        n = len(p1.genes)
        c1_genes = []
        c2_genes = []
        for i in range(n):
            if random.random() < self.swap_prob:
                c1_genes.append(p2.genes[i])
                c2_genes.append(p1.genes[i])
            else:
                c1_genes.append(p1.genes[i])
                c2_genes.append(p2.genes[i])
        return [Individual(c1_genes), Individual(c2_genes)]


class NewCrossoverOperatorHeuristic(CrossoverOperator):
    """
    NewCrossoverOperatorHeuristic - это мягкий направленный оператор кроссинговера.
    Он определяет лучшего родителя (по fitness) и формирует двух потомков как взвешенные комбинации родителей:
    - первый потомок смещён в сторону лучшего родителя,
    - второй потомок расположен между центром родителей и лучшим,
    добавляется небольшой адаптивный шум для сохранения разнообразия.
    """

    def __init__(self, bias=0.6, noise_scale=0.02):
        self.bias = bias            # 0.5 = стандартный BLX, 0.6–0.7 = улучшенный
        self.noise_scale = noise_scale

    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        # определяем лучшего родителя
        if p1.fitness is None or p2.fitness is None:
            best, worst = p1, p2
        else:
            best, worst = (p1, p2) if p1.fitness <= p2.fitness else (p2, p1)

        c1_genes = []
        c2_genes = []

        for gb, gw in zip(best.genes, worst.genes):

            # нормальная смесь (как BLX)
            low = min(gb, gw)
            high = max(gb, gw)
            d = high - low

            # offspring 1 — ближе к лучшему
            g1 = self.bias * gb + (1 - self.bias) * gw

            # offspring 2 — ближе к центру
            g_mid = (gb + gw) / 2
            g2 = self.bias * g_mid + (1 - self.bias) * gb

            # адаптивный шум (очень маленький!)
            noise = random.gauss(0, self.noise_scale * d)

            c1_genes.append(g1 + noise)
            c2_genes.append(g2 + noise)

        return [Individual(c1_genes), Individual(c2_genes)]

class NewCrossoverOperatorUniform(CrossoverOperator):
    """
    Adaptive Uniform Crossover
    - Каждый ген выбирается как у Uniform crossover.
    - Но: если один из родителей лучше, то вероятность взять его ген чуть выше.
    """

    def __init__(self, bias_strength=0.1):
        # насколько оператор предпочитает лучшего родителя
        # 0.0 => обычный uniform
        # 0.1 => лёгкий перекос к лучшему (оптимально)
        self.bias_strength = bias_strength

    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        # определяем лучшего родителя
        if p1.fitness <= p2.fitness:
            best = p1
            worst = p2
        else:
            best = p2
            worst = p1

        # базовая вероятность взять ген от каждого родителя
        base_p = 0.5

        # небольшое смещение в пользу лучшего
        prob_best = base_p + self.bias_strength

        c1_genes = []
        c2_genes = []

        for gb, gw in zip(best.genes, worst.genes):
            # ген потомка 1
            if random.random() < prob_best:
                c1_genes.append(gb)
            else:
                c1_genes.append(gw)

            # ген потомка 2 (можно сделать зеркальным)
            if random.random() < prob_best:
                c2_genes.append(gb)
            else:
                c2_genes.append(gw)

        return [Individual(c1_genes), Individual(c2_genes)]
    
class NewCrossoverOperatorShuffle(CrossoverOperator):
    """
    Shuffle Crossover:
    - объединяет гены обоих родителей
    - случайно перемешивает их
    - делит поровну на двух потомков
    """

    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        # объединяем гены родителей
        combined = p1.genes + p2.genes

        # перемешиваем случайно
        random.shuffle(combined)

        # делим список на две части
        half = len(p1.genes)

        c1_genes = combined[:half]
        c2_genes = combined[half:half*2]

        return [Individual(c1_genes), Individual(c2_genes)]


class NewCrossoverOperatorAdaptiveShuffle(CrossoverOperator):
    """
    Adaptive Shuffle Crossover (ASC):
    - если родители похожи -> мягкое смешивание (uniform)
    - если сильно различаются -> частичное случайное перемешивание
    - если один родитель намного лучше -> смещение в его сторону
    """

    def __init__(self, shuffle_strength=0.3, bias_strength=0.15):
        self.shuffle_strength = shuffle_strength
        self.bias_strength = bias_strength

    def crossover(self, p1: Individual, p2: Individual) -> List[Individual]:
        # определяем лучшего родителя
        if p1.fitness <= p2.fitness:
            best, worst = p1, p2
        else:
            best, worst = p2, p1

        g1 = []
        g2 = []

        for gb, gw in zip(best.genes, worst.genes):
            diff = abs(gb - gw)

            # 1) вероятность шафлинга растёт с различием родителей
            shuffle_prob = min(1.0, diff * self.shuffle_strength)

            if random.random() < shuffle_prob:
                # полный random pick
                if random.random() < 0.5:
                    g1.append(gw)
                    g2.append(gb)
                else:
                    g1.append(gb)
                    g2.append(gw)
            else:
                # мягкое смешивание как в улучшенном uniform
                prob_best = 0.5 + self.bias_strength
                if random.random() < prob_best:
                    g1.append(gb)
                else:
                    g1.append(gw)

                if random.random() < prob_best:
                    g2.append(gb)
                else:
                    g2.append(gw)

        return [Individual(g1), Individual(g2)]

# ====== MutationOperator ======

class MutationOperator(ABC):
    """Интерфейс MutationOperator (UML)."""

    @abstractmethod
    def mutate(self, individual: Individual,
               params: GAParameters,
               problem: OptimizationProblem) -> None:
        """
        Мутация особи in-place.
        Для удобства добавлены params и problem (границы).
        """


class SimpleMutation(MutationOperator):
    """
    Простая гауссовская мутация: для каждого гена с вероятностью mutationProb
    добавляем случайное смещение и обрезаем по границам задачи.
    """

    def __init__(self, sigma_fraction: float = 0.1):
        self.sigma_fraction = sigma_fraction

    def mutate(self, individual: Individual,
               params: GAParameters,
               problem: OptimizationProblem) -> None:
        for i, (lb, ub) in enumerate(problem.bounds):
            if random.random() < params.mutationProb:
                width = ub - lb
                sigma = self.sigma_fraction * width
                new_val = individual.genes[i] + random.gauss(0.0, sigma)
                # обрезаем по границам
                new_val = max(lb, min(ub, new_val))
                individual.genes[i] = new_val
        # после мутации значение fitness нужно пересчитать
        individual.fitness = None

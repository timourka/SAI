# ga/experiment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable
import statistics as stats
import itertools

from .core import GAParameters, GAResult
from .core import GeneticAlgorithm
from .operators import SelectionOperator, CrossoverOperator, MutationOperator
from .problems import OptimizationProblem


@dataclass
class ExperimentConfig:
    """
    По UML:
    + problem       : OptimizationProblem
    + gaParams      : GAParameters
    + crossoverOp   : CrossoverOperator
    + selectionOp   : SelectionOperator
    + mutationOp    : MutationOperator
    + runsCount     : int
    """
    problem: OptimizationProblem
    gaParams: GAParameters
    crossoverOp: CrossoverOperator
    selectionOp: SelectionOperator
    mutationOp: MutationOperator
    runsCount: int = 10


@dataclass
class ExperimentStats:
    """
    По UML:
    + results          : List<GAResult>
    + meanBestFitness  : float
    + stdBestFitness   : float
    + minBestFitness   : float
    + maxBestFitness   : float
    """
    results: List[GAResult]
    meanBestFitness: float
    stdBestFitness: float
    minBestFitness: float
    maxBestFitness: float

    @classmethod
    def from_results(cls, results: List[GAResult]) -> "ExperimentStats":
        bests = [res.bestIndividual.fitness for res in results]
        mean_val = stats.mean(bests)
        std_val = stats.pstdev(bests) if len(bests) > 1 else 0.0
        return cls(
            results=results,
            meanBestFitness=mean_val,
            stdBestFitness=std_val,
            minBestFitness=min(bests),
            maxBestFitness=max(bests),
        )


@dataclass
class HyperparameterSpace:
    """
    По UML:
    + parameterGrid : Map
    + getConfigs(baseConfig : ExperimentConfig) : List<ExperimentConfig>
    parameterGrid:
        ключи: имена параметров ('populationSize', 'crossoverProb', ...)
        значения: итерируемые наборы значений
    """
    parameterGrid: Dict[str, Iterable[Any]]

    def getConfigs(self, baseConfig: ExperimentConfig) -> List[ExperimentConfig]:
        keys = list(self.parameterGrid.keys())
        values_list = [list(self.parameterGrid[k]) for k in keys]
        configs: List[ExperimentConfig] = []

        for combo in itertools.product(*values_list):
            params_copy = GAParameters(
                populationSize=baseConfig.gaParams.populationSize,
                crossoverProb=baseConfig.gaParams.crossoverProb,
                mutationProb=baseConfig.gaParams.mutationProb,
                maxGenerations=baseConfig.gaParams.maxGenerations,
                randomSeed=baseConfig.gaParams.randomSeed,
                otherParams=dict(baseConfig.gaParams.otherParams),
            )

            for key, value in zip(keys, combo):
                if hasattr(params_copy, key):
                    setattr(params_copy, key, value)
                else:
                    params_copy.otherParams[key] = value

            cfg = ExperimentConfig(
                problem=baseConfig.problem,
                gaParams=params_copy,
                crossoverOp=baseConfig.crossoverOp,
                selectionOp=baseConfig.selectionOp,
                mutationOp=baseConfig.mutationOp,
                runsCount=baseConfig.runsCount,
            )
            configs.append(cfg)

        return configs


class ExperimentRunner:
    """
    По UML:
    + runSingle(config : ExperimentConfig) : GAResult
    + runSeries(config : ExperimentConfig) : ExperimentStats
    + runHyperparameterSearch(space : HyperparameterSpace) : List<ExperimentStats>
    """

    def runSingle(self, config: ExperimentConfig) -> GAResult:
        ga = GeneticAlgorithm(
            problem=config.problem,
            selectionOperator=config.selectionOp,
            crossoverOperator=config.crossoverOp,
            mutationOperator=config.mutationOp,
            params=config.gaParams,
        )
        return ga.run()

    def runSeries(self, config: ExperimentConfig) -> ExperimentStats:
        results: List[GAResult] = []
        for i in range(config.runsCount):
            # чтобы не повторяться, можно менять seed
            if config.gaParams.randomSeed is not None:
                config.gaParams.randomSeed += 1
            res = self.runSingle(config)
            results.append(res)
        return ExperimentStats.from_results(results)

    def runHyperparameterSearch(self, space: HyperparameterSpace,
                                baseConfig: ExperimentConfig) -> List[ExperimentStats]:
        configs = space.getConfigs(baseConfig)
        stats_list: List[ExperimentStats] = []
        for cfg in configs:
            stats_list.append(self.runSeries(cfg))
        return stats_list

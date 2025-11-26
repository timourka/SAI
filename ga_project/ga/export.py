# ga/export.py
from __future__ import annotations

from typing import Iterable
import csv
from pathlib import Path

from .core import GAResult, GenerationStat
from .experiment import ExperimentStats


class ResultExporter:
    """
    По UML:
    + exportCSV(stats : ExperimentStats, filePath : String) : void
    + exportHistory(result : GAResult, filePath : String) : void
    """

    def exportCSV(self, stats: ExperimentStats, filePath: str) -> None:
        path = Path(filePath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["run_index", "best_fitness"])
            for i, res in enumerate(stats.results):
                writer.writerow([i, res.bestIndividual.fitness])

    def exportHistory(self, result: GAResult, filePath: str) -> None:
        path = Path(filePath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["generation", "best_fitness", "mean_fitness"])
            for gen_stat in result.history:
                writer.writerow([
                    gen_stat.generationIndex,
                    gen_stat.bestFitness,
                    gen_stat.meanFitness,
                ])

# main.py
from ga import (
    GAParameters,
    OptimizationProblem,
    SphereProblem,
    RastriginProblem,
    AnotherTestProblem,
    TournamentSelection,
    RouletteSelection,
    SimpleMutation,
    OnePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    NewCrossoverOperatorHeuristic,
    NewCrossoverOperatorUniform,
    NewCrossoverOperatorShuffle,
    NewCrossoverOperatorAdaptiveShuffle,
    NewCrossoverOperatorSmartShuffle,
    NewCrossoverOperatorShuffleSelective,
    NewCrossoverOperatorGoodGenePreservingShuffle,
    NewCrossoverOperatorRandomFixedShuffle,
    ExperimentConfig,
    ExperimentRunner,
    ResultExporter,
)

def run_problem(problem : OptimizationProblem, problem_name : str):
    # 2. Задаём гиперпараметры ГА
    params = GAParameters(
        populationSize=100,
        crossoverProb=0.5,
        mutationProb=0.1,
        maxGenerations=100,
        randomSeed=42,
    )

    # 3. Фиксируем селекцию и мутацию
    selection = TournamentSelection()
    mutation = SimpleMutation(sigma_fraction=0.1)

    # 4. Набор операторов кроссинговера: стандартные + новый
    crossovers = {
        "one_point": OnePointCrossover(),
        "two_point": TwoPointCrossover(),
        "uniform": UniformCrossover(),
        "newHeuristic": NewCrossoverOperatorHeuristic(),
        "newUniform": NewCrossoverOperatorUniform(),
        "newShuffle": NewCrossoverOperatorShuffle(),
        "newAdaptiveShuffle": NewCrossoverOperatorAdaptiveShuffle(),
        "newSmartShuffle": NewCrossoverOperatorSmartShuffle(),
        "newShuffleSelective": NewCrossoverOperatorShuffleSelective(),
        "newGoodGenePreservingShuffle": NewCrossoverOperatorGoodGenePreservingShuffle(problem=problem),
        "newRandomFixedShuffle": NewCrossoverOperatorRandomFixedShuffle(),
    }

    runner = ExperimentRunner()
    exporter = ResultExporter()

    for name, crossover in crossovers.items():
        print(f"\n--- Crossover: {name} ---")

        config = ExperimentConfig(
            problem=problem,
            gaParams=params,
            crossoverOp=crossover,
            selectionOp=selection,
            mutationOp=mutation,
            runsCount=10,  # сколько раз повторяем эксперимент
        )

        stats = runner.runSeries(config)

        print(f"mean best fitness = {stats.meanBestFitness:.6f}")
        print(f"std  best fitness = {stats.stdBestFitness:.6f}")
        print(f"min  best fitness = {stats.minBestFitness:.6f}")
        print(f"max  best fitness = {stats.maxBestFitness:.6f}")

        # при желании можно сохранять результаты в CSV
        exporter.exportCSV(stats, f"results/{problem_name}/best_{name}.csv")

    print("\nСравнение завершено. CSV-файлы лежат в папке 'results'.")
    
def run_comparison():
    run_problem(SphereProblem(dimension=10), "SphereProblem")
    run_problem(RastriginProblem(dimension=10), "RastriginProblem")
    run_problem(AnotherTestProblem(dimension=10), "AnotherTestProblem")

if __name__ == "__main__":
    run_comparison()
    
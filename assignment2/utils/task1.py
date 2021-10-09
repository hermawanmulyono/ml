import copy
import json
import logging
import multiprocessing
import os
import time
from typing import List, Callable, Tuple, Dict, Any, NamedTuple

import numpy as np

from utils.grid import serialize_grid_table, OptimizationResults, \
    MultipleResults, \
    grid_args_generator, parse_grid_table, GridTable, summarize_grid_table, \
    serialize_grid_optimization_summary, GridOptimizationSummary, \
    parse_grid_optimization_summary
from utils.outputs import optimization_grid_table, optimization_grid_summary, \
    optimization_parameter_plot

import mlrose_hiive as mlrose

from utils.plots import parameter_plot

REPEATS = 24


def simulated_annealing_wrapper(problem,
                                init_temp=1.0,
                                decay=0.99,
                                min_temp=0.001,
                                max_attempts=10,
                                max_iters=np.inf,
                                init_state=None,
                                curve=False,
                                fevals=False,
                                random_state=None,
                                state_fitness_callback=None,
                                callback_user_info=None):
    """A wrapper for mlrose.simulated_annealing function

    The arguments are exactly the same, except the schedule
    has been decomposed into GeomDecay's `init_temp`,
    `decay`, and `min_temp`.

    """
    schedule = mlrose.GeomDecay(init_temp, decay, min_temp)
    return mlrose.simulated_annealing(problem, schedule, max_attempts,
                                      max_iters, init_state, curve, fevals,
                                      random_state, state_fitness_callback,
                                      callback_user_info)


def run_single(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable, kwargs: dict):
    kwargs = copy.deepcopy(kwargs)
    kwargs['curve'] = True
    start = time.time()
    best_state, best_fitness, fitness_curve = alg_fn(problem_fit, **kwargs)
    end = time.time()
    duration = end - start

    # The second column of fitness_curve is the function_evaluations.
    # Just take the very last entry. We're not interested in knowing the
    # function evaluations in every iteration.
    function_evaluations = int(fitness_curve[-1, -1])

    optimization_results = OptimizationResults(best_state, best_fitness,
                                               function_evaluations, duration)

    return optimization_results


def run_multiple(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable,
                 kwargs: dict, repeats: int) -> MultipleResults:

    def args_generator():
        for _ in range(repeats):
            yield problem_fit, alg_fn, kwargs

    n_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_cpus) as pool:
        multiple_results: List[OptimizationResults] = pool.starmap(
            run_single, args_generator())

    return multiple_results


def grid_run(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable,
             param_grid: dict, repeats: int):

    table: GridTable = []
    for kwargs in grid_args_generator(param_grid):
        logging.info(f'Running {kwargs}')
        multiple_results = run_multiple(problem_fit, alg_fn, kwargs, repeats)
        table.append((kwargs, multiple_results))

    return table


def onemax_task():
    fitness = mlrose.OneMax()
    vector_lengths = [20, 100, 500]
    problems = [
        mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True)
        for length in vector_lengths
    ]
    problem_names = [f'onemax_{length}' for length in vector_lengths]

    return _task1_template(problems, problem_names)


def fourpeaks_task():
    fitness = mlrose.FourPeaks()
    vector_lengths = [20, 100, 500]
    problems = [
        mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True)
        for length in vector_lengths
    ]
    problem_names = [f'fourpeaks_{length}' for length in vector_lengths]

    return _task1_template(problems, problem_names)


def maxkcolor_edges(num_vertices: int, num_edges: int):
    """Generates edges for MaxKColor problems randomly

    Args:
        num_vertices: Number of vertices
        num_edges: Number of edges

    Returns:
        List of randomly generated undirected edges
        `[..., (v1, v2) ,...]`. Note that (v1, v2) and
        (v2, v1) are equivalent.

    """
    if num_edges <= 1 or num_vertices <= 1:
        raise ValueError

    if num_edges > num_vertices * (num_vertices - 1) / 2:
        raise ValueError

    edges = set()
    while len(edges) < num_edges:
        v1 = np.random.randint(0, num_vertices)
        v2 = v1
        while v2 == v1:
            v2 = np.random.randint(0, num_vertices)

        if ((v1, v2) not in edges) and ((v2, v1) not in edges):
            edges.add((v1, v2))

    return list(edges)


def max_kcolor_task():
    vert_edge_pairs = [(30, 20), (60, 40), (90, 60)]

    problems = []
    problem_names = []
    for num_vertices, num_edges in vert_edge_pairs:
        fitness = mlrose.MaxKColor(maxkcolor_edges(num_vertices, num_edges))
        problem_fit = mlrose.DiscreteOpt(length=num_vertices,
                                         fitness_fn=fitness,
                                         maximize=True)
        problems.append(problem_fit)
        problem_names.append(f'maxkcolor_v{num_vertices}_e{num_edges}')

    return _task1_template(problems, problem_names)


def _task1_template(problems: List[mlrose.DiscreteOpt],
                    problem_names: List[str]):

    for problem, problem_name in zip(problems, problem_names):
        # Each problem is solved multiple times with different algorithms
        # and different parameters
        algs_params_tuples = _make_alg_params_tuple(problem)

        # See the AlgorithmExperimentSetup definition about these variables
        alg: Callable
        params_grid: Dict[str, Any]
        alg_plots: List[Tuple[str, str]]

        for alg, params_grid, alg_plots in algs_params_tuples:
            alg_name = alg.__name__
            json_path = optimization_grid_table(f'{problem_name}', alg_name)

            # Only run grid-search if the JSON file doesn't exist
            if not os.path.exists(json_path):
                logging.info(f'Running {problem_name} {alg_name}')
                grid_table = grid_run(problem,
                                      alg,
                                      params_grid,
                                      repeats=REPEATS)
                grid_table_serialized = serialize_grid_table(grid_table)

                # Write results to disk
                with open(json_path, 'w') as j:
                    json.dump(grid_table_serialized, j, indent=2)

            with open(json_path, 'r') as j:
                grid_table_serialized = json.load(j)

            grid_table = parse_grid_table(grid_table_serialized)
            grid_summary = summarize_grid_table(grid_table, 'optimization')

            grid_summary_json_path = optimization_grid_summary(
                problem_name, alg_name)

            # Only summarize results when the JSON file doesn't exist
            if not os.path.exists(grid_summary_json_path):
                with open(grid_summary_json_path, 'w') as j:
                    grid_summary_serialized = \
                        serialize_grid_optimization_summary(grid_summary)
                    json.dump(grid_summary_serialized, j, indent=2)

            # Check by opening the serialized results
            with open(grid_summary_json_path) as j:
                grid_summary_serialized = json.load(j)

            grid_summary: GridOptimizationSummary = \
                parse_grid_optimization_summary(grid_summary_serialized)

            sync_optimization_plots(grid_summary, alg_plots, problem_name,
                                    alg_name)


def sync_optimization_plots(grid_summary: GridOptimizationSummary,
                            alg_plots: List[Tuple[str, str]], problem_name: str,
                            alg_name: str):
    for y_axis in ['best_fitness', 'duration', 'function_evaluations']:
        for param_name, scale in alg_plots:
            figure_path = optimization_parameter_plot(problem_name, alg_name,
                                                      param_name, y_axis)

            # Only generate plot if it doesn't exist
            if os.path.exists(figure_path):
                continue

            fig = parameter_plot(grid_summary,
                                 param_name,
                                 scale,
                                 y_axis=y_axis)

            fig.write_image(figure_path)


class AlgorithmExperimentSetup(NamedTuple):
    """Algorithm experiment setup

    The members are:
      - `alg_function`: A function to solve an optimization
        problem e.g. mlrose.hill_climb, etc.
      - `param_grid`: Parameter grid in Scikit-Learn style
      - `plots`: Which parameters to plot, and whether to
        use linear/log scale `[..., (param_name, s) ,...]`
        where `s` is either `linear` or `logarithmic`.

    """
    alg_function: Callable
    param_grid: Dict[str, Any]
    plots: List[Tuple[str, str]]


def _make_alg_params_tuple(
        problem: mlrose.DiscreteOpt) -> List[AlgorithmExperimentSetup]:
    """Helper to create list of AlgorithmExperimentSetup objects

    Args:
        problem: A discrete optimization problem. This must
            have been initialized with the proper fitness
            function and all the necessary parameters.

    Returns:
        List of AlgorithmExperimentSetup objects
        corresponding to hill climb, simulated annealing,
        genetic algorithm, and MIMIC.

    """
    algs_params_tuples = []

    # Hill climbing
    hill_climb_param_grid = {'restarts': [0, 10, 50]}
    hill_climb_plots = [('restarts', 'linear')]
    algs_params_tuples.append(
        AlgorithmExperimentSetup(mlrose.hill_climb, hill_climb_param_grid,
                                 hill_climb_plots))

    # Simulated annealing
    vector_length = problem.length
    sa_param_grid = {
        'max_attempts': [10, vector_length],
        'init_temp': [1.0, 10.0, 100.0],
        'decay': [0.99, 0.999, 0.9999]
    }
    sa_plots = [('init_temp', 'logarithmic'), ('decay', 'logarithmic')]

    algs_params_tuples.append(
        AlgorithmExperimentSetup(simulated_annealing_wrapper, sa_param_grid,
                                 sa_plots))

    # Genetic algorithm
    ga_param_grid = {
        'pop_size': [200],
        'mutation_prob': np.logspace(-1, -5, 10),
        'max_attempts': [10, vector_length]
    }
    ga_plots = [('mutation_prob', 'logarithmic')]
    algs_params_tuples.append(
        AlgorithmExperimentSetup(mlrose.genetic_alg, ga_param_grid, ga_plots))

    # MIMIC
    mimic_param_grid = {
        'keep_pct': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'max_attempts': [10, vector_length]
    }
    mimic_plots = [('keep_pct', 'linear')]
    algs_params_tuples.append(
        AlgorithmExperimentSetup(mlrose.mimic, mimic_param_grid, mimic_plots))

    return algs_params_tuples


def task1():
    """A task to compare some optimization problems
    """
    onemax_task()
    fourpeaks_task()
    max_kcolor_task()

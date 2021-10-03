import copy
import json
import logging
import multiprocessing
import os
import time
from typing import List, Callable, Tuple, Dict, Any

import numpy as np
import six
import sys

from utils.grid import serialize_grid_table, OptimizationResults, \
    MultipleResults, \
    grid_args_generator, parse_grid_table, GridTable
from utils.outputs import grid_results_json

sys.modules['sklearn.externals.six'] = six
import mlrose


def run_single(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable, kwargs: dict):
    kwargs = copy.deepcopy(kwargs)
    kwargs['curve'] = True
    start = time.time()
    best_state, best_fitness, fitness_curve = alg_fn(problem_fit, **kwargs)
    end = time.time()
    duration = end - start

    optimization_results = OptimizationResults(best_state, best_fitness,
                                               fitness_curve, duration)

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
        algs_params_tuples = _make_alg_params_tuple(problem)

        for alg, params_grid in algs_params_tuples:
            alg_name = alg.__name__
            json_path = grid_results_json(f'{problem_name}', alg_name)

            if not os.path.exists(json_path):
                logging.info(f'Running {problem_name} {alg_name}')
                grid_results = grid_run(problem, alg, params_grid, repeats=20)

                # Write results to disk
                with open(json_path, 'w') as j:
                    serialized = serialize_grid_table(grid_results)
                    json.dump(serialized, j, indent=4)

            with open(json_path, 'r') as j:
                json_grid_results = json.load(j)

            grid_results = parse_grid_table(json_grid_results)


def _make_alg_params_tuple(
        problem: mlrose.DiscreteOpt) -> List[Tuple[Callable, Dict[str, Any]]]:
    """Helper to create list of alg, param_grid tuples

    The returned list of tuples can be used for grid-search
    purposes.

    Args:
        problem: A discrete optimization problem. This must
            have been initialized with the proper fitness
            function and all the necessary parameters.

    Returns:
        List of tuples [..., (alg_function, param_grid),...]
        corresponding to hill climb, simulated annealing,
        genetic algorithm, and MIMIC.

    """
    algs_params_tuples = []

    # Hill climbing
    hill_climb_param_grid = {'restarts': [0, 10, 50]}
    algs_params_tuples.append((mlrose.hill_climb, hill_climb_param_grid))

    # Simulated annealing
    vector_length = problem.length
    sa_param_grid = {'max_attempts': [10, vector_length]}
    algs_params_tuples.append((mlrose.simulated_annealing, sa_param_grid))

    # Genetic algorithm
    ga_param_grid = {
        'pop_size': [200],
        'mutation_prob': np.logspace(-1, -5, 10),
        'max_attempts': [10, vector_length]
    }
    algs_params_tuples.append((mlrose.genetic_alg, ga_param_grid))

    return algs_params_tuples


def task1():
    """A task to compare some optimization problems
    """
    onemax_task()
    fourpeaks_task()
    max_kcolor_task()

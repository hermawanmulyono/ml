import copy
import json
import logging
import multiprocessing
import time
from collections import Callable
from typing import NamedTuple, List, Tuple

import numpy as np
import six
import sys

from utils.outputs import grid_results_json

sys.modules['sklearn.externals.six'] = six
import mlrose


class SingleResults(NamedTuple):
    best_state: np.ndarray
    best_fitness: float
    fitness_curve: np.ndarray
    duration: float


def serialize_single_results(single_results: SingleResults):
    s = {
        'best_state': single_results.best_state.astype(int).tolist(),
        'best_fitness': single_results.best_fitness,
        'fitness_curve': single_results.fitness_curve.astype(float).tolist(),
        'duration': single_results.duration
    }
    return s


# For containing repeated experiment results
MultipleResults = List[SingleResults]


def serialize_multiple_results(multiple_results: MultipleResults):
    return [serialize_single_results(s) for s in multiple_results]


# List of [..., (kwargs, multiple_results) ,...]
GridResults = List[Tuple[dict, MultipleResults]]


def serialize_grid_results(grid_results: GridResults):
    serialized = [
        (kwargs, serialize_multiple_results(m)) for kwargs, m in grid_results
    ]
    return serialized


def run_single(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable, kwargs: dict):
    kwargs = copy.deepcopy(kwargs)
    kwargs['curve'] = True
    start = time.time()
    best_state, best_fitness, fitness_curve = alg_fn(problem_fit, **kwargs)
    end = time.time()
    duration = end - start

    single_results = SingleResults(best_state, best_fitness, fitness_curve,
                                   duration)

    return single_results


def _increase_grid_index(grid_index_: List[int], param_grid: dict):
    param_names: List[str] = list(param_grid.keys())

    grid_index_ = copy.deepcopy(grid_index_)

    for i in range(len(grid_index_)):
        grid_index_[i] += 1

        if grid_index_[i] >= len(param_grid[param_names[i]]):
            grid_index_[i] = 0
        else:
            break

    return grid_index_


def run_multiple(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable,
                 kwargs: dict, repeats: int) -> MultipleResults:

    def args_generator():
        for _ in range(repeats):
            yield problem_fit, alg_fn, kwargs

    n_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_cpus) as pool:
        multiple_results: List[SingleResults] = pool.starmap(
            run_single, args_generator())

    return multiple_results


def grid_run(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable,
             param_grid: dict, repeats: int):
    param_names: List[str] = list(param_grid.keys())

    def args_generator():
        grid_index = [0] * len(param_names)

        while True:
            kwargs = copy.deepcopy(param_grid)
            update = {
                key: param_grid[key][gi]
                for gi, key in zip(grid_index, param_names)
            }
            kwargs.update(update)

            yield kwargs

            grid_index = _increase_grid_index(grid_index, param_grid)
            if all([gi == 0 for gi in grid_index]):
                break

    table: List[Tuple[dict, MultipleResults]] = []
    for kwargs in args_generator():
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

        algs_params_tuples = []
        hill_climb_params_grid = {'restarts': [0, 10, 50]}
        algs_params_tuples.append((mlrose.hill_climb, hill_climb_params_grid))

        vector_length = problem.length
        sa_params_grid = {'max_attempts': [10, vector_length]}
        algs_params_tuples.append((mlrose.simulated_annealing, sa_params_grid))

        ga_params_grid = {
            'pop_size': [200],
            'mutation_prob': np.logspace(-1, -5, 10),
            'max_attempts': [10, vector_length]
        }
        algs_params_tuples.append((mlrose.genetic_alg, ga_params_grid))

        mimic_params_grid = {'max_attempts': [10, vector_length]}
        algs_params_tuples.append((mlrose.mimic, mimic_params_grid))

        for alg, params_grid in algs_params_tuples:
            alg_name = alg.__name__
            logging.info(f'Running {problem_name} {alg_name}')
            grid_results = grid_run(problem, alg, params_grid, repeats=20)
            json_path = grid_results_json(f'{problem_name}', alg_name)
            with open(json_path, 'w') as j:
                serialized = serialize_grid_results(grid_results)
                json.dump(serialized, j, indent=4)


def task1():
    """A task to compare some optimization problems
    """
    onemax_task()
    fourpeaks_task()
    max_kcolor_task()

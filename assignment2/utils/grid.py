"""
Grid search functions
"""

import copy
from typing import NamedTuple, List, Tuple, Dict, Any

import numpy as np


class OptimizationResults(NamedTuple):
    """Random Optimization Results

    """
    best_state: np.ndarray
    best_fitness: float
    fitness_curve: np.ndarray
    duration: float


class NNResults(NamedTuple):
    train_accuracy: float
    val_accuracy: float
    fit_time: float


class Stats(NamedTuple):
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    min: float
    max: float


class MultipleResultsSummary(NamedTuple):
    best_fitness: Stats
    duration: Stats


# JSON-serializable single results
JSONSingleResults = Dict[str, Any]

# For containing repeated experiment results
MultipleResults = List[OptimizationResults]

# JSON-serializable multiple results
JSONMultipleResults = List[JSONSingleResults]

# List of [..., (kwargs, multiple_results) ,...]
GridResults = List[Tuple[dict, MultipleResults]]

# JSON-serializable grid-results
JSONGridResults = List[Tuple[dict, JSONMultipleResults]]


def serialize_single_results(
        single_results: OptimizationResults) -> JSONSingleResults:
    s = {
        'best_state': single_results.best_state.astype(int).tolist(),
        'best_fitness': single_results.best_fitness,
        'fitness_curve': single_results.fitness_curve.astype(float).tolist(),
        'duration': single_results.duration
    }
    return s


def parse_single_results(d: JSONSingleResults) -> OptimizationResults:
    """Parses the given dictionary to SingleResults object

    Args:
        d: A JSON-compatible dictionary

    Returns:
        A SingleResults object

    """

    if set(d.keys()) != {
            'best_state', 'best_fitness', 'fitness_curve', 'duration'
    }:
        raise ValueError

    best_state = np.array(d['best_state'])
    best_fitness = d['best_fitness']
    fitness_curve = np.ndarray(d['fitness_curve'])
    duration = d['duration']

    return OptimizationResults(best_state, best_fitness, fitness_curve, duration)


def serialize_multiple_results(
        multiple_results: MultipleResults) -> JSONMultipleResults:
    return [serialize_single_results(s) for s in multiple_results]


def parse_multiple_results(input_list: JSONMultipleResults) -> MultipleResults:
    return [parse_single_results(d) for d in input_list]


def serialize_grid_results(grid_results: GridResults) -> JSONGridResults:
    serialized = [
        (kwargs, serialize_multiple_results(m)) for kwargs, m in grid_results
    ]
    return serialized


def parse_grid_results(json_grid_results: JSONGridResults) -> GridResults:
    return [
        (kwargs, parse_multiple_results(m)) for kwargs, m in json_grid_results
    ]


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


def grid_args_generator(param_grid: Dict[str, Any]):
    """Generates arguments for a grid search

    Args:
        param_grid: A dictionary with the following
            structure

            {'param1': [..., value1, ...],
             'param2': [..., value2, ...],
             ...
             'paramN': [..., valueN, ...]}

            Note that this is similar to Scikit-Learn
            structure.

    Returns:
        A generator which yields the arguments for grid
        search.

    """
    param_names: List[str] = list(param_grid.keys())

    grid_index = [0] * len(param_names)

    while True:
        kwargs = copy.deepcopy(param_grid)
        update = {
            key: param_grid[key][gi] for gi, key in zip(grid_index, param_names)
        }
        kwargs.update(update)

        yield kwargs

        grid_index = _increase_grid_index(grid_index, param_grid)
        if all([gi == 0 for gi in grid_index]):
            break


def array_stats(x: np.ndarray) -> Stats:
    """Compute statistics of the given array

    Args:
        x: 1-dimensional array

    Returns:
        A Stats object of the given array

    """
    mean = float(np.mean(x))
    min_, q1, median, q3, max_ = np.percentile(x, [0, 25, 50, 75, 100])
    std = float(np.std(x))
    return Stats(mean, float(median), std, float(q1), float(q3), float(min_),
                 float(max_))


def summarize_multiple_results(multiple_results: MultipleResults):

    best_fitness = np.array([m.best_fitness for m in multiple_results])
    best_fitness_stats = array_stats(best_fitness)

    duration = np.array([m.duration for m in multiple_results])
    duration_stats = array_stats(duration)

    return MultipleResultsSummary(best_fitness_stats, duration_stats)


def summarize_grid_results(grid_results: GridResults):
    return [
        (kwargs, summarize_multiple_results(m)) for kwargs, m in grid_results
    ]

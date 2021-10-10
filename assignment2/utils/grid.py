"""
Grid search functions
"""

import copy
from typing import NamedTuple, List, Tuple, Dict, Any, Union

import numpy as np


class OptimizationResults(NamedTuple):
    """Random Optimization Results"""
    best_state: np.ndarray
    best_fitness: float
    function_evaluations: int
    duration: float


class NNResults(NamedTuple):
    train_accuracy: float
    val_accuracy: float
    fit_time: float


class Stats(NamedTuple):
    """Statistics of a scalar

    This is usually used for a repeated experiment.

    The members are:
      - mean
      - median
      - std
      - q1
      - q3
      - min
      - max

    """
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    min: float
    max: float


class OptimizationSummary(NamedTuple):
    """Optimization summary

    The members are:
      - `best_fitness`: The best_fitness statistics
      - `duration`: Duration statistics
      - `function_evaluations`: Function evaluations
        statistics

    """
    best_fitness: Stats
    duration: Stats
    function_evaluations: Stats


class NNSummary(NamedTuple):
    train_accuracy: Stats
    val_accuracy: Stats
    fit_time: Stats


# JSON-serializable single results
JSONSingleResults = Dict[str, Any]

# For containing repeated experiment results
MultipleResults = List[Union[OptimizationResults, NNResults]]

# JSON-serializable multiple results
JSONMultipleResults = List[JSONSingleResults]

# List of [..., (kwargs, multiple_results) ,...]
GridTable = List[Tuple[dict, MultipleResults]]

# JSON-serializable grid-results
JSONGridTable = List[Tuple[dict, JSONMultipleResults]]


class GridOptimizationSummary(NamedTuple):
    """Grid optimization summary

    The members are:
      - `best_fitness`: The best model fitness statistics
      - `duration`: The best model duration statistics
      - `function_evaluations`: The best model number of
        function evaluations statistics
      - `kwargs`: The best model kwargs
      - `table`: The grid table
        `[..., (kwargs, OptimizationSummary), ...]`

    """
    best_fitness: Stats
    duration: Stats
    function_evaluations: Stats
    kwargs: dict
    table: List[Tuple[dict, OptimizationSummary]]


class GridNNSummary(NamedTuple):
    train_accuracy: Stats
    val_accuracy: Stats
    fit_time: Stats
    kwargs: dict
    table: List[Tuple[dict, NNSummary]]


GridSummary = Union[GridOptimizationSummary, GridNNSummary]

ParamGrid = Dict[str, Union[list, np.ndarray]]


def _serialize_single_results(
        single_results: Union[OptimizationResults,
                              NNResults]) -> JSONSingleResults:

    if type(single_results) == OptimizationResults:
        s = {
            'best_state': single_results.best_state.astype(int).tolist(),
            'best_fitness': single_results.best_fitness,
            'function_evaluations': single_results.function_evaluations,
            'duration': single_results.duration
        }
        return s
    elif type(single_results) == NNResults:
        s = {
            'train_accuracy': single_results.train_accuracy,
            'val_accuracy': single_results.val_accuracy,
            'fit_time': single_results.fit_time
        }
        return s
    else:
        raise NotImplementedError


def _parse_single_results(
        d: JSONSingleResults) -> Union[OptimizationResults, NNResults]:
    """Parses the given dictionary to SingleResults object

    Args:
        d: A JSON-compatible dictionary

    Returns:
        An OptimizationResults or NNResults object, depending on the
            input `d`.

    """

    if set(d.keys()) == {
            'best_state', 'best_fitness', 'function_evaluations', 'duration'
    }:
        best_state = np.array(d['best_state'])
        best_fitness = d['best_fitness']
        function_evaluations = d['function_evaluations']
        duration = d['duration']

        return OptimizationResults(best_state, best_fitness,
                                   function_evaluations, duration)
    elif set(d.keys()) == {'train_accuracy', 'val_accuracy', 'fit_time'}:
        train_accuracy = d['train_accuracy']
        val_accuracy = d['val_accuracy']
        fit_time = d['fit_time']

        return NNResults(train_accuracy, val_accuracy, fit_time)
    else:
        raise NotImplementedError


def _serialize_multiple_results(
        multiple_results: MultipleResults) -> JSONMultipleResults:
    return [_serialize_single_results(s) for s in multiple_results]


def _parse_multiple_results(input_list: JSONMultipleResults) -> MultipleResults:
    return [_parse_single_results(d) for d in input_list]


def serialize_grid_table(grid_table: GridTable) -> JSONGridTable:
    serialized = [
        (kwargs, _serialize_multiple_results(m)) for kwargs, m in grid_table
    ]
    return serialized


def parse_grid_table(json_grid_table: JSONGridTable) -> GridTable:
    return [
        (kwargs, _parse_multiple_results(m)) for kwargs, m in json_grid_table
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


def grid_args_generator(param_grid: ParamGrid):
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


def _array_stats(x: np.ndarray) -> Stats:
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


def _summarize_multiple_results(multiple_results: MultipleResults,
                                problem: str):
    if problem == 'optimization':
        best_fitness = np.array([m.best_fitness for m in multiple_results])
        best_fitness_stats = _array_stats(best_fitness)

        duration = np.array([m.duration for m in multiple_results])
        duration_stats = _array_stats(duration)

        function_evaluations = np.array(
            [m.function_evaluations for m in multiple_results])
        function_evaluations_stats = _array_stats(function_evaluations)

        return OptimizationSummary(best_fitness_stats, duration_stats,
                                   function_evaluations_stats)

    elif problem == 'nn':
        train_accuracy = np.array([m.train_accuracy for m in multiple_results])
        train_accuracy_stats = _array_stats(train_accuracy)

        val_accuracy = np.array([m.val_accuracy for m in multiple_results])
        val_accuracy_stats = _array_stats(val_accuracy)

        fit_time = np.array([m.fit_time for m in multiple_results])
        fit_time_stats = _array_stats(fit_time)

        return NNSummary(train_accuracy_stats, val_accuracy_stats,
                         fit_time_stats)

    else:
        raise NotImplementedError


def summarize_grid_table(grid_table: GridTable, problem: str):
    """

    Args:
        grid_table:
        problem: Either 'optimization' or 'nn'

    Returns:

    """
    summary_table = [(kwargs, _summarize_multiple_results(m, problem))
                     for kwargs, m in grid_table]

    if problem == 'nn':
        metric_vals = [x.val_accuracy.mean for _, x in summary_table]
    elif problem == 'optimization':
        metric_vals = [x.best_fitness.mean for _, x in summary_table]
    else:
        raise NotImplementedError

    argmax = np.argmax(metric_vals)

    kwargs, summary = summary_table[argmax]

    if problem == 'nn':
        summary: NNSummary
        return GridNNSummary(summary.train_accuracy, summary.val_accuracy,
                             summary.fit_time, kwargs, summary_table)

    elif problem == 'optimization':
        summary: OptimizationSummary
        return GridOptimizationSummary(summary.best_fitness, summary.duration,
                                       summary.function_evaluations, kwargs,
                                       summary_table)

    else:
        raise NotImplementedError


def _serialize_stats(stats: Stats):
    d = dict(stats._asdict())
    return d


def _parse_stats(d: Dict[str, Any]):
    return Stats(**d)


def _serialize_summary(summary: Union[NNSummary, OptimizationSummary]):
    if type(summary) == NNSummary:
        summary: NNSummary
        return {
            'train_accuracy': _serialize_stats(summary.train_accuracy),
            'val_accuracy': _serialize_stats(summary.val_accuracy),
            'fit_time': _serialize_stats(summary.fit_time)
        }
    elif type(summary) == OptimizationSummary:
        summary: OptimizationSummary
        return {
            'best_fitness':
                _serialize_stats(summary.best_fitness),
            'duration':
                _serialize_stats(summary.duration),
            'function_evaluations':
                _serialize_stats(summary.function_evaluations)
        }
    else:
        raise NotImplementedError


def _parse_summary(d: Dict[str, Any]):
    keys = set(d.keys())

    if keys == {'train_accuracy', 'val_accuracy', 'fit_time'}:
        train_accuracy = _parse_stats(d['train_accuracy'])
        val_accuracy = _parse_stats(d['val_accuracy'])
        fit_time = _parse_stats(d['fit_time'])

        return NNSummary(train_accuracy, val_accuracy, fit_time)

    elif keys == {'best_fitness', 'duration', 'function_evaluations'}:
        best_fitness = _parse_stats(d['best_fitness'])
        duration = _parse_stats(d['duration'])
        function_evaluations = _parse_stats(d['function_evaluations'])

        return OptimizationSummary(best_fitness, duration, function_evaluations)

    else:
        raise NotImplementedError


def _serialize_summary_table(table: List[tuple]):

    table: Tuple[dict, Union[NNSummary, OptimizationSummary]]

    table = copy.deepcopy(table)
    serialized = [
        (kwargs, _serialize_summary(summary)) for kwargs, summary in table
    ]
    return serialized


def _parse_summary_table(table: List[tuple]):
    table: Tuple[dict, dict]

    table = copy.deepcopy(table)
    parsed = [(kwargs, _parse_summary(summary)) for kwargs, summary in table]

    return parsed


def serialize_grid_nn_summary(grid_nn_summary: GridNNSummary):
    train_accuracy = _serialize_stats(grid_nn_summary.train_accuracy)
    val_accuracy = _serialize_stats(grid_nn_summary.val_accuracy)
    fit_time = _serialize_stats(grid_nn_summary.fit_time)
    kwargs = grid_nn_summary.kwargs
    table = _serialize_summary_table(grid_nn_summary.table)

    return {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'fit_time': fit_time,
        'kwargs': kwargs,
        'table': table
    }


def parse_grid_nn_summary(d: Dict[str, Any]):
    train_accuracy = _parse_stats(d['train_accuracy'])
    val_accuracy = _parse_stats(d['val_accuracy'])
    fit_time = _parse_stats(d['fit_time'])
    kwargs = d['kwargs']
    table = _parse_summary_table(d['table'])
    return GridNNSummary(train_accuracy, val_accuracy, fit_time, kwargs, table)


def serialize_grid_optimization_summary(
        grid_opt_summary: GridOptimizationSummary):
    best_fitness = _serialize_stats(grid_opt_summary.best_fitness)
    duration = _serialize_stats(grid_opt_summary.duration)
    function_evaluations = _serialize_stats(
        grid_opt_summary.function_evaluations)
    kwargs = grid_opt_summary.kwargs
    table = _serialize_summary_table(grid_opt_summary.table)

    return {
        'best_fitness': best_fitness,
        'duration': duration,
        'function_evaluations': function_evaluations,
        'kwargs': kwargs,
        'table': table
    }


def parse_grid_optimization_summary(d: Dict[str, Any]):
    best_fitness = _parse_stats(d['best_fitness'])
    duration = _parse_stats(d['duration'])
    function_evaluations = _parse_stats(d['function_evaluations'])
    kwargs = d['kwargs']
    table = _parse_summary_table(d['table'])
    return GridOptimizationSummary(best_fitness, duration, function_evaluations,
                                   kwargs, table)

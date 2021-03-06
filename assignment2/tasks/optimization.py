import copy
import json
import logging
import multiprocessing
import os
import time
from typing import List, Callable, Tuple, Dict, Any, NamedTuple

import numpy as np

from tasks.base import ExperimentBase
from utils.data import temp_seed
from utils.grid import OptimizationResults, \
    MultipleResults, \
    grid_args_generator, GridTable, summarize_grid_table, \
    serialize_grid_optimization_summary, GridOptimizationSummary, \
    parse_grid_optimization_summary, GridSummary, OptimizationSummary, Stats
from utils.outputs import optimization_grid_table, optimization_grid_summary, \
    optimization_parameter_plot, optimization_fitness_vs_iteration_plot, \
    optimization_alg_vs_problem_size

import mlrose_hiive as mlrose

from utils.plots import parameter_plot, alg_vs_problem_size_plot

REPEATS = 12


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


class OptimizationExperiment(ExperimentBase):

    def __init__(self, problem: mlrose.DiscreteOpt, problem_name: str,
                 algorithm_setup: AlgorithmExperimentSetup, repeats: int,
                 maximize: bool):
        """Initializes an OptimizationExperiment object

        Args:
            problem: The problem definition
            problem_name: Problem unique name, which may
                include some additional information
                e.g. problem size.
            algorithm_setup: An AlgorithmExperimentSetup
                object.
            repeats: How many times each set of
                hyperparameters is run
            maximize: If True, it is a maximization problem.
                If False, it is minimization.

        """
        self.problem = problem
        self.problem_name = problem_name

        self.alg = algorithm_setup.alg_function
        self.param_grid = algorithm_setup.param_grid
        self.alg_plots = algorithm_setup.plots

        self.repeats = repeats
        self._maximize = maximize

    @property
    def alg_name(self):
        return self.alg.__name__

    @property
    def grid_table_json(self) -> str:
        return optimization_grid_table(self.problem_name, self.alg_name)

    def grid_run(self) -> GridTable:
        logging.info(f'Running {self.problem_name} {self.alg_name}')
        grid_table = grid_run(self.problem, self.alg, self.param_grid,
                              self.repeats)

        return grid_table

    def summarize_grid_table(self, grid_table: GridTable):
        return summarize_grid_table(grid_table, 'optimization')

    @property
    def grid_summary_json_path(self):
        return optimization_grid_summary(self.problem_name, self.alg_name)

    def serialize_grid_summary(self,
                               grid_summary: GridSummary) -> Dict[str, Any]:
        return serialize_grid_optimization_summary(grid_summary)

    def parse_grid_summary(
            self, grid_summary_serialized: Dict[str, Any]) -> GridSummary:
        return parse_grid_optimization_summary(grid_summary_serialized)

    def sync_parameter_plots(self, grid_summary: GridSummary):
        negate_y_axis = not self._maximize
        sync_optimization_parameter_plots(grid_summary, self.alg_plots,
                                          self.problem_name, self.alg_name,
                                          negate_y_axis)

    @property
    def plot_hyperparameters(self) -> List[Tuple[str, str]]:
        return self.alg_plots

    @property
    def plot_metrics(self) -> List[str]:
        return [
            'best_fitness', 'duration', 'function_evaluations', 'iterations'
        ]

    def hyperparameter_plot_path(self, param_name: str, metric: str):
        return optimization_parameter_plot(self.problem_name, self.alg_name,
                                           param_name, metric)

    def generate_fitness_curve(self, best_kwargs: Dict[str, Any]) -> np.ndarray:
        kwargs = copy.deepcopy(best_kwargs)
        kwargs['curve'] = True
        best_state, best_fitness, fitness_curves = self.alg(
            self.problem, **kwargs)

        # Fitness value is the first column.
        # The second one is the function evaluations
        fitness_curve = fitness_curves[:, 0]

        if not self._maximize:
            fitness_curve = -1 * fitness_curve

        return fitness_curve

    @property
    def fitness_curve_name(self) -> str:
        return optimization_fitness_vs_iteration_plot(self.problem_name,
                                                      self.alg_name)


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
    iterations = len(fitness_curve)

    optimization_results = OptimizationResults(best_state, best_fitness,
                                               function_evaluations, duration,
                                               iterations)

    return optimization_results


def run_multiple(problem_fit: mlrose.DiscreteOpt, alg_fn: Callable,
                 kwargs: dict, repeats: int) -> MultipleResults:

    def args_generator():
        start = time.time()

        for _ in range(repeats):
            yield problem_fit, alg_fn, kwargs

            if time.time() - start > 600:
                # Break after 10 minutes
                break

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
    vector_lengths = [20, 60, 100, 140]
    problems = [
        mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True)
        for length in vector_lengths
    ]
    problem_names = [f'onemax_{length}' for length in vector_lengths]

    return _task1_template(problems, problem_names)


def fourpeaks_task():
    fitness = mlrose.FourPeaks(t_pct=0.4)
    vector_lengths = [20, 40, 60, 80]
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
        raise ValueError('Numbers of edges and vertices must be at least 1')

    if num_edges > num_vertices * (num_vertices - 1) / 2:
        raise ValueError('number of edges must be less than the maximum'
                         'V (V-1) / 2')

    edges = set()
    with temp_seed(1234):
        while len(edges) < num_edges:
            v1 = np.random.randint(0, num_vertices)
            v2 = v1
            while v2 == v1:
                v2 = np.random.randint(0, num_vertices)

            if ((v1, v2) not in edges) and ((v2, v1) not in edges):
                edges.add((v1, v2))

    return list(edges)


def max_kcolor_task():
    vert_edge_k_tuples = [(10, 40, 2), (20, 160, 3), (40, 500, 4)]

    problems = []
    problem_names = []
    for num_vertices, num_edges, k in vert_edge_k_tuples:
        fitness = mlrose.MaxKColor(maxkcolor_edges(num_vertices, num_edges))
        problem_fit = mlrose.DiscreteOpt(length=num_vertices,
                                         fitness_fn=fitness,
                                         maximize=False,
                                         max_val=k)
        problems.append(problem_fit)
        problem_names.append(f'maxkcolor_v{num_vertices}_e{num_edges}_k{k}')

    return _task1_template(problems, problem_names)


ProblemSizePlotsData = Dict[str, List[Tuple[str, OptimizationSummary]]]


def _task1_template(problems: List[mlrose.DiscreteOpt],
                    problem_names: List[str]):

    # problem_size_plots_data[alg_name] = [..., summary ,...]
    problem_size_plots_data: ProblemSizePlotsData = {}

    for problem, problem_name in zip(problems, problem_names):
        # Each problem is solved multiple times with different algorithms
        # and different parameters
        algs_params_tuples = _make_alg_params_tuple(problem)

        # See the AlgorithmExperimentSetup definition about these variables
        alg: Callable
        params_grid: Dict[str, Any]
        alg_plots: List[Tuple[str, str]]

        # The maximization flag is read from the problem object
        fitness_sign = problem.maximize
        maximization = (problem.maximize > 0)

        for alg_setup in algs_params_tuples:
            experiment = OptimizationExperiment(problem, problem_name,
                                                alg_setup, REPEATS,
                                                maximization)
            experiment.run()

            # Collect data for problem_size_plots_data
            grid_summary_json_path = experiment.grid_summary_json_path
            with open(grid_summary_json_path) as j:
                grid_summary_serialized = json.load(j)

            grid_summary: GridOptimizationSummary = \
                parse_grid_optimization_summary(grid_summary_serialized)

            if maximization:
                signed_fitness = grid_summary.best_fitness
            else:
                f = grid_summary.best_fitness
                signed_fitness = Stats(-f.mean, -f.median, f.std, -f.q3, -f.q1,
                                       -f.max, -f.min)

            best_model_summary = OptimizationSummary(
                signed_fitness, grid_summary.duration,
                grid_summary.function_evaluations, grid_summary.iterations)

            alg_name = experiment.alg_name
            if alg_name not in problem_size_plots_data:
                problem_size_plots_data[alg_name] = []

            problem_size_plots_data[alg_name].append(
                (problem_name, best_model_summary))

    sync_alg_vs_problem_size_plots(problem_size_plots_data)


def sync_alg_vs_problem_size_plots(
        problem_size_plots_data: ProblemSizePlotsData):
    for alg_name, plot_data in problem_size_plots_data.items():
        problem_names: List[str]
        summaries: List[OptimizationSummary]
        problem_names, summaries = zip(*plot_data)

        general_problem_name = _infer_general_problem_name(problem_names)

        for metric in OptimizationSummary._fields:

            figure_path = optimization_alg_vs_problem_size(
                general_problem_name, alg_name, metric)

            if os.path.exists(figure_path):
                continue

            fig = alg_vs_problem_size_plot(problem_names, summaries, metric)
            fig.write_image(figure_path)


def _infer_general_problem_name(problem_names: List[str]):
    """Infers general problem name from the specific ones

    For example, `[onemax_20, onemax_40, onemax_60]` will
    return `onemax`.

    Args:
        problem_names: A list of problem names with format
            `general_more_info`.

    Returns:
        The general problem name

    """

    if not problem_names:
        raise ValueError('problem_names must not be empty')

    general_name = problem_names[0].split('_')[0]

    if any([not n.startswith(general_name) for n in problem_names]):
        raise ValueError('All problem names must start with the same general '
                         'name.')

    return general_name


def sync_optimization_parameter_plots(grid_summary: GridOptimizationSummary,
                                      alg_plots: List[Tuple[str, str]],
                                      problem_name: str, alg_name: str,
                                      negate: bool):
    for y_axis in OptimizationSummary._fields:
        for param_name, scale in alg_plots:
            figure_path = optimization_parameter_plot(problem_name, alg_name,
                                                      param_name, y_axis)

            # Only generate plot if it doesn't exist
            if os.path.exists(figure_path):
                continue

            fig = parameter_plot(grid_summary,
                                 param_name,
                                 scale,
                                 y_axis=y_axis,
                                 negate_y_axis=negate)

            fig.write_image(figure_path)


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
    vector_length = problem.length

    # Hill climbing
    hill_climb_param_grid = {'restarts': [0, 10, 50, 250]}
    hill_climb_plots = [('restarts', 'linear')]
    algs_params_tuples.append(
        AlgorithmExperimentSetup(mlrose.hill_climb, hill_climb_param_grid,
                                 hill_climb_plots))

    # Simulated annealing
    sa_param_grid = {
        'max_attempts': [vector_length],
        'init_temp': [1.0, 10.0, 100.0],
        'decay': [0.99, 0.999, 0.9999]
    }
    sa_plots = [('init_temp', 'logarithmic'), ('decay', 'logarithmic')]

    algs_params_tuples.append(
        AlgorithmExperimentSetup(simulated_annealing_wrapper, sa_param_grid,
                                 sa_plots))

    # Genetic algorithm
    ga_param_grid = {
        'pop_size': [200, 800, 1400, 2000],
        'mutation_prob': 0.5 * np.logspace(0, -3, 5),
        'max_attempts': [vector_length]
    }
    ga_plots = [('mutation_prob', 'logarithmic'), ('pop_size', 'linear')]
    algs_params_tuples.append(
        AlgorithmExperimentSetup(mlrose.genetic_alg, ga_param_grid, ga_plots))

    # MIMIC
    mimic_param_grid = {
        'keep_pct': [0.1, 0.2, 0.3, 0.4],
        'pop_size': [2000],
        'max_attempts': [vector_length]
    }
    mimic_plots = [('keep_pct', 'linear')]
    algs_params_tuples.append(
        AlgorithmExperimentSetup(mlrose.mimic, mimic_param_grid, mimic_plots))

    return algs_params_tuples


def run_optimization():
    """A task to compare some optimization problems
    """
    onemax_task()
    fourpeaks_task()
    max_kcolor_task()

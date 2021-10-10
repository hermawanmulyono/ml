import copy
import json
import logging
import multiprocessing
import os
import time
from typing import Dict, Any, List, Tuple, Union

import joblib
import numpy as np

from sklearn.metrics import accuracy_score

from tasks.base import ExperimentBase
from utils.grid import NNResults, MultipleResults, GridTable, \
    grid_args_generator, summarize_grid_table, serialize_grid_table, \
    parse_grid_table, serialize_grid_nn_summary, parse_grid_nn_summary, \
    GridNNSummary, ParamGrid, GridSummary
from utils.outputs import nn_joblib, nn_grid_table, nn_grid_summary, \
    nn_parameter_plot

import mlrose_hiive as mlrose
import plotly.express as px

from utils.data import gen_2d_data
from utils.plots import parameter_plot

HIDDEN_NODES = [16] * 4

REPEATS = 24


class NNExperiment(ExperimentBase):

    def __init__(self, algorithm_name: str, param_grid: ParamGrid,
                 alg_plots: List[Tuple[str, str]], x_train: np.ndarray,
                 y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                 repeats: int):
        self.algorithm_name = algorithm_name
        self.param_grid = param_grid
        self.alg_plots = alg_plots
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.repeats = repeats

    @property
    def grid_table_json(self) -> str:
        return nn_grid_table(self.algorithm_name)

    def grid_run(self) -> GridTable:
        return grid_run(self.x_train, self.y_train, self.x_val, self.y_val,
                        self.param_grid, self.repeats)

    def summarize_grid_table(self, grid_table: GridTable):
        return summarize_grid_table(grid_table, 'nn')

    @property
    def grid_summary_json_path(self):
        return nn_grid_summary(self.algorithm_name)

    def serialize_grid_summary(self,
                               grid_summary: GridSummary) -> Dict[str, Any]:
        return serialize_grid_nn_summary(grid_summary)

    def parse_grid_summary(
            self, grid_summary_serialized: Dict[str, Any]) -> GridSummary:
        return parse_grid_nn_summary(grid_summary_serialized)

    def sync_parameter_plots(self, grid_summary: GridSummary):
        sync_nn_parameter_plots(grid_summary, self.alg_plots,
                                self.algorithm_name)


def num_trainable_params(hidden_nodes: List[int], n_inputs: int,
                         n_outputs: int):
    """Calculate the number of weights in a neural network

    Illustration:
    https://stats.stackexchange.com/questions/296981/formula-for-number-of-weights-in-neural-network

    Args:
        hidden_nodes: Hidden nodes definition
        n_inputs: Number of inputs
        n_outputs: Number of outputs

    Returns:
        Number of trainable parameters

    """
    nodes = [n_inputs] + hidden_nodes + [n_outputs]
    num_params = 0
    for i in range(1, len(nodes)):
        layer_width = nodes[i]
        prev_layer_width = nodes[i - 1]

        weights = layer_width * prev_layer_width
        num_bias = layer_width

        num_params += weights
        num_params += num_bias

    return num_params


NN_NUM_PARAMS = num_trainable_params(HIDDEN_NODES, 2, 1)


def schedule_fn(t, offset: int):
    return mlrose.GeomDecay().evaluate(t + offset)


def get_nn(hidden_nodes=None,
           activation='relu',
           algorithm='random_hill_climb',
           max_iters=100,
           bias=True,
           is_classifier=True,
           learning_rate=0.1,
           early_stopping=False,
           clip_max=10000000000.0,
           restarts=0,
           init_temp=1.0,
           decay=0.99,
           min_temp=0.001,
           pop_size=200,
           mutation_prob=0.1,
           max_attempts=10,
           random_state=None,
           curve=False):
    """Constructs a mlrose.NeuralNetwork object, but
     with serializable arguments. In particular, the
     GeomDecay() is replaced with `init_temp`, `decay`,
     and `min_temp`."""
    schedule = mlrose.GeomDecay(init_temp, decay, min_temp)

    nn = mlrose.NeuralNetwork(hidden_nodes, activation, algorithm, max_iters,
                              bias, is_classifier, learning_rate,
                              early_stopping, clip_max, restarts, schedule,
                              pop_size, mutation_prob, max_attempts,
                              random_state, curve)

    return nn


def run_single(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
               y_val: np.ndarray, kwargs: Dict[str, Any]):
    """Run a single neural-network experiment

    Args:
        x_train: Training set features
        y_train: Training set labels
        x_val: Validation set features
        y_val: Validation set labels
        kwargs: Keyword arguments dictionary for constructing
            mlrose.NeuralNetwork

    Returns:

    """
    hidden_nodes = HIDDEN_NODES
    kwargs = copy.deepcopy(kwargs)
    kwargs['hidden_nodes'] = hidden_nodes
    kwargs['curve'] = True
    nn_model = get_nn(**kwargs)

    start = time.time()
    nn_model.fit(x_train, y_train)
    end = time.time()
    fit_time = end - start

    y_pred = nn_model.predict(x_train)
    train_acc = accuracy_score(y_train, y_pred)

    y_pred = nn_model.predict(x_val)
    val_acc = accuracy_score(y_val, y_pred)

    nn_results = NNResults(train_acc, val_acc, fit_time)

    return nn_results


def run_multiple(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, kwargs: Dict[str, Any], repeats: int):

    def args_generator():
        for _ in range(repeats):
            yield x_train, y_train, x_val, y_val, kwargs

    n_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_cpus) as pool:
        multiple_results: MultipleResults = pool.starmap(
            run_single, args_generator())

    return multiple_results


def grid_run(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
             y_val: np.ndarray, param_grid: Dict[str, Union[list, np.ndarray]],
             repeats: int):

    table: GridTable = []
    for kwargs in grid_args_generator(param_grid):
        logging.info(f'Running {kwargs}')
        multiple_results = run_multiple(x_train, y_train, x_val, y_val, kwargs,
                                        repeats)
        table.append((kwargs, multiple_results))

    return table


def simulated_annealing(x_train: np.ndarray, y_train: np.ndarray,
                        x_val: np.ndarray, y_val: np.ndarray, repeats: int):

    algorithm_name = 'simulated_annealing'

    param_grid = {
        'algorithm': [algorithm_name],
        'init_temp': [1.0, 10.0, 100.0],
        'decay': [0.99, 0.999, 0.9999],
        'learning_rate': [1, 0.1, 0.01, 0.001, 0.001]
    }

    alg_plots = [('init_temp', 'logarithmic'), ('decay', 'logarithmic')]

    NNExperiment(algorithm_name, param_grid, alg_plots, x_train, y_train, x_val,
                 y_val, repeats).run()


def hill_climbing(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, repeats: int):
    algorithm_name = 'random_hill_climb'

    param_grid = {
        'algorithm': [algorithm_name],
        'restarts': [0, 10, 50],
        'learning_rate': [1, 0.1, 0.01, 0.001, 0.001]
    }

    alg_plots = [('restarts', 'linear')]

    NNExperiment(algorithm_name, param_grid, alg_plots, x_train, y_train, x_val,
                 y_val, repeats).run()


def genetic_algorithm(x_train: np.ndarray, y_train: np.ndarray,
                      x_val: np.ndarray, y_val: np.ndarray, repeats: int):
    algorithm_name = 'genetic_alg'

    param_grid = {
        'algorithm': [algorithm_name],
        'mutation_prob': np.logspace(-1, -5, 5),
        'learning_rate': [1, 0.1, 0.01, 0.001, 0.001]
    }

    alg_plots = [('mutation_prob', 'logarithmic')]

    NNExperiment(algorithm_name, param_grid, alg_plots, x_train, y_train, x_val,
                 y_val, repeats).run()


def _nn_task_template(algorithm_name: str, param_grid: ParamGrid,
                      alg_plots: List[Tuple[str, str]], x_train: np.ndarray,
                      y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                      repeats: int):

    grid_table_json_path = nn_grid_table(algorithm_name)

    # Only run grid search if the JSON file doesn't exist
    if not os.path.exists(grid_table_json_path):
        grid_table: GridTable = grid_run(x_train, y_train, x_val, y_val,
                                         param_grid, repeats)
        grid_table_serialized = serialize_grid_table(grid_table)

        # Write results to disk
        with open(grid_table_json_path, 'w') as j:
            json.dump(grid_table_serialized, j, indent=2)

    with open(grid_table_json_path) as j:
        grid_table_serialized = json.load(j)

    grid_table = parse_grid_table(grid_table_serialized)
    grid_summary = summarize_grid_table(grid_table, 'nn')

    grid_summary_json_path = nn_grid_summary(algorithm_name)

    # Only summarize results when the JSON file doesn't exist
    if not os.path.exists(grid_summary_json_path):
        with open(grid_summary_json_path, 'w') as j:
            grid_summary_serialized = serialize_grid_nn_summary(grid_summary)
            json.dump(grid_summary_serialized, j, indent=2)

    # Check by opening the serialized results
    with open(grid_summary_json_path) as j:
        grid_summary_serialized = json.load(j)

    grid_summary: GridNNSummary = parse_grid_nn_summary(grid_summary_serialized)
    sync_nn_parameter_plots(grid_summary,
                            alg_plots=alg_plots,
                            alg_name=algorithm_name)


def sync_nn_parameter_plots(grid_summary: GridNNSummary,
                            alg_plots: List[Tuple[str, str]], alg_name: str):
    for y_axis in ['train_accuracy', 'val_accuracy', 'fit_time']:
        for param_name, scale in alg_plots:
            figure_path = nn_parameter_plot(alg_name, param_name, y_axis)

            # Only generate plot if it doesn't exist
            if os.path.exists(figure_path):
                continue

            fig = parameter_plot(grid_summary, param_name, scale, y_axis=y_axis)

            fig.write_image(figure_path)


def run_nn_weights():
    x1_size = 5
    x2_size = 2
    n_train = 5000
    n_val = 500
    n_test = 500
    noise_prob = 0.01

    x_train, y_train, x_val, y_val, x_test, y_test = gen_2d_data(
        x1_size, x2_size, n_train, n_val, n_test, noise_prob)

    assert len(x_train) == len(y_train) == n_train
    assert len(x_val) == len(y_val) == n_val
    assert len(x_test) == len(y_test) == n_test
    '''
    Need to take care of all algorithms
      1. simulated_annealing
      2. simulated_annealing
      3. hill_climb
    '''

    simulated_annealing(x_train, y_train, x_val, y_val, 24)
    hill_climbing(x_train, y_train, x_val, y_val, 24)
    genetic_algorithm(x_train, y_train, x_val, y_val, 24)
    exit(0)

    hidden_nodes = [16] * 4

    weights = None
    steps = 0

    logging.info('Training neural network')
    for i in range(20000):
        kwargs = {'offset': steps}
        schedule = mlrose.CustomSchedule(schedule_fn, **kwargs)
        max_iters = 1
        nn_model = mlrose.NeuralNetwork(
            hidden_nodes,
            algorithm='simulated_annealing',
            # learning_rate=1e-5,
            max_iters=max_iters,
            curve=True,
            schedule=schedule)
        nn_model.fit(x_train, y_train, init_weights=weights)
        steps += max_iters
        weights = nn_model.fitted_weights

        if i % 100 == 0:
            y_pred = nn_model.predict(x_train)
            train_acc = accuracy_score(y_train, y_pred)
            logging.info(f'Train accuracy: {train_acc}')

            y_pred = nn_model.predict(x_val)
            val_acc = accuracy_score(y_val, y_pred)
            logging.info(f'Val accuracy: {val_acc}')

    nn_path = nn_joblib('simulated_annealing')
    joblib.dump(nn_model, nn_path)

    fitness_curve = nn_model.fitness_curve
    fig = px.line(x=list(range(len(fitness_curve))), y=fitness_curve)
    fig.show()

    nn_model = joblib.load(nn_path)

    y_pred = nn_model.predict(x_train)
    train_acc = accuracy_score(y_train, y_pred)
    logging.info(f'Train accuracy: {train_acc}')

    y_pred = nn_model.predict(x_val)
    val_acc = accuracy_score(y_val, y_pred)
    logging.info(f'Val accuracy: {val_acc}')

    fitness_curve = nn_model.fitness_curve
    fig = px.line(x=list(range(len(fitness_curve))), y=fitness_curve)
    fig.show()

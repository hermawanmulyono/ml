import logging
from typing import List, Callable, Dict, Any, Iterable

import numpy as np

from utils.models import get_decision_tree, grid_search
from utils.plots import training_size_curve, complexity_curve, ModelType


def dt_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray, train_sizes: List[float], dataset_name: str,
            n_jobs):
    """Task for decision tree

    The tasks are:
      1. Grid search to tune `ccp_alpha`
      2. Training size curve
      3. Complexity curve based on `ccp_alpha`


    Args:
        x_train: Training set features
        y_train: Training set labels
        x_val: Validation set features
        y_val: Validation set labels
        train_sizes: List of fractions of training sizes
        dataset_name: Name of the dataset
        n_jobs: Number of jobs

    Returns:

    """
    constructor_fn = get_decision_tree
    default_params = {'ccp_alpha': 0.001}
    ccp_alphas = np.logspace(0, -5, num=10)
    param_grid = {'ccp_alpha': ccp_alphas}
    param_name = 'ccp_alpha'
    param_range = ccp_alphas
    dataset_name = dataset_name
    model_name = 'DT'

    _task_template(constructor_fn, default_params,
                   param_grid, x_train, y_train, x_val, y_val, train_sizes,
                   param_name, param_range, dataset_name, model_name, n_jobs)
    #
    # dt = get_decision_tree(ccp_alpha=0.001)
    #
    # logging.info(f'{dataset_name} - DT - Grid search')
    # dt_gs = grid_search(dt, {'ccp_alpha': ccp_alphas}, x_train, y_train, x_val,
    #                     y_val, n_jobs)
    #
    # best_params = dt_gs.best_params_
    # dt = get_decision_tree(**best_params)
    #
    # logging.info(f'{dataset_name} - DT - Training size curve')
    # fig = training_size_curve(dt,
    #                           x_train,
    #                           y_train,
    #                           x_val,
    #                           y_val,
    #                           train_sizes,
    #                           title=f'{dataset_name} - DT Training Size Curve',
    #                           n_jobs=n_jobs)
    # fig.show()
    #
    # logging.info(f'{dataset_name} - DT - Complexity curve')
    # fig = complexity_curve(dt,
    #                        x_train,
    #                        y_train,
    #                        x_val,
    #                        y_val,
    #                        param_name='ccp_alpha',
    #                        param_range=ccp_alphas,
    #                        title=f'{dataset_name} - DT Complexity Curve',
    #                        log_scale=True,
    #                        n_jobs=n_jobs)
    # fig.show()


def _task_template(constructor_fn: Callable[..., ModelType],
                   default_params: Dict[str, Any], param_grid: Dict[str, Any],
                   x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                   y_val: np.ndarray, train_sizes: List[float], param_name: str,
                   param_range: Iterable, dataset_name: str, model_name: str,
                   n_jobs: int):
    """Template function for the tasks for different models

    Args:
        constructor_fn: `constructor_fn(**default_params)` gives a default model
            object
        default_params: `constructor_fn(**default_params)` gives a default model
            object
        param_grid: Parameter grid for grid search
        x_train: Training set features
        y_train: Training set labels
        x_val: Validation set features
        y_val: Validation set labels
        train_sizes: List of fractions of training sizes
        param_name: Parameter name to plot on complexity curve
        param_range: Parameter values to plot on complexity curve
        dataset_name: Dataset name for plot titles and logging purposes
        model_name: Model name for plot titles and logging purposes
        n_jobs: Number of jobs

    Returns:
        None.

    """
    dt = constructor_fn(**default_params)
    logging.info(f'{dataset_name} - {model_name} - Grid search')
    dt_gs = grid_search(dt, param_grid, x_train, y_train, x_val, y_val, n_jobs)

    best_params = dt_gs.best_params_
    dt = get_decision_tree(**best_params)

    logging.info(f'{dataset_name} - {model_name} - Training size curve')
    fig = training_size_curve(dt,
                              x_train,
                              y_train,
                              x_val,
                              y_val,
                              train_sizes,
                              title=f'{dataset_name} - DT Training Size Curve',
                              n_jobs=n_jobs)
    fig.show()

    logging.info(f'{dataset_name} - {model_name} - Complexity curve')
    fig = complexity_curve(
        dt,
        x_train,
        y_train,
        x_val,
        y_val,
        param_name=param_name,
        param_range=param_range,
        title=f'{dataset_name} - {model_name} Complexity Curve',
        log_scale=True,
        n_jobs=n_jobs)
    fig.show()

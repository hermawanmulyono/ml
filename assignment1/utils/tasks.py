import copy
import logging
import os
from typing import List, Callable, Dict, Any, Iterable

import numpy as np

from utils.models import get_decision_tree, grid_search, get_knn, get_svm, \
    get_boosting, get_nn, grid_search_nn
from utils.nnestimator import NeuralNetworkEstimator, training_curves
from utils.plots import training_size_curve, complexity_curve, ModelType

OUTPUT_DIRECTORY = 'outputs'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def dt_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray, train_sizes: List[float], dataset_name: str,
            n_jobs: int):
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

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs)


def knn_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
             y_val: np.ndarray, train_sizes: List[float], dataset_name: str,
             n_jobs: int):
    constructor_fn = get_knn
    default_params = {'n_neighbors': 9}
    k_values = [2**p + 1 for p in range(6)]
    param_grid = {'n_neighbors': k_values}
    param_name = 'n_neighbors'
    param_range = k_values
    dataset_name = dataset_name
    model_name = 'KNN'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs)


def svm_poly_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, train_sizes: List[float],
                  dataset_name: str, n_jobs: int):
    constructor_fn = get_svm
    default_params = {'kernel': 'poly'}
    degree_values = [1, 2, 3, 4, 5]
    param_grid = {'C': [0.1, 1.0, 10.0], 'degree': [1, 2, 3, 4, 5]}
    param_name = 'degree'
    param_range = degree_values
    dataset_name = dataset_name
    model_name = 'SVM - Polynomial'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs)


def svm_rbf_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, train_sizes: List[float], dataset_name: str,
                 n_jobs: int):
    constructor_fn = get_svm
    default_params = {'kernel': 'rbf'}
    gamma_values = np.arange(2, -3, 5)
    param_grid = {'C': [0.1, 1.0, 10.0], 'gamma': gamma_values}
    param_name = 'gamma'
    param_range = gamma_values
    dataset_name = dataset_name
    model_name = 'SVM - RBF'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs)


def boosting_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, train_sizes: List[float],
                  dataset_name: str, n_jobs: int):
    constructor_fn = get_boosting
    default_params = {'n_estimators': 512, 'ccp_alpha': 0.001}
    param_grid = {'n_estimators': [512]}
    param_name = 'n_estimators'
    param_range = [512]
    dataset_name = dataset_name
    model_name = 'Boosting'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs)


def neural_network_task(x_train: np.ndarray, y_train: np.ndarray,
                        x_val: np.ndarray, y_val: np.ndarray,
                        train_sizes: List[float], dataset_name: str,
                        n_jobs: int):
    in_features = x_train.shape[1]

    model_name = 'NN'

    path_to_state_dict = f'{OUTPUT_DIRECTORY}/{model_name.lower()}' \
                         f'_{dataset_name.lower()}.pt'
    if os.path.exists(path_to_state_dict):
        nn = NeuralNetworkEstimator.from_state_dict(path_to_state_dict)
    else:
        default_params = {
            'in_features': in_features,
            'num_classes': 10,
            'hidden_layers': [24] * 4,
            'learning_rate': 1e-5,
            'batch_size': 128,
            'epochs': 100,
            'verbose': True
        }

        param_grid = {
            'hidden_layers': [[16] * n for n in [2, 4, 8, 16]],
            'learning_rate': [3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
        }
        best_score_kwargs = grid_search_nn(default_params, param_grid, x_train,
                                           y_train, x_val, y_val)

        nn = best_score_kwargs[2]
        nn.save(path_to_state_dict)

    loss_fig, acc_fig = training_curves(nn.training_log)

    loss_fig_path = f'{OUTPUT_DIRECTORY}/loss_{model_name.lower()}' \
                    f'_{dataset_name}.png'
    loss_fig.write_image(loss_fig_path)

    acc_fig_path = f'{OUTPUT_DIRECTORY}/acc_{model_name.lower()}' \
                   f'_{dataset_name}.png'
    acc_fig.write_image(acc_fig_path)


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
    model = constructor_fn(**default_params)
    logging.info(f'{dataset_name} - {model_name} - Grid search')
    dt_gs = grid_search(model, param_grid, x_train, y_train, x_val, y_val,
                        n_jobs)

    logging.info(f'Best parameters found {dt_gs.best_params_}')

    best_params = copy.deepcopy(default_params)
    best_params.update(dt_gs.best_params_)

    model = constructor_fn(**best_params)

    logging.info(f'{dataset_name} - {model_name} - Training size curve')
    fig = training_size_curve(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        train_sizes,
        title=f'{dataset_name} - {model_name} Training Size Curve',
        n_jobs=n_jobs)
    training_fig_path = f'{OUTPUT_DIRECTORY}/' \
                        f'training_{model_name}_{dataset_name}.png'
    fig.write_image(training_fig_path)

    logging.info(f'{dataset_name} - {model_name} - Complexity curve')
    fig = complexity_curve(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        param_name=param_name,
        param_range=param_range,
        title=f'{dataset_name} - {model_name} Complexity Curve',
        log_scale=True,
        n_jobs=n_jobs)
    complexity_fig_path = f'{OUTPUT_DIRECTORY}/' \
                          f'complexity_{model_name}_{dataset_name}.png'
    fig.write_image(complexity_fig_path)

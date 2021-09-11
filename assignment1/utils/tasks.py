import copy
import json
import logging
import os
import time

import joblib
import pickle
from typing import List, Callable, Dict, Any, Iterable, NamedTuple, Optional, \
    Tuple, Union

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from utils.models import get_decision_tree, get_knn, get_svm, \
    get_boosting, get_nn, SklearnModel
from utils.nnestimator import NeuralNetworkEstimator, training_curves
from utils.plots import training_size_curve, complexity_curve, ModelType

OUTPUT_DIRECTORY = 'outputs'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def dt_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray, train_sizes: List[float], dataset_name: str,
            n_jobs: int, retrain: bool):
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
        retrain: If True, the decision tree will be retrained

    Returns:

    """
    constructor_fn = get_decision_tree
    default_params = {'ccp_alpha': 0.001}
    ccp_alphas: List[float] = np.logspace(0, -5, num=10).tolist()
    param_grid = {'ccp_alpha': ccp_alphas}
    param_name = 'ccp_alpha'
    param_range = ccp_alphas
    dataset_name = dataset_name
    model_name = 'DT'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs, retrain)


def knn_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
             y_val: np.ndarray, train_sizes: List[float], dataset_name: str,
             n_jobs: int, retrain: bool):
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
                   dataset_name, model_name, n_jobs, retrain)


def svm_poly_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, train_sizes: List[float],
                  dataset_name: str, n_jobs: int, retrain: bool):
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
                   dataset_name, model_name, n_jobs, retrain)


def svm_rbf_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, train_sizes: List[float], dataset_name: str,
                 n_jobs: int, retrain: bool):
    constructor_fn = get_svm

    # Equivalent to "scale" option in scikit-learn
    default_gamma = 1 / (x_val.shape[1] * x_train.var())

    default_params = {'kernel': 'rbf'}
    gamma_values_np: np.ndarray = default_gamma * np.logspace(3, -3, num=10)
    gamma_values: List[float] = gamma_values_np.tolist()
    param_grid = {'C': [0.1, 1.0, 10.0, 100], 'gamma': gamma_values}
    param_name = 'gamma'
    param_range = gamma_values
    dataset_name = dataset_name
    model_name = 'SVM - RBF'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs, retrain)


def boosting_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, train_sizes: List[float],
                  dataset_name: str, n_jobs: int, retrain: bool):
    constructor_fn = get_boosting
    default_params = {'n_estimators': 256, 'ccp_alpha': 0.001}
    param_grid = {'n_estimators': [256]}
    param_name = 'n_estimators'
    param_range = [256]
    dataset_name = dataset_name
    model_name = 'Boosting'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, train_sizes, param_name, param_range,
                   dataset_name, model_name, n_jobs, retrain)


def neural_network_task(x_train: np.ndarray, y_train: np.ndarray,
                        x_val: np.ndarray, y_val: np.ndarray,
                        train_sizes: List[float], dataset_name: str,
                        n_jobs: int, retrain: bool):
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
            'hidden_layers': [[16] * n for n in [2, 4, 6, 8, 10, 12, 14, 16]],
            'learning_rate': [3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
        }
        best_score_kwargs_nn = grid_search_nn(default_params, param_grid,
                                              x_train, y_train, x_val, y_val)

        nn = best_score_kwargs_nn.best_model
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
                   n_jobs: int, retrain: bool):
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
        retrain: If True, the model will be retrained

    Returns:
        None.

    """

    # Instantiate model
    model_filename = f'{OUTPUT_DIRECTORY}/' \
                     f'{model_name.lower()}_{dataset_name.lower()}.joblib'

    if retrain or (not os.path.exists(model_filename)):
        # Grid search
        logging.info(f'{dataset_name} - {model_name} - Grid search')
        gs = grid_search(constructor_fn, default_params, param_grid, x_train,
                         y_train, x_val, y_val, n_jobs)

        # Save results into JSON
        gs_dict = {
            'best_accuracy': gs.best_accuracy,
            'best_kwargs': gs.best_kwargs,
            'table': gs.table
        }

        # Save grid search results
        gs_results_filepath = f'{OUTPUT_DIRECTORY}/' \
                              f'{model_name}_{dataset_name}_gs.json'
        with open(gs_results_filepath, 'w') as file:
            json.dump(gs_dict, file, indent=4)

        logging.info(f'Best parameters found {gs.best_kwargs}')
        best_params = copy.deepcopy(default_params)
        best_params.update(gs.best_kwargs)

        # Instantiate object based on the best parameters
        model = gs.best_model

        # Save best model
        joblib.dump(model, model_filename)

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
                            f'{model_name}_{dataset_name}_training_size.png'
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
                              f'{model_name}_{dataset_name}_complexity.png'
        fig.write_image(complexity_fig_path)

    else:
        logging.info(f'{model_filename} found. Loading model from disk.')
        model = joblib.load(model_filename)

    return model


class GridSearchResults(NamedTuple):
    best_accuracy: Optional[float]
    best_kwargs: Optional[dict]
    best_fit_time: Optional[float]
    best_model: Optional[Union[NeuralNetworkEstimator, SklearnModel]]
    table: List[Tuple[dict, dict]]


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


def grid_search(constructor_fn: Callable[..., ModelType], default_params: dict,
                param_grid: dict, x_train: np.ndarray, y_train: np.ndarray,
                x_val: np.ndarray, y_val: np.ndarray, n_jobs):

    def fit_fn(**kwargs):
        model = constructor_fn(**kwargs)
        start = time.time()
        model.fit(x_train, y_train)
        end = time.time()
        fit_time = end - start

        return model, fit_time

    results = _grid_search_template(fit_fn, default_params, param_grid,
                                    x_train, y_train, x_val, y_val)

    return results


def grid_search_nn(default_params: dict, param_grid: dict, x_train: np.ndarray,
                   y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):

    def fit_fn(in_features, num_classes, hidden_layers, learning_rate,
               batch_size, epochs, verbose):
        nn_ = get_nn(in_features=in_features,
                     num_classes=num_classes,
                     hidden_layers=hidden_layers)
        start = time.time()
        nn_.fit(x_train,
                y_train,
                x_val,
                y_val,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose)
        end = time.time()
        fit_time = end - start

        return nn_, fit_time

    results = _grid_search_template(fit_fn, default_params, param_grid,
                                    x_train, y_train, x_val, y_val)

    return results


def _grid_search_template(fit_fn: Callable[..., tuple], default_params: dict,
                          param_grid: dict, x_train: np.ndarray,
                          y_train: np.ndarray, x_val: np.ndarray,
                          y_val: np.ndarray):

    if len(x_train) != len(y_train):
        raise ValueError

    if len(x_val) != len(y_val):
        raise ValueError

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError

    param_names: List[str] = list(param_grid.keys())
    grid_index = [0] * len(param_names)

    best_accuracy: Optional[float] = None
    best_kwargs: Optional[dict] = None
    best_fit_time: Optional[float] = None
    best_model: Optional[NeuralNetworkEstimator] = None
    table: List[Tuple[dict, dict]] = []

    while True:
        kwargs = copy.deepcopy(default_params)
        update = {
            key: param_grid[key][gi] for gi, key in zip(grid_index, param_names)
        }
        kwargs.update(update)

        logging.info(f'{kwargs}')

        model, fit_time = fit_fn(**kwargs)

        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)

        table.append((kwargs, {'accuracy': accuracy, 'fit_time': fit_time}))

        if (best_accuracy is None) or (best_accuracy < accuracy):
            logging.info('Updating model')
            best_accuracy = accuracy
            best_fit_time = fit_time
            best_kwargs = kwargs
            best_model = model

        grid_index = _increase_grid_index(grid_index, param_grid)
        if all([gi == 0 for gi in grid_index]):
            break

    results = GridSearchResults(best_accuracy, best_kwargs, best_fit_time,
                                best_model, table)

    return results

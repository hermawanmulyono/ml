import copy
import logging
import time
from typing import NamedTuple, Optional, Union, List, Tuple, Callable

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.models import SklearnModel, get_nn
from utils.nnestimator import NeuralNetworkEstimator


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


ModelType = Union[KNeighborsClassifier, SVC, DecisionTreeClassifier,
                  AdaBoostClassifier, NeuralNetworkEstimator]


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

    results = _grid_search_template(fit_fn, default_params, param_grid, x_train,
                                    y_train, x_val, y_val)

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

    results = _grid_search_template(fit_fn, default_params, param_grid, x_train,
                                    y_train, x_val, y_val)

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

        y_pred = model.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_pred)

        y_pred = model.predict(x_val)
        val_accuracy = accuracy_score(y_val, y_pred)

        table.append((kwargs, {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'fit_time': fit_time
        }))

        if (best_accuracy is None) or (best_accuracy < val_accuracy):
            logging.info('Updating model')
            best_accuracy = val_accuracy
            best_fit_time = fit_time
            best_kwargs = kwargs
            best_model = model

        grid_index = _increase_grid_index(grid_index, param_grid)
        if all([gi == 0 for gi in grid_index]):
            break

    results = GridSearchResults(best_accuracy, best_kwargs, best_fit_time,
                                best_model, table)

    return results

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

from utils.models import SklearnModel, get_boosting
from utils.nnestimator import NeuralNetworkEstimator, train_nn_multiple


class GridSearchResults(NamedTuple):
    """A structure to store grid search results.

    The members are:
      - best_accuracy: Best accuracy of the model on the
        validation set
      - best_kwargs: Best keyword arguments to construct
        the model with best accuracy. This contains the
        best parameters to use.
      - best_fit_time: The best model training time.
      - best_model: The best model object
      - table: A list `[..., (kwargs, results), ...]` where
        `kwargs` is a set of keyword arguments, `results`
        is the corresponding results

             {'train_accuracy': ...,
             'val_accuracy': ...,
             'fit_time': ...}

        This is the table of all grid search results.


    """
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
                x_val: np.ndarray, y_val: np.ndarray):

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

    def fit_fn(in_features, num_classes, nn_size, learning_rate,
               batch_size, epochs, verbose):
        return train_nn_multiple(x_train, y_train, x_val, y_val, in_features,
                                 num_classes, nn_size, learning_rate,
                                 batch_size, epochs, verbose)

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

        logging.info(f'Training with {kwargs}')
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


def grid_search_boosting(constructor_fn: Callable[..., AdaBoostClassifier],
                         default_params: dict, param_grid: dict,
                         x_train: np.ndarray, y_train: np.ndarray,
                         x_val: np.ndarray, y_val: np.ndarray):

    def fit_fn(n_estimators, ccp_alpha, max_depth):
        boosting_model = constructor_fn(n_estimators, ccp_alpha, max_depth)
        start = time.time()
        boosting_model.fit(x_train, y_train)
        finish = time.time()
        _fit_time = finish - start
        return boosting_model, _fit_time

    param_names: List[str] = list(param_grid.keys())
    grid_index = [0] * len(param_names)

    best_accuracy: Optional[float] = None
    best_kwargs: Optional[dict] = None
    best_fit_time: Optional[float] = None
    table: List[Tuple[dict, dict]] = []

    while True:
        kwargs = copy.deepcopy(default_params)
        update = {
            key: param_grid[key][gi] for gi, key in zip(grid_index, param_names)
        }
        kwargs.update(update)

        model, fit_time = fit_fn(**kwargs)

        train_scores = [s for s in model.staged_score(x_train, y_train)]
        val_scores = [s for s in model.staged_score(x_val, y_val)]
        assert len(train_scores) == len(val_scores)

        iterations = [i + 1 for i, _ in enumerate(val_scores)]

        for train_accuracy, val_accuracy, num_estimators in zip(
                train_scores, val_scores, iterations):
            params = copy.deepcopy(kwargs)
            params['n_estimators'] = num_estimators
            table.append((params, {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'fit_time': fit_time
            }))

            if (best_accuracy is None) or (best_accuracy < val_accuracy):
                logging.info('Updating model')
                best_accuracy = val_accuracy
                best_fit_time = fit_time
                best_kwargs = params

        grid_index = _increase_grid_index(grid_index, param_grid)
        if all([gi == 0 for gi in grid_index]):
            break

    best_model: AdaBoostClassifier = get_boosting(**best_kwargs)
    best_model.fit(x_train, y_train)

    results = GridSearchResults(best_accuracy, best_kwargs, best_fit_time,
                                best_model, table)

    return results

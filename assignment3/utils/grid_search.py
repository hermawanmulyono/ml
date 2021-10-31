import copy
import logging
from typing import List, Callable, Optional, Tuple, NamedTuple

import numpy as np
from sklearn.metrics import accuracy_score

from utils.nnestimator import train_nn, NeuralNetworkEstimator


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
    best_model: Optional[NeuralNetworkEstimator]
    table: List[Tuple[dict, dict]]


def _increase_grid_index(grid_index_: List[int], param_grid: dict):
    """Gets the next set of parameters for grid search

    Note:
        This function assumes the default ordered dictionary
        keys in Python 3.7.

    Args:
        grid_index_: Current grid index. Each element
            indicates the corresponding element in
            param_grid.
        param_grid: Parameter grid dictionary in
            Scikit-Learn style.

    Returns:
        The next grid_index.

    """
    param_names: List[str] = list(param_grid.keys())

    grid_index_ = copy.deepcopy(grid_index_)

    for i in range(len(grid_index_)):
        grid_index_[i] += 1

        if grid_index_[i] >= len(param_grid[param_names[i]]):
            grid_index_[i] = 0
        else:
            break

    return grid_index_


def grid_search_nn(param_grid: dict, x_train: np.ndarray,
                   y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    """Performs grid search for NeuralNetworkEstimator

    Args:
        param_grid:
        x_train:
        y_train:
        x_val:
        y_val:

    Returns:

    """

    def fit_fn(in_features, num_classes, nn_size, learning_rate, batch_size,
               epochs, verbose):
        return train_nn(x_train, y_train, x_val, y_val, in_features,
                        num_classes, nn_size, learning_rate, batch_size, epochs,
                        verbose)

    results = _grid_search_template(fit_fn, param_grid, x_train,
                                    y_train, x_val, y_val)

    return results


def _grid_search_template(fit_fn: Callable[..., tuple],
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
        kwargs = {
            key: param_grid[key][gi] for gi, key in zip(grid_index, param_names)
        }

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

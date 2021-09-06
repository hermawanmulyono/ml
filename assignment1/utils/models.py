import copy
import logging
from typing import Optional, List, Union, Tuple

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from utils.nnestimator import NeuralNetworkEstimator

SklearnModel = Union[KNeighborsClassifier, SVC, DecisionTreeClassifier,
                     AdaBoostClassifier]


def get_decision_tree(ccp_alpha: float) -> DecisionTreeClassifier:
    """Constructs a decision tree with pruning

    Args:
        ccp_alpha: Pruning minimal cost-complexity parameter alpha

    Returns:
        A decision tree classifier object

    """
    dt = DecisionTreeClassifier(criterion='entropy',
                                splitter='random',
                                ccp_alpha=ccp_alpha)
    return dt


def get_boosting(n_estimators: int, ccp_alpha: float) -> AdaBoostClassifier:
    base_estimator = DecisionTreeClassifier(criterion='entropy',
                                            splitter='best',
                                            ccp_alpha=ccp_alpha)

    adaboost = AdaBoostClassifier(base_estimator, n_estimators=n_estimators)

    return adaboost


def get_svm(kernel: str, **kwargs) -> SVC:
    return SVC(kernel=kernel, **kwargs)


def get_knn(n_neighbors: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors)


def get_nn(in_features: int, num_classes: int,
           hidden_layers: List[int]) -> NeuralNetworkEstimator:

    nn_est = NeuralNetworkEstimator(in_features, num_classes, hidden_layers)

    return nn_est


def grid_search(model: SklearnModel, param_grid: dict, x_train: np.ndarray,
                y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                n_jobs):

    if len(x_train) != len(y_train):
        raise ValueError

    if len(x_val) != len(y_val):
        raise ValueError

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError

    x_concat = np.concatenate([x_train, x_val], axis=0)
    y_concat = np.concatenate([y_train, y_val], axis=0)

    assert len(x_concat) == len(y_concat)

    n_train = len(x_train)
    n_concat = len(x_concat)

    cv = [(np.arange(n_train), np.arange(n_train, n_concat))]

    gscv = GridSearchCV(model,
                        param_grid,
                        verbose=2,
                        refit=False,
                        cv=cv,
                        n_jobs=n_jobs)

    gscv.fit(x_concat, y_concat)

    return gscv


def grid_search_nn(default_params: dict, param_grid: dict, x_train: np.ndarray,
                   y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):

    param_names: List[str] = list(param_grid.keys())
    grid_index = [0] * len(param_names)

    def increase(grid_index_: List[int]):
        grid_index_ = copy.deepcopy(grid_index_)

        for i in range(len(grid_index_)):
            grid_index_[i] += 1

            if grid_index_[i] >= len(param_grid[param_names[i]]):
                grid_index_[i] = 0
            else:
                break

        return grid_index_

    def fit_nn(in_features, num_classes, hidden_layers, learning_rate,
               batch_size, epochs, verbose):
        nn_ = get_nn(in_features=in_features,
                     num_classes=num_classes,
                     hidden_layers=hidden_layers)
        nn_.fit(x_train,
                y_train,
                x_val,
                y_val,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose)

        return nn_

    best_score_kwargs: Tuple[float, dict, NeuralNetworkEstimator] = (None, None, None)

    while True:
        kwargs = copy.deepcopy(default_params)
        update = {
            key: param_grid[key][gi] for gi, key in zip(grid_index, param_names)
        }
        kwargs.update(update)

        logging.info(f'{kwargs}')
        nn = fit_nn(**kwargs)

        vall_acc = np.array(nn.training_log['val_accuracy'])
        vall_acc = vall_acc[np.logical_not(np.isnan(vall_acc))]
        score = np.max(vall_acc)
        if best_score_kwargs[0] is None or best_score_kwargs[0] > score:
            best_score_kwargs = (score, kwargs, nn)

        grid_index = increase(grid_index)
        if all([gi == 0 for gi in grid_index]):
            break

    return best_score_kwargs

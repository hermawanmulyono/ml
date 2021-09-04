import copy
from typing import List, Union, Tuple, Dict, Any

import numpy as np
from plotly import graph_objects as go
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from utils.nnestimator import NeuralNetworkEstimator

ModelType = Union[KNeighborsClassifier, SVC, DecisionTreeClassifier,
                  AdaBoostClassifier, NeuralNetworkEstimator]


def visualize_2d_data(x_data: np.ndarray,
                      y_data: np.ndarray,
                      title: str = None) -> go.Figure:
    """Visualizes 2D data

    Args:
        x_data: A (num_examples, 2) array.
        y_data: A (num_examples, ) array of labels corresponding to `x_data`.
        title: If given, will add plot title

    Returns:
        A plotly figure object

    """

    fig = go.Figure()
    fig = _add_scatter(fig, x_data, y_data, scatter_alpha=0.2, scatter_size=15)
    fig.update_layout({'xaxis_title': 'x1', 'yaxis_title': 'x2'})

    if title is not None:
        fig.update_layout({'title': title})

    return fig


def visualize_2d_decision_boundary(model,
                                   x1_max: float,
                                   x2_max: float,
                                   x_data,
                                   y_data,
                                   title: str = None) -> go.Figure:
    """Visualizes the decision boundary of a trained model

    This function only works with model trained with 2D data.

    A meshgrid [0, x1_max] x [0, x2_max] is created.

    Args:
        x_data:
        y_data:
        model: A trained model which implements predict_proba() or predict()
        x1_max: Maximum value of first axis
        x2_max: Maximum value of second axis
        title: If given, will add plot title

    Returns:
        A plotly figure object
    """

    # Implementation inspired by scikit-learn example
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    step = min(x1_max / 100, x2_max / 100)

    x1 = np.arange(0, x1_max, step)
    x2 = np.arange(0, x2_max, step)
    xx1, xx2 = np.meshgrid(np.arange(0, x1_max, step),
                           np.arange(0, x2_max, step))

    x_grid = np.stack([xx1.flatten(), xx2.flatten()], axis=-1)

    if type(model) == SVC:
        model: SVC
        z = model.decision_function(x_grid)
        z = (z > 0).astype(np.float)

    else:
        # Grab the second class
        z = model.predict_proba(x_grid)[:, 1]

    z = z.reshape(xx1.shape)
    colorscale = [[0, 'rgba(128,128,200,0.3)'], [1.0, 'rgba(200,128,128,0.3)']]
    fig = go.Figure()
    fig.add_trace(
        go.Contour(z=z,
                   x=x1,
                   y=x2,
                   colorscale=colorscale,
                   contours={'showlines': False},
                   colorbar={
                       'len': 0.8,
                       'nticks': 10
                   }))

    fig = _add_scatter(fig, x_data, y_data, scatter_alpha=0.5, scatter_size=5)
    fig.update_layout({'xaxis_title': 'x1', 'yaxis_title': 'x2'})

    if title:
        fig.update_layout({'title': title})

    return fig


def _add_scatter(fig: go.Figure, x_data, y_data, scatter_alpha: float,
                 scatter_size: int):
    positive_indices = y_data == 1
    x_positive = x_data[positive_indices]
    x_negative = x_data[np.logical_not(positive_indices)]

    fig.add_trace(
        go.Scatter(x=x_positive[:, 0],
                   y=x_positive[:, 1],
                   mode='markers',
                   name='positive',
                   marker={
                       'color': f'rgba(255,0,0,{scatter_alpha})',
                       'size': scatter_size
                   }))
    fig.add_trace(
        go.Scatter(x=x_negative[:, 0],
                   y=x_negative[:, 1],
                   mode='markers',
                   name='negative',
                   marker={
                       'color': f'rgba(0,0,255,{scatter_alpha})',
                       'size': scatter_size
                   }))
    return fig


def training_size_curve(model: ModelType, x_train: np.ndarray,
                        y_train: np.ndarray, x_val: np.ndarray,
                        y_val: np.ndarray, sizes: List[float], title: str,
                        n_jobs) -> go.Figure:
    """Produces a training curve with respect to training size

    Args:
        model: An untrained model
        x_train: Training set features (n_train, n_features)
        y_train: Training set labels (n_train, )
        x_val: Validation set features (n_train, n_features)
        y_val: Validation set labels (n_train, )
        sizes: List of training sizes as fractions e.g.
            `[0.1, 0.25, 0.5, 0.75, 1.0]`.
        title: Plot title
        n_jobs: Number of jobs

    Returns:

    """

    if len(x_train) != len(y_train):
        raise ValueError

    if len(x_val) != len(y_val):
        raise ValueError

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError

    if not all([0 <= s <= 1.0 for s in sizes]):
        raise ValueError

    x_concat = np.concatenate([x_train, x_val], axis=0)
    y_concat = np.concatenate([y_train, y_val], axis=0)

    assert len(x_concat) == len(y_concat)

    n_train = len(x_train)
    n_concat = len(x_concat)

    cv = [(np.arange(n_train), np.arange(n_train, n_concat))]

    # def train_and_eval(size_: float) -> Tuple[float, float]:
    #     # Copy so that the original model is intact
    #     model_copy = copy.deepcopy(model)
    #
    #     n_train = int(size_ * len(x_train))
    #
    #     x_train_ = x_shuffle[:n_train]
    #     y_train_ = y_shuffle[:n_train]
    #
    #     model_copy.fit(x_train_, y_train_)
    #
    #     y_pred_train_ = model_copy.predict(x_train_)
    #     train_acc_ = accuracy_score(y_train_, y_pred_train_)
    #
    #     y_pred_val_ = model_copy.predict(x_val)
    #     val_acc_ = accuracy_score(y_val, y_pred_val_)
    #
    #     return train_acc_, val_acc_

    train_sizes_ags, train_accs_, val_accs_ = learning_curve(
        model,
        x_concat,
        y_concat,
        train_sizes=sizes,
        cv=cv,
        # scoring='accuracy',
        n_jobs=n_jobs,
        shuffle=True)

    # train_val_accs = map(train_and_eval, sizes)
    # train_accs, val_accs = zip(*train_val_accs)

    train_accs = train_accs_.flatten()
    val_accs = val_accs_.flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=train_accs, mode='lines', name='train'))
    fig.add_trace(go.Scatter(x=sizes, y=val_accs, mode='lines', name='val'))
    fig.update_layout({
        'xaxis_title': 'Training size',
        'yaxis_title': 'Accuracy'
    })

    fig.update_layout({'title': title})

    return fig


def complexity_curve(model: ModelType,
                     x_train: np.ndarray,
                     y_train: np.ndarray,
                     x_val: np.ndarray,
                     y_val: np.ndarray,
                     param_name: str,
                     param_range: Any,
                     title: str,
                     log_scale,
                     n_jobs: int = 1) -> go.Figure:
    """Plots a complexity curve of a model

    Args:
        log_scale:
        model: An untrained model
        x_train: Training set features (n_train, n_features)
        y_train: Training set labels (n_train, )
        x_val: Validation set features (n_train, n_features)
        y_val: Validation set labels (n_train, )
        param_name: Parameter name to sweep
        param_range: Values of the `param_name`
        title: Plot title
        n_jobs: Number of jobs

    Returns:

    """

    # def train_and_eval(params_: Dict[str, Any]) -> Tuple[float, float]:
    #     # Copy so that the original model is intact
    #     model_copy = copy.deepcopy(model)
    #
    #     model_copy.fit(x_train, y_train)
    #
    #     y_pred_train_ = model_copy.predict(x_train)
    #     train_acc_ = accuracy_score(y_train, y_pred_train_)
    #
    #     y_pred_val_ = model_copy.predict(x_val)
    #     val_acc_ = accuracy_score(y_val, y_pred_val_)
    #
    #     return train_acc_, val_acc_
    #
    # keywords = set(params.keys())
    # if not keywords:
    #     raise ValueError

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

    train_scores_, val_scores_ = validation_curve(model,
                                                  x_concat,
                                                  y_concat,
                                                  param_name=param_name,
                                                  param_range=param_range,
                                                  cv=cv,
                                                  scoring='accuracy',
                                                  n_jobs=n_jobs)

    train_scores = train_scores_.flatten()
    val_scores = val_scores_.flatten()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=param_range, y=train_scores, mode='lines', name='train'))
    fig.add_trace(
        go.Scatter(x=param_range, y=val_scores, mode='lines', name='val'))
    fig.update_layout({'xaxis_title': param_name, 'yaxis_title': 'Accuracy'})

    if log_scale:
        fig.update_xaxes(type='log')

    fig.update_layout({'title': title})
    return fig

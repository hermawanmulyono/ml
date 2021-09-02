import os

import numpy as np
import plotly.graph_objects as go
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import torchvision


def gen_2d_data(x1_max: float, x2_max: float, num_examples: int,
                noise_prob: float):
    """Generates 2D dataset

    Each example has coordinates [x1, x2] where:
      - x1 is on interval [0, x1]
      - x2 is on interval [0, x2]

    Args:
        x1_max: Maximum value of first axis.
        x2_max: Maximum value of second axis
        num_examples: Number of examples to generate
        noise_prob: Probability of label flip

    Returns:
        A tuple `(x_data, y_data)` of shapes `(num_points, 2)` and
        `(num_points, )`, respectively.

    """
    x_data = np.random.rand(num_examples, 2) * np.array([[x1_max, x2_max]])

    x1 = x_data[:, 0]
    x2 = x_data[:, 1]

    # Construct labels
    y_data = np.zeros((num_examples,))

    # Conditions for positive labels
    cond1 = x1 / (x1_max / 2) + x2 / x2_max - 1
    cond2 = np.square(
        (x1 - x1_max) / (x1_max / 2)) + np.square(x2 / (x2_max / 2)) - 1
    cond3 = np.max(np.stack(
        [x1_max / 2 - x1, x2_max * 0.7 - x2, x1 - 0.75 * x1_max], axis=-1),
                   axis=-1)

    cond = np.minimum(np.minimum(cond1, cond2), cond3)
    prob = 1 / (1 + np.exp(cond / 0.1))
    rand = np.random.uniform(low=0.1, high=1.0, size=(num_examples,))
    y_data[rand <= prob] = 1

    # Noise
    noise_flags: np.ndarray = np.random.binomial(
        n=2, p=noise_prob, size=(num_examples,)).astype(bool)
    y_data[noise_flags] = 1 - y_data[noise_flags]

    y_data = y_data.astype(np.int)

    return x_data, y_data


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


def visualize_2d_decision_boundary(model, x1_max: float, x2_max: float, x_data,
                                   y_data, title: str = None) -> \
        go.Figure:
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
                   colorbar={'len': 0.8, 'nticks': 10}))

    fig = _add_scatter(fig, x_data, y_data, scatter_alpha=0.5, scatter_size=5)
    fig.update_layout({'xaxis_title': 'x1', 'yaxis_title': 'x2'})

    if title:
        fig.update_layout({'title': title})

    return fig


def get_mnist(train: bool):
    """Gets x_data, y_data of the MNIST dataset

    Args:
        train: If True, this function gets the training set. Otherwise,
            it gets the validation set.

    Returns:
        Tuple (x_data, y_data) where
            1. `x_data` is an array of shape (num_examples, 784)
            2. `y_data` is an array of shape (num_examples, )

    """
    dir_name = 'mnist'
    os.makedirs(dir_name, exist_ok=True)
    mnist = torchvision.datasets.MNIST(dir_name, train, download=True)
    x = np.stack([np.array(x).flatten().copy() for x, _ in mnist])
    y = np.array([y for _, y in mnist])

    return x, y


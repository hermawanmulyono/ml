from typing import List, Iterable

import numpy as np
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def visualize_3d_data(x_data: np.ndarray, y_data: np.ndarray,
                      categories: List[str]) -> go.Figure:
    """Visualizes 3D data

    Args:
        x_data: A (num_examples, 3) array.
        y_data: A (num_examples, ) array of labels corresponding to `x_data`.
        categories: Category names of the data, ordered by indices

    Returns:
        A plotly figure object

    """

    fig = go.Figure()
    fig = _add_scatter_dataset3d(fig,
                                 x_data,
                                 y_data,
                                 scatter_alpha=0.5,
                                 scatter_size=4,
                                 categories=categories)
    fig.update_layout({'xaxis_title': 'x1', 'yaxis_title': 'x2'})

    return fig


def _add_scatter_dataset3d(fig: go.Figure, x_data, y_data, scatter_alpha: float,
                           scatter_size: int, categories: List[str]):
    """Adds Dataset2D scatter plot

    Args:
        fig: A Figure object
        x_data: Dataset2D features
        y_data: Dataset2D labels
        scatter_alpha: Transparency
        scatter_size: Scatter size
        categories: Category names of the data, ordered by indices

    Returns:
        A Figure object with scatter


    """
    possible_labels = sorted(set(y_data))
    if not ((0 <= min(possible_labels)) and
            (max(possible_labels) < len(categories))):
        raise ValueError(
            f'Possible labels are not consistent with the labels {categories}')

    for label in possible_labels:
        category_name = categories[label]

        selected_indices = (y_data == label)
        x_selected = x_data[selected_indices]

        fig.add_trace(
            go.Scatter3d(x=x_selected[:, 0],
                         y=x_selected[:, 1],
                         z=x_selected[:, 2],
                         mode='markers',
                         name=category_name,
                         marker={
                             'size': scatter_size,
                             'opacity': 1 - scatter_alpha
                         }))
    return fig


def simple_line_plot(x, y, x_title: str, y_title: str) -> go.Figure:
    """Generates a simple line plot

    Args:
        x: x-axis data
        y: y-axis data
        x_title: x-axis title
        y_title: y-axis title

    Returns:
        A go.Figure object with corresponding x and y data.

    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.update_layout({'xaxis_title': x_title, 'yaxis_title': y_title})
    return fig


class MatplotlibAdapter:
    """Adapter for a Matplotlib figure to plotly go.Figure()

    Matplotlib offers low-level operations for advanced
    figure manipulations. However, most of the time,
    it is faster to work with plotly. This class can be used
    with code that expects plotly's `write_image()`.

    """

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        plt.close()

    def write_image(self, path_to_image):
        self.fig.savefig(path_to_image)


def visualize_fashion_mnist(x_data: np.ndarray, y_data: np.ndarray,
                            categories: List[str]):
    """Visualizes Fashion-Mnist dataset

    The t-SNE algorithm is used to map a the 784-dimensional
    vectors into 2-dimensional data.

    Some thumbnails will be displayed, along with the
    t-SNE projection.

    The Matplotlib visualization is adapted from
    https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

    Args:
        x_data: Fashion-MNIST feature array
        y_data: The `x_data` labels
        categories: An ordered list of category names

    Returns:
        A MatplotlibAdapter object, which has a
            `write_image()` method, just like go.Figure.
    """

    if len(x_data) != len(y_data):
        raise ValueError('x_data and y_data must have the same length.')

    if x_data.shape[1] != 784:
        raise ValueError('x_data of Fashion-MNIST must be 784-dimensional')

    tsne = TSNE()
    transformed = tsne.fit_transform(x_data)
    assert len(transformed.shape) == 2

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    transformed = MinMaxScaler().fit_transform(transformed)

    possible_labels = sorted(set(y_data))
    if not ((0 <= min(possible_labels)) and
            (max(possible_labels) < len(categories))):
        raise ValueError(
            f'Possible labels are not consistent with the labels {categories}')

    original_indices = np.arange(len(x_data))

    shown_images = np.array([[1.0, 1.0]])  # initially, just something big

    for label in possible_labels:
        category_name = categories[label]

        selected_indices = original_indices[y_data == label]
        x_selected = transformed[selected_indices]

        plt.scatter(x_selected[:, 0], x_selected[:, 1], label=category_name)

        for i in selected_indices:
            # show an annotation box for a group of digits
            dist = np.sum((transformed[i] - shown_images)**2, 1)
            if np.min(dist) < 6e-3:
                # don't show points that are too close
                continue
            shown_images = np.concatenate([shown_images, [transformed[i]]],
                                          axis=0)

            x_disp = x_data[i]
            thumbnail = np.reshape(x_disp, (28, 28))

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(thumbnail, cmap=plt.cm.gray_r),
                transformed[i])
            ax.add_artist(imagebox)

    ax.legend()
    adapter_figure = MatplotlibAdapter(fig, ax)

    return adapter_figure


def feature_importance_chart(feature_importances: np.ndarray):
    """Feature importance bar chart

    The bar plot shows the feature importance scores,
    sorted from the highest to lowest.

    Args:
        feature_importances:

    Returns:
        A go.Figure() object containing the bar chart.

    """

    sorted_importances = -np.sort(-feature_importances)
    x = np.arange(0, len(feature_importances))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=sorted_importances))
    return fig


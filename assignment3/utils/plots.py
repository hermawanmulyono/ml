from typing import List

import numpy as np
from plotly import graph_objects as go


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

    for label, category_name in zip(possible_labels, categories):
        selected_indices = (y_data == label)
        x_positive = x_data[selected_indices]

        fig.add_trace(
            go.Scatter3d(x=x_positive[:, 0],
                         y=x_positive[:, 1],
                         z=x_positive[:, 2],
                         mode='markers',
                         name=category_name,
                         marker={
                             'size': scatter_size,
                             'opacity': 1 - scatter_alpha
                         }))
    return fig

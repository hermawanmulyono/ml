import numpy as np
from plotly import graph_objects as go


def visualize_3d_data(x_data: np.ndarray,
                      y_data: np.ndarray,
                      title: str = None) -> go.Figure:
    """Visualizes 3D data

    Args:
        x_data: A (num_examples, 3) array.
        y_data: A (num_examples, ) array of labels corresponding to `x_data`.
        title: If given, will add plot title

    Returns:
        A plotly figure object

    """

    fig = go.Figure()
    fig = _add_scatter_dataset3d(fig,
                                 x_data,
                                 y_data,
                                 scatter_alpha=0.2,
                                 scatter_size=2)
    fig.update_layout({
        'xaxis_title': 'x1',
        'yaxis_title': 'x2',
        # 'width': 960,
        # 'height': 540
    })

    if title is not None:
        fig.update_layout({'title': title})

    return fig


def _add_scatter_dataset3d(fig: go.Figure, x_data, y_data, scatter_alpha: float,
                           scatter_size: int):
    """Adds Dataset2D scatter plot

    Args:
        fig: A Figure object
        x_data: Dataset2D features
        y_data: Dataset2D labels
        scatter_alpha: Transparency
        scatter_size: Scatter size

    Returns:
        A Figure object with scatter

    """
    positive_indices = y_data == 1

    # Sanity check
    assert np.all(y_data[np.logical_not(positive_indices)] == 0)

    x_positive = x_data[positive_indices]
    x_negative = x_data[np.logical_not(positive_indices)]

    fig.add_trace(
        go.Scatter3d(x=x_positive[:, 0],
                     y=x_positive[:, 1],
                     z=x_positive[:, 2],
                     mode='markers',
                     name='positive',
                     marker={
                         'color': f'rgba(255,0,0,{scatter_alpha})',
                         'size': scatter_size
                     }))
    fig.add_trace(
        go.Scatter3d(x=x_negative[:, 0],
                     y=x_negative[:, 1],
                     z=x_negative[:, 2],
                     mode='markers',
                     name='negative',
                     marker={
                         'color': f'rgba(0,0,255,{scatter_alpha})',
                         'size': scatter_size
                     }))
    return fig

import numpy as np
import plotly.graph_objects as go


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
    cond1 = x1 / (x1_max / 2) + x2 / x2_max
    cond2 = np.square(
        (x1 - x1_max) / (x1_max / 2)) + np.square(x2 / (x2_max / 2))

    cond = np.minimum(cond1, cond2)
    prob = 1 / (1 + np.exp((cond - 1) / 0.1))
    rand = np.random.uniform(low=0.1, high=1.0, size=(num_examples, ))
    y_data[rand <= prob] = 1

    # Noise
    noise_flags: np.ndarray = np.random.binomial(
        n=2, p=noise_prob, size=(num_examples,)).astype(bool)
    y_data[noise_flags] = 1 - y_data[noise_flags]

    return x_data, y_data


def visualize_2d_data(x_data: np.ndarray, y_data: np.ndarray) -> go.Figure:
    """Visualizes 2D data

    Args:
        x_data: A (num_examples, 2) array.
        y_data: A (num_examples, ) array of labels corresponding to `x_data`.

    Returns:
        A plotly figure object

    """

    positive_indices = y_data == 1
    x_positive = x_data[positive_indices]
    x_negative = x_data[np.logical_not(positive_indices)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_positive[:, 0],
                   y=x_positive[:, 1],
                   mode='markers',
                   name='positive',
                   marker={
                       'color': 'rgba(255,0,0,.2)',
                       'size': 15
                   }))
    fig.add_trace(
        go.Scatter(x=x_negative[:, 0],
                   y=x_negative[:, 1],
                   mode='markers',
                   name='negative',
                   marker={
                       'color': 'rgba(0,0,255,.2)',
                       'size': 15
                   }))

    fig.update_layout({'xaxis_title': 'x1', 'yaxis_title': 'x2'})

    return fig

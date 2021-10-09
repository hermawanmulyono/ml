import numpy as np
import plotly.graph_objects as go

from utils.grid import GridSummary, OptimizationSummary, Stats


def parameter_plot(grid_summary: GridSummary, param_name: str, x_scale: str,
                   y_axis: str):
    """Generates plots from the grid search summary

    Args:
        grid_summary: A grid summary of either an
            optimization or a neural network task.
        param_name: Parameter name to plot on x-axis. This
            must be one of the parameters in
            grid_summary.kwargs
        x_scale: Either 'linear' or 'logarithmic'
        y_axis: y-axis variable to plot. This must be one of
            the named parameters of `grid_summary`, except
            'kwargs' and 'table'.

    Returns:
        Plotly go.Figure object.

    """
    if x_scale not in ['linear', 'logarithmic']:
        raise ValueError

    best_params = grid_summary.kwargs
    if param_name not in best_params:
        raise ValueError

    y_axis_valid = hasattr(grid_summary,
                           y_axis) and (y_axis not in {'kwargs', 'table'})
    if not y_axis_valid:
        raise ValueError

    other_params = {k: v for k, v in best_params.items() if k != param_name}

    param_values = []
    y_values = []

    d: OptimizationSummary
    for params, d in grid_summary.table:
        _other_params = {k: v for k, v in params.items() if k != param_name}

        if other_params != _other_params:
            continue

        param_values.append(params[param_name])

        y_value: Stats = getattr(d, y_axis)
        y_values.append(y_value.mean)

    sort_indices = np.argsort(param_values)
    sorted_param_values = [param_values[i] for i in sort_indices]
    sorted_y_values = [y_values[i] for i in sort_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=sorted_param_values,
                   y=sorted_y_values,
                   mode='lines',
                   name=y_axis))

    fig.update_layout({
        'xaxis_title': param_name,
        'yaxis_title': y_axis,
        'width': 640,
        'height': 480
    })

    if x_scale == 'logarithmic':
        fig.update_xaxes(type='log')

    return fig

from typing import List

import numpy as np
import plotly.graph_objects as go

from utils.grid import GridSummary, OptimizationSummary, Stats


def parameter_plot(grid_summary: GridSummary, param_name: str, x_scale: str,
                   y_axis: str, negate_y_axis: bool = False):
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
        negate_y_axis: If True, the y-axis is flipped.

    Returns:
        Plotly go.Figure object.

    """
    if x_scale not in ['linear', 'logarithmic']:
        raise ValueError("x_scale must be either 'linear' or 'logarithmic'.")

    best_params = grid_summary.kwargs
    if param_name not in best_params:
        raise ValueError(f'param_name {param_name} is not one of the '
                         f'best_params keys {set(best_params.keys())}')

    y_axis_valid = hasattr(grid_summary,
                           y_axis) and (y_axis not in {'kwargs', 'table'})
    if not y_axis_valid:
        raise ValueError('Invalid y_axis')

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

    if not negate_y_axis:
        sign = 1.0
    else:
        sign = -1.0

    sorted_y_values = [sign * y_values[i] for i in sort_indices]

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


def fitness_curve_plot(fitness_curve: np.ndarray):
    """Generates fitness curve plot

    Args:
        fitness_curve: A 1-D array indicating fitness
            function of every iteration.

    Returns:
        A go.Figure object fitness vs iteration.

    """
    fig = go.Figure()
    iters = np.arange(1, len(fitness_curve) + 1)
    fig.add_trace(go.Scatter(x=iters, y=fitness_curve, mode='lines'))
    fig.update_layout({
        'xaxis_title': 'Iteration',
        'yaxis_title': 'Fitness function'
    })
    return fig


def box_plot():
    """This is an example of displaying a box plot.

    Still need to figure out how to display summary
    across algorithms and problem sizes.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Box(median=(62.0,),
               q1=(60,),
               q3=(68.75,),
               lowerfence=(58,),
               upperfence=(71,),
               name='first',
               x=(-1,)))
    fig.add_trace(
        go.Box(median=(60.0,),
               q1=(58,),
               q3=(66.75,),
               lowerfence=(54,),
               upperfence=(70,),
               name='second',
               x=(1,)))
    fig.show()


def alg_vs_problem_size_plot(problem_names: List[str],
                             summaries: List[OptimizationSummary], metric: str):

    if len(problem_names) != len(summaries):
        raise ValueError('problem_names and summaries must have the same '
                         'length')

    if not OptimizationSummary._fields:
        raise ValueError('metric is not a valid OptimizationSummary attribute')

    medians = [getattr(s, metric).median for s in summaries]
    q1s = [getattr(s, metric).q1 for s in summaries]
    q3s = [getattr(s, metric).q3 for s in summaries]
    mins = [getattr(s, metric).min for s in summaries]
    maxes = [getattr(s, metric).max for s in summaries]
    x = list(range(len(problem_names)))

    fig = go.Figure()
    fig.add_trace(
        go.Box(median=medians,
               q1=q1s,
               q3=q3s,
               lowerfence=mins,
               upperfence=maxes,
               x=x))

    fig.update_xaxes(
        ticktext=problem_names,
        tickvals=x,
    )

    fig.update_layout({'yaxis_title': metric, 'width': 640, 'height': 480})

    return fig

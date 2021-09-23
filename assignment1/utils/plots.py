import copy
import logging
import time
from typing import List, Any, Iterable, Optional, Union

import numpy as np
import plotly.figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.utils._testing import ignore_warnings

from utils.grid_search import GridSearchResults, ModelType
from utils.models import get_nn, get_svm
from utils.output_grabber import OutputGrabber


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
    fig = _add_scatter_dataset2d(fig,
                                 x_data,
                                 y_data,
                                 scatter_alpha=0.2,
                                 scatter_size=7)
    fig.update_layout({
        'xaxis_title': 'x1',
        'yaxis_title': 'x2',
        'width': 960,
        'height': 540
    })

    if title is not None:
        fig.update_layout({'title': title})

    return fig


def visualize_2d_decision_boundary(model,
                                   x1_max: float,
                                   x2_max: float,
                                   x_data,
                                   y_data,
                                   title: str = None,
                                   scatter_size=2) -> go.Figure:
    """Visualizes the decision boundary of a trained model

    This function only works with model trained with 2D data.

    A meshgrid [0, x1_max] x [0, x2_max] is created.

    Args:
        x_data: Dataset2D features
        y_data: Dataset2D labels
        model: A trained model which implements predict_proba() or predict()
        x1_max: Maximum value of first axis
        x2_max: Maximum value of second axis
        title: If given, will add plot title
        scatter_size: Dataset scatter size

    Returns:
        A plotly figure object
    """

    # Implementation inspired by scikit-learn example
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    x1_max = np.maximum(np.max(x_data[:, 0]), x1_max)
    x2_max = np.maximum(np.max(x_data[:, 1]), x2_max)

    x1_min = np.minimum(np.min(x_data[:, 0]), 0)
    x2_min = np.minimum(np.min(x_data[:, 1]), 0)

    step = min((x1_max - x1_min) / 100, (x2_max - x2_min) / 100)

    x1 = np.arange(x1_min, x1_max, step)
    x2 = np.arange(x2_min, x2_max, step)
    xx1, xx2 = np.meshgrid(x1, x2)

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

    fig = _add_scatter_dataset2d(fig,
                                 x_data,
                                 y_data,
                                 scatter_alpha=0.5,
                                 scatter_size=scatter_size)
    fig.update_layout({'xaxis_title': 'x1', 'yaxis_title': 'x2'})

    if title:
        fig.update_layout({'title': title})

    fig.update_layout({
        'width': 960,
        'height': 540,
    })

    # fig.update_layout(legend={
    #     'yanchor': 'top',
    #     'y': 0.99,
    #     'xanchor': 'left',
    #     'x': 0.01
    # })

    return fig


def _add_scatter_dataset2d(fig: go.Figure, x_data, y_data, scatter_alpha: float,
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
                        y_val: np.ndarray, sizes: List[float],
                        title: str) -> go.Figure:
    """Produces a training curve with respect to training size

    This function is responsible for:
      - Training the given `model` with all training sizes
        in the `sizes` parameter.
      - Generates the corresponding training size curve

    Args:
        model: An untrained model
        x_train: Training set features (n_train, n_features)
        y_train: Training set labels (n_train, )
        x_val: Validation set features (n_train, n_features)
        y_val: Validation set labels (n_train, )
        sizes: List of training sizes as fractions e.g.
            `[0.1, 0.25, 0.5, 0.75, 1.0]`.
        title: Plot title

    Returns:
        A Figure object

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

    lc_results = learning_curve(model,
                                x_concat,
                                y_concat,
                                train_sizes=sizes,
                                cv=cv,
                                shuffle=True,
                                return_times=True)

    train_sizes_ags, train_accs_, val_accs_, fit_times, score_times = lc_results
    fit_times: np.ndarray = fit_times.flatten()
    score_times: np.ndarray = score_times.flatten()

    train_accs = train_accs_.flatten()
    val_accs = val_accs_.flatten()

    fig = _generate_training_size_curves(sizes, train_accs, val_accs, title,
                                         fit_times, score_times)

    return fig


def training_size_curve_nn(params: dict, x_train: np.ndarray,
                           y_train: np.ndarray, x_val: np.ndarray,
                           y_val: np.ndarray, sizes: List[float],
                           title: str) -> go.Figure:
    """Plots a neural network training curve

    The figure has the following properties:
      - x-axis is training sizes
      - y-axis is accuracy
      - Training and validation sets are plotted in the same
        figure.

    Args:
        params: Parameters to instantiate and `fit()` the
            neural network i.e. NeuralNetworkEstimator
        x_train: Training features
        y_train: Training labels
        x_val: Validation features
        y_val: Validation labels
        sizes: Training sizes
        title: Plot title

    Returns:
        A Figure object

    """

    if len(x_train) != len(y_train):
        raise ValueError

    if len(x_val) != len(y_val):
        raise ValueError

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError

    if not all([0 <= s <= 1.0 for s in sizes]):
        raise ValueError

    num_examples = len(x_train)

    x_train, y_train = shuffle(x_train, y_train)

    lengths = [int(s * num_examples) for s in sizes]

    train_accs: List[float] = []
    val_accs: List[float] = []

    train_times_: List[float] = []
    predict_times_: List[float] = []

    for length in lengths:
        x_train_sampled = x_train[:length]
        y_train_sampled = y_train[:length]

        # Fit a neural network
        nn = get_nn(in_features=params['in_features'],
                    num_classes=params['num_classes'],
                    layer_width=params['layer_width'],
                    num_layers=params['num_layers'])

        start = time.time()
        nn.fit(x_train_sampled,
               y_train_sampled,
               x_val,
               y_val,
               learning_rate=params['learning_rate'],
               batch_size=params['batch_size'],
               epochs=params['epochs'],
               verbose=params['verbose'])
        finish = time.time()
        train_times_.append(finish - start)

        y_pred = nn.predict(x_train_sampled)
        train_accs.append(accuracy_score(y_train_sampled, y_pred))

        start = time.time()
        y_pred = nn.predict(x_val)
        val_accs.append(accuracy_score(y_val, y_pred))
        finish = time.time()
        predict_times_.append(finish - start)

    train_times = np.array(train_times_)
    predict_times = np.array(predict_times_)

    fig = _generate_training_size_curves(sizes, train_accs, val_accs, title,
                                         train_times, predict_times)

    return fig


def _generate_training_size_curves(sizes: Iterable[float],
                                   train_accs: Iterable[float],
                                   val_accs: Iterable[float], title: str,
                                   train_times, predict_times):
    """Generates training size curves

    There are two curves:
      1. Accuracy with respect to training sizes
      2. Training/prediction times with respect to training sizes

    Args:
        sizes: Fractions of training sizes i.e. x-axis values
        train_accs: Training accuracy values
        val_accs: Validation accuracy values
        title: Plot title
        train_times: Training times
        predict_times: Prediction times

    Returns:
        A Figure object

    """
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=sizes,
                             y=train_accs,
                             mode='lines',
                             name='train_acc'),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=sizes, y=val_accs, mode='lines', name='val_acc'),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=sizes,
                             y=train_times,
                             mode='lines',
                             name='train_times'),
                  row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=sizes,
                             y=predict_times,
                             mode='lines',
                             name='prediction_times'),
                  row=2,
                  col=1)

    fig.update_xaxes(title_text='Training size')
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Time (s)', row=2, col=1)

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


# Decorator trick from the following Stack Overflow post
# https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
@ignore_warnings(category=ConvergenceWarning)
def svm_training_curve_iteration(best_params: dict, x_train: np.ndarray,
                                 y_train: np.ndarray, x_val: np.ndarray,
                                 y_val: np.ndarray):
    """Generates an SVM training curve

    The curve has the following characteristics:
      1. x-axis is number of iterations
      2. y-axis is accuracy
      3. Train and Val sets are plotted

    Args:
        best_params: Best parameters for the SVM constructor
        x_train: Training features
        y_train: Training labels
        x_val: Validation features
        y_val: Validation labels

    Returns:
        A graph object with a curve described above

    """

    params = copy.deepcopy(best_params)

    def equal(model1: SVC, model2: SVC):
        """Tests equality of two SVC models"""
        cond1 = np.array_equal(model1.intercept_, model2.intercept_)
        cond2 = np.array_equal(model1.support_, model2.support_)
        cond3 = np.array_equal(model1.dual_coef_, model2.dual_coef_)
        return cond1 and cond2 and cond3

    last_model: Optional[SVC] = None
    params['verbose'] = True
    out = OutputGrabber()
    with out:
        model = get_svm(**params)
        model.fit(x_train, y_train)

    # Capture the max_iter from all SVMs
    s: str = out.capturedtext
    split: List[str] = [s_ for s_ in s.split('\n') if '#iter = ' in s_]
    all_iters = [int(s_.split()[-1]) for s_ in split]
    max_iter = np.max(all_iters)
    interval = max(1, max_iter // 100)

    params['verbose'] = False

    iters = []
    train_accs = []
    val_accs = []
    iter_ = 0

    while True:
        iter_ += interval

        logging.info(f'SVM training curve iteration {iter_}')

        params['max_iter'] = iter_
        model = get_svm(**params)
        model.fit(x_train, y_train)

        # Update iters
        iters.append(iter_)

        # Update train_accs
        y_pred = model.predict(x_train)
        train_accs.append(accuracy_score(y_train, y_pred))

        # Update val_accs
        y_pred = model.predict(x_val)
        val_accs.append(accuracy_score(y_val, y_pred))

        # If converges, then break
        if (last_model is not None) and equal(model, last_model):
            break

        last_model = model
        continue

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=train_accs, mode='lines', name='train'))
    fig.add_trace(go.Scatter(x=iters, y=val_accs, mode='lines', name='val'))
    fig.update_layout({'xaxis_title': 'Iterations', 'yaxis_title': 'Accuracy'})

    return fig


def gs_results_validation_curve(gs: GridSearchResults,
                                param_name: str,
                                plot_title,
                                log_scale: bool = True,
                                other_params: Union[str, dict] = 'best'):
    """Generates a validation curve

    The GridSearchResults contains a table with information
    about the train and val accuracy scores. The generated
    plot has `param_name` on the x-axis and the corresponding
    accuracy on the y-axis. The other parameters are derived
    from the best parameters in the `gs` object.

    Args:
        plot_title:
        gs: A GridSearchResults object
        param_name: Parameter name to plot
        log_scale: If True, use log scale x-axis.
        other_params: Keyword arguments for parameters
            other than `param_name`. If 'best' is given,
            they will be inferred from `gs.best_kwargs`.

    Returns:
        A graph object

    """

    if other_params == 'best':
        best_params = gs.best_kwargs
        other_params = {k: v for k, v in best_params.items() if k != param_name}
    else:
        if param_name in other_params.keys():
            raise ValueError

    param_values = []
    val_acc = []
    train_acc = []

    for params, d in gs.table:
        _other_params = {k: v for k, v in params.items() if k != param_name}

        if other_params != _other_params:
            continue

        train_acc.append(d['train_accuracy'])
        val_acc.append(d['val_accuracy'])
        param_values.append(params[param_name])

    sort_indices = np.argsort(param_values)
    sorted_param_values = [param_values[i] for i in sort_indices]
    sorted_val_acc = [val_acc[i] for i in sort_indices]
    sorted_train_acc = [train_acc[i] for i in sort_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=sorted_param_values,
                   y=sorted_train_acc,
                   mode='lines',
                   name='train'))
    fig.add_trace(
        go.Scatter(x=sorted_param_values,
                   y=sorted_val_acc,
                   mode='lines',
                   name='val'))

    fig.update_layout({
        'xaxis_title': param_name,
        'yaxis_title': 'Accuracy',
        'title': plot_title
    })

    if log_scale:
        fig.update_xaxes(type='log')

    return fig


def model_confusion_matrix(cm, labels: List[str], plot_title: str):
    """Generates a model confusion matrix

    Args:
        cm: Confusion matrix. `cm[i, j]` corresponds to
            ground truth `i` and predicted `j`.
        labels: Ordered label strings
        plot_title: Plot title string

    Returns:
        A figure object

    """

    # Code inspired from
    # https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap
    z_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(cm,
                                      labels,
                                      labels,
                                      annotation_text=z_text,
                                      colorscale='blues',
                                      reversescale=True)

    fig['layout']['yaxis']['autorange'] = 'reversed'

    fig.update_layout({
        'xaxis_title': 'predicted',
        'yaxis_title': 'ground truth',
        'title': plot_title
    })

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 20

    return fig


def sample_mnist_dataset(x_data: np.ndarray, y_data: np.ndarray,
                         samples_per_label: int) -> np.ndarray:
    """Picks some samples of the Fashion-MNIST dataset

    A grid of 10 columns will be made, where each column
    represents a class. The number of rows is determined
    by the `samples_per_label` argument.

    Args:
        x_data: Features
        y_data: Labels
        samples_per_label: Number of samples per label

    Returns:
        A numpy array of shape
            `(samples_per_label * 28, 10 * 28)`.

    """
    if len(x_data.shape) != 2:
        raise ValueError

    if x_data.shape[1] != 28 * 28:
        raise ValueError

    if len(x_data) != len(y_data):
        raise ValueError

    labels = list(range(10))

    n_samples = len(x_data)

    columns = []
    for label in labels:
        # Sample `samples_per_label`
        indices = np.random.choice(np.arange(n_samples)[y_data == label],
                                   samples_per_label,
                                   replace=False)
        x_sampled = x_data[indices]

        # Reshape 28 by 28
        x_reshaped = [np.reshape(x, (28, 28)) for x in x_sampled]

        # Concatenate to make a column
        x_concat = np.concatenate(x_reshaped, axis=0)

        # Append `columns`
        columns.append(x_concat)

    x_all = np.concatenate(columns, axis=1)

    return x_all

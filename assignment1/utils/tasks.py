import json
import logging
import os

import joblib
from typing import List, Callable, Dict, Any, Iterable

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import plotly.graph_objects as go
from sklearn.svm import SVC

from utils.grid_search import GridSearchResults, grid_search, grid_search_nn, \
    ModelType
from utils.models import get_decision_tree, get_knn, get_svm, \
    get_boosting
from utils.nnestimator import NeuralNetworkEstimator, training_curves
from utils.plots import training_size_curve, complexity_curve, \
    training_size_curve_nn, svm_training_curve_iteration, \
    gs_results_validation_curve, model_confusion_matrix

OUTPUT_DIRECTORY = 'outputs'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def training_fig_path(model_name: str, dataset_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}' \
           f'_{dataset_name.lower()}_training_size.png'


def complexity_fig_path(model_name: str, dataset_name: str, param_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_complexity.png'


def gs_results_filepath(model_name: str, dataset_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_gs.json'


def model_file_path(model_name: str, dataset_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}.joblib'


def model_state_dict_path(model_name: str, dataset_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_{dataset_name.lower()}.pt'


def loss_fig_path(model_name: str, dataset_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_loss.png'


def acc_fig_path(model_name: str, dataset_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_accuracy.png'


def confusion_matrix_fig_path(model_name: str, dataset_name: str):
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_confusion_matrix.png'


def dt_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
            dataset_name: str, n_jobs: int, retrain: bool):
    """Task for decision tree

    The tasks are:
      1. Grid search to tune `ccp_alpha`
      2. Training size curve
      3. Complexity curve based on `ccp_alpha`


    Args:
        x_test:
        y_test:
        x_train: Training set features
        y_train: Training set labels
        x_val: Validation set features
        y_val: Validation set labels
        train_sizes: List of fractions of training sizes
        dataset_name: Name of the dataset
        n_jobs: Number of jobs
        retrain: If True, the decision tree will be retrained

    Returns:

    """
    constructor_fn = get_decision_tree
    default_params = {'ccp_alpha': 0.001}
    ccp_alphas: List[float] = np.logspace(0, -5, num=10).tolist()
    param_grid = {'ccp_alpha': ccp_alphas}
    param_name = 'ccp_alpha'
    param_range = ccp_alphas
    dataset_name = dataset_name
    model_name = 'DT'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, x_test, y_test, train_sizes, param_name,
                   param_range, dataset_name, model_name, n_jobs, retrain)


def knn_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
             y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
             dataset_name: str, n_jobs: int, retrain: bool):
    constructor_fn = get_knn
    default_params = {'n_neighbors': 9}
    k_values = [2**p + 1 for p in range(6)]
    param_grid = {'n_neighbors': k_values}
    param_name = 'n_neighbors'
    param_range = k_values
    dataset_name = dataset_name
    model_name = 'KNN'

    _task_template(constructor_fn, default_params, param_grid, x_train, y_train,
                   x_val, y_val, x_test, y_test, train_sizes, param_name,
                   param_range, dataset_name, model_name, n_jobs, retrain)


def svm_poly_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                  dataset_name: str, n_jobs: int, retrain: bool):
    constructor_fn = get_svm
    default_params = {'kernel': 'poly'}
    degree_values = [1, 2, 3, 4, 5]
    param_grid = {'C': [0.1, 1.0, 10.0], 'degree': [1, 2, 3, 4, 5]}
    param_name = 'degree'
    param_range = degree_values
    dataset_name = dataset_name
    model_name = 'SVM-Polynomial'

    model: SVC = _task_template(constructor_fn, default_params, param_grid,
                                x_train, y_train, x_val, y_val, x_test, y_test,
                                train_sizes, param_name, param_range,
                                dataset_name, model_name, n_jobs, retrain)

    best_params = _gs_load(model_name, dataset_name)['best_kwargs']

    # Plot accuracy figure
    svm_accuracy_fig_path = acc_fig_path(model_name, dataset_name)
    if retrain or (not os.path.exists(svm_accuracy_fig_path)):
        fig = svm_training_curve_iteration(best_params, x_train, y_train,
                                           x_val, y_val)
        fig.update_layout(
            {'title': f'{model_name} {dataset_name} training curve'})
        fig.write_image(svm_accuracy_fig_path)


def svm_rbf_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                 dataset_name: str, n_jobs: int, retrain: bool):
    constructor_fn = get_svm

    # Equivalent to "scale" option in scikit-learn
    default_gamma = 1 / (x_val.shape[1] * x_train.var())

    default_params = {'kernel': 'rbf'}
    gamma_values_np: np.ndarray = default_gamma * np.logspace(3, -3, num=10)
    gamma_values: List[float] = gamma_values_np.tolist()
    param_grid = {'C': [0.1, 1.0, 10.0, 100], 'gamma': gamma_values}
    param_name = 'gamma'
    param_range = gamma_values
    dataset_name = dataset_name
    model_name = 'SVM-RBF'

    model: SVC = _task_template(constructor_fn, default_params, param_grid,
                                x_train, y_train, x_val, y_val, x_test, y_test,
                                train_sizes, param_name, param_range,
                                dataset_name, model_name, n_jobs, retrain)

    best_params = _gs_load(model_name, dataset_name)['best_kwargs']

    # Plot accuracy figure
    svm_accuracy_fig_path = acc_fig_path(model_name, dataset_name)
    if retrain or (not os.path.exists(svm_accuracy_fig_path)):
        fig = svm_training_curve_iteration(best_params, x_train, y_train,
                                           x_val, y_val)
        fig.update_layout(
            {'title': f'{model_name} {dataset_name} training curve'})
        fig.write_image(svm_accuracy_fig_path)


def boosting_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                  dataset_name: str, n_jobs: int, retrain: bool):
    constructor_fn = get_boosting
    default_params = {'n_estimators': 256, 'ccp_alpha': 0.001}
    param_grid = {'n_estimators': [256]}
    param_name = 'n_estimators'
    param_range = [256]
    dataset_name = dataset_name
    model_name = 'Boosting'

    model: AdaBoostClassifier = _task_template(constructor_fn, default_params,
                                               param_grid, x_train, y_train,
                                               x_val, y_val, x_test, y_test,
                                               train_sizes, param_name,
                                               param_range, dataset_name,
                                               model_name, n_jobs, retrain)

    # Plot validation curve, which is some sort of iteration curve
    train_scores = [s for s in model.staged_score(x_train, y_train)]
    train_iterations = [i for i, _ in enumerate(train_scores)]
    val_scores = [s for s in model.staged_score(x_val, y_val)]
    val_iterations = [i for i, _ in enumerate(val_scores)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_iterations,
                   y=train_scores,
                   mode='lines',
                   name='train'))
    fig.add_trace(
        go.Scatter(x=val_iterations, y=val_scores, mode='lines', name='val'))
    fig.update_layout({
        'xaxis_title': 'Training size',
        'yaxis_title': 'Accuracy'
    })

    fig.update_layout({'title': 'Boosting Validation curve'})

    # Overwrite the complexity figure path
    fig_path = complexity_fig_path(model_name, dataset_name, param_name)
    fig.write_image(fig_path)


def neural_network_task(x_train: np.ndarray, y_train: np.ndarray,
                        x_val: np.ndarray, y_val: np.ndarray, x_test, y_test,
                        train_sizes: List[float], dataset_name: str,
                        n_jobs: int, retrain: bool):
    in_features = x_train.shape[1]

    model_name = 'NN'

    path_to_state_dict = model_state_dict_path(model_name, dataset_name)
    if os.path.exists(path_to_state_dict):
        nn = NeuralNetworkEstimator.from_state_dict(path_to_state_dict)
    else:
        default_params = {
            'in_features': in_features,
            'num_classes': 10,
            'hidden_layers': [24] * 4,
            'learning_rate': 1e-5,
            'batch_size': 128,
            'epochs': 100,
            'verbose': True
        }

        param_grid = {
            'hidden_layers': [[16] * n for n in [2, 4, 6, 8, 10, 12, 14, 16]],
            'learning_rate': [3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
        }
        gs = grid_search_nn(default_params, param_grid, x_train, y_train, x_val,
                            y_val)

        # Save results into JSON
        _gs_results_to_json(gs, model_name, dataset_name)

        # Save the best model
        nn = gs.best_model
        nn.save(path_to_state_dict)

        # Training size curve
        logging.info(f'{dataset_name} - {model_name} - Training size curve')
        training_size_fig_title = f'{model_name} - {dataset_name} ' \
                                  f'Training Size Curve'
        fig = training_size_curve_nn(gs.best_kwargs,
                                     x_train,
                                     y_train,
                                     x_val,
                                     y_val,
                                     train_sizes,
                                     title=training_size_fig_title)
        fig_path = training_fig_path(model_name, dataset_name)
        fig.write_image(fig_path)

        # Complexity curve
        fig_path = complexity_fig_path(model_name, dataset_name,
                                       'learning_rate')
        if not os.path.exists(fig_path):
            fig_learning_rate = gs_results_validation_curve(gs, 'learning_rate')
            fig_learning_rate.write_image(fig_path)

        # fig_hidden_layers = gs_results_validation_curve(gs, 'hidden_layers')

    loss_fig, acc_fig = training_curves(nn.training_log)

    fig_path = loss_fig_path(model_name, dataset_name)
    loss_fig.write_image(fig_path)

    fig_path = acc_fig_path(model_name, dataset_name)
    acc_fig.write_image(fig_path)


def _task_template(constructor_fn: Callable[..., ModelType],
                   default_params: Dict[str, Any], param_grid: Dict[str, Any],
                   x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                   y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                   param_name: str, param_range: Iterable, dataset_name: str,
                   model_name: str, n_jobs: int, retrain: bool):
    """Template function for the tasks for different models

    Args:
        x_test:
        y_test:
        constructor_fn: `constructor_fn(**default_params)` gives a default model
            object
        default_params: `constructor_fn(**default_params)` gives a default model
            object
        param_grid: Parameter grid for grid search
        x_train: Training set features
        y_train: Training set labels
        x_val: Validation set features
        y_val: Validation set labels
        train_sizes: List of fractions of training sizes
        param_name: Parameter name to plot on complexity curve
        param_range: Parameter values to plot on complexity curve
        dataset_name: Dataset name for plot titles and logging purposes
        model_name: Model name for plot titles and logging purposes
        n_jobs: Number of jobs
        retrain: If True, the model will be retrained

    Returns:
        None.

    """

    saved_model_path = model_file_path(model_name, dataset_name)

    if retrain or (not os.path.exists(saved_model_path)):
        # Grid search
        logging.info(f'{dataset_name} - {model_name} - Grid search')
        gs = grid_search(constructor_fn, default_params, param_grid, x_train,
                         y_train, x_val, y_val, n_jobs)

        # Save results into JSON
        _gs_results_to_json(gs, model_name, dataset_name)

        # Save best model
        model = gs.best_model
        joblib.dump(model, saved_model_path)

        # Training size curve
        logging.info(f'{dataset_name} - {model_name} - Training size curve')
        fig = training_size_curve(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            train_sizes,
            title=f'{model_name} - {dataset_name} Training Size Curve',
            n_jobs=n_jobs)
        fig_path = training_fig_path(model_name, dataset_name)
        fig.write_image(fig_path)

        # Complexity curve
        logging.info(f'{dataset_name} - {model_name} - Complexity curve')
        fig = complexity_curve(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            param_name=param_name,
            param_range=param_range,
            title=f'{dataset_name} - {model_name} Complexity Curve',
            log_scale=True,
            n_jobs=n_jobs)

        fig_path = complexity_fig_path(model_name, dataset_name, param_name)
        fig.write_image(fig_path)

    else:
        logging.info(f'{saved_model_path} found. Loading model from disk.')
        model = joblib.load(saved_model_path)

    # Hack! Figure out label strings from the y_test
    num_labels = len(set(y_test))
    label_strings = [f'{n}' for n in range(num_labels)]
    fig = model_confusion_matrix(model, x_test, y_test, label_strings)
    cm_fig_path = confusion_matrix_fig_path(model_name, dataset_name)
    fig.show()
    fig.write_image(cm_fig_path)

    return model


def _gs_results_to_json(gs: GridSearchResults, model_name: str,
                        dataset_name: str):
    gs_dict = {
        'best_accuracy': gs.best_accuracy,
        'best_kwargs': gs.best_kwargs,
        'table': gs.table
    }

    json_path = gs_results_filepath(model_name, dataset_name)
    with open(json_path, 'w') as file:
        json.dump(gs_dict, file, indent=4)


def _gs_load(model_name: str, dataset_name: str):
    json_path = gs_results_filepath(model_name, dataset_name)
    with open(json_path) as file:
        d = json.load(file)

    return d

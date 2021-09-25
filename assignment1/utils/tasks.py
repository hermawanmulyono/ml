from typing import List, Callable, Dict, Any
import json
import logging
import os
import joblib

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import plotly.graph_objects as go

import utils.nnestimator as nnestimator
from utils.grid_search import GridSearchResults, grid_search, grid_search_nn, \
    ModelType, grid_search_boosting
from utils.models import get_decision_tree, get_knn, get_svm, \
    get_boosting, SklearnModel
from utils.output_files import training_fig_path, validation_fig_path, \
    gs_results_filepath, model_file_path, model_state_dict_path, \
    loss_fig_path, acc_fig_path, confusion_matrix_fig_path, test_json_path
from utils.plots import training_size_curve, training_size_curve_nn, \
    svm_training_curve_iteration, gs_results_validation_curve, \
    model_confusion_matrix


def dt_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
            dataset_name: str, dataset_labels):
    """Task for decision tree

    The tasks are:
      1. Grid search to tune `ccp_alpha`
      2. Training size curve
      3. Complexity curve based on `ccp_alpha`

    Args:
        dataset_labels:
        x_test:
        y_test:
        x_train: Training set features
        y_train: Training set labels
        x_val: Validation set features
        y_val: Validation set labels
        train_sizes: List of fractions of training sizes
        dataset_name: Name of the dataset

    Returns:
        Decision tree model

    """
    constructor_fn = get_decision_tree
    default_params = {'ccp_alpha': 0.001, 'splitter': 'random'}
    ccp_alphas: List[float] = np.logspace(0, -5, num=10).tolist()
    splitter_values = ['random', 'best']
    param_grid = {'ccp_alpha': ccp_alphas, 'splitter': splitter_values}
    param_name = 'ccp_alpha'
    dataset_name = dataset_name
    model_name = 'DT'

    model = _train_task(constructor_fn, default_params, param_grid, x_train,
                        y_train, x_val, y_val, train_sizes, param_name,
                        dataset_name, model_name, log_scale=True)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    # For DT, we want to plot the validation curve for the other `splitter`.
    _sync_other_validation_curves(param_name,
                                  dataset_name,
                                  model_name,
                                  'splitter',
                                  splitter_values,
                                  log_scale=True)

    return model


def knn_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
             y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
             dataset_name: str, dataset_labels):
    constructor_fn = get_knn
    default_params = {'n_neighbors': 9, 'weights': 'uniform'}
    k_values = list(range(1, 64))
    weights_values = ['uniform', 'distance']
    param_grid = {'n_neighbors': k_values, 'weights': weights_values}
    param_name = 'n_neighbors'
    dataset_name = dataset_name
    model_name = 'KNN'

    model = _train_task(constructor_fn, default_params, param_grid, x_train,
                        y_train, x_val, y_val, train_sizes, param_name,
                        dataset_name, model_name)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    # For KNN, we want to plot the validation curve for the other `weights`.
    _sync_other_validation_curves(param_name,
                                  dataset_name,
                                  model_name,
                                  'weights',
                                  weights_values,
                                  log_scale=False)

    return model


def svm_poly_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                  dataset_name: str, dataset_labels):
    constructor_fn = get_svm
    default_params = {'kernel': 'poly'}
    param_grid = {'C': [0.1, 1.0, 10.0], 'degree': [1, 2, 3, 4, 5, 6, 7]}
    param_name = 'degree'
    dataset_name = dataset_name
    model_name = 'SVM-Polynomial'

    model: SVC = _train_task(constructor_fn, default_params, param_grid,
                             x_train, y_train, x_val, y_val, train_sizes,
                             param_name, dataset_name, model_name)

    best_params = _gs_load(model_name, dataset_name)['best_kwargs']

    # Plot accuracy training curve
    svm_accuracy_fig_path = acc_fig_path(model_name, dataset_name)
    if not os.path.exists(svm_accuracy_fig_path):
        fig = svm_training_curve_iteration(best_params, x_train, y_train, x_val,
                                           y_val)
        fig.update_layout(
            {'title': f'{model_name} {dataset_name} training curve'})
        fig.write_image(svm_accuracy_fig_path)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def svm_rbf_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                 dataset_name: str, dataset_labels):
    constructor_fn = get_svm

    # Equivalent to "scale" option in scikit-learn
    default_gamma = 1 / (x_val.shape[1] * x_train.var())

    default_params = {'kernel': 'rbf'}
    gamma_values_np: np.ndarray = default_gamma * np.logspace(3, -3, num=10)
    gamma_values: List[float] = gamma_values_np.tolist()
    param_grid = {'C': [0.1, 1.0, 10.0, 100], 'gamma': gamma_values}
    param_name = 'gamma'
    dataset_name = dataset_name
    model_name = 'SVM-RBF'

    model: SVC = _train_task(constructor_fn, default_params, param_grid,
                             x_train, y_train, x_val, y_val, train_sizes,
                             param_name, dataset_name, model_name,
                             log_scale=True)

    best_params = _gs_load(model_name, dataset_name)['best_kwargs']

    # Plot accuracy training curve
    svm_accuracy_fig_path = acc_fig_path(model_name, dataset_name)
    if not os.path.exists(svm_accuracy_fig_path):
        fig = svm_training_curve_iteration(best_params, x_train, y_train, x_val,
                                           y_val)
        fig.update_layout(
            {'title': f'{model_name} {dataset_name} training curve'})
        fig.write_image(svm_accuracy_fig_path)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def boosting_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                  dataset_name: str, dataset_labels):
    constructor_fn = get_boosting
    default_params = {'n_estimators': 256, 'ccp_alpha': 0.001}
    param_grid = {'n_estimators': [512], 'max_depth': [4, 8, 16]}
    param_name = 'n_estimators'
    dataset_name = dataset_name
    model_name = 'Boosting'

    model: AdaBoostClassifier = _train_task(constructor_fn, default_params,
                                            param_grid, x_train, y_train, x_val,
                                            y_val, train_sizes, param_name,
                                            dataset_name, model_name,
                                            grid_search_fn=grid_search_boosting,
                                            log_scale=True)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def neural_network_task(x_train: np.ndarray, y_train: np.ndarray,
                        x_val: np.ndarray, y_val: np.ndarray, x_test, y_test,
                        train_sizes: List[float], dataset_name: str,
                        dataset_labels):
    in_features = x_train.shape[1]

    model_name = 'NN'

    path_to_state_dict = model_state_dict_path(model_name, dataset_name)
    if os.path.exists(path_to_state_dict):
        nn_model = nnestimator.NeuralNetworkEstimator.from_state_dict(
            path_to_state_dict)
    else:
        # Infer number of classes from y_train, y_val, y_test
        classes = set(y_train).union(set(y_val)).union(set(y_test))
        num_classes = len(classes)

        default_params = {
            'in_features': in_features,
            'num_classes': num_classes,
            'nn_size': 4,
            'learning_rate': 1e-5,
            'batch_size': 128,
            'epochs': 500,
            'verbose': True
        }

        param_grid = {
            'nn_size': [2, 4, 8, 16, 32, 64],
            'learning_rate': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
        }
        gs = grid_search_nn(default_params, param_grid, x_train, y_train, x_val,
                            y_val)

        # Save results into JSON
        _gs_results_to_json(gs, model_name, dataset_name)

        # Save the best model
        nn_model = gs.best_model
        nn_model.save(path_to_state_dict)

    # Training size curve
    logging.info(f'{dataset_name} - {model_name} - Training size curve')
    training_size_fig_title = f'{model_name} - {dataset_name} ' \
                              f'Training Size Curve'
    gs = GridSearchResults(**_gs_load(model_name, dataset_name),
                           best_model=nn_model)
    fig = training_size_curve_nn(gs,
                                 x_train,
                                 y_train,
                                 x_val,
                                 y_val,
                                 train_sizes,
                                 title=training_size_fig_title)
    fig_path = training_fig_path(model_name, dataset_name)
    fig.write_image(fig_path)

    # Validation curve
    val_curve_params = [
        'learning_rate', 'nn_size'
    ]
    for param_name in val_curve_params:
        _sync_validation_curves(param_name,
                                dataset_name,
                                model_name,
                                log_scale=True)

    # Training curves
    loss_fig, acc_fig = nnestimator.training_curves(nn_model.training_log)

    fig_path = loss_fig_path(model_name, dataset_name)
    loss_fig.write_image(fig_path)

    fig_path = acc_fig_path(model_name, dataset_name)
    acc_fig.write_image(fig_path)

    # Test task
    _test_task(nn_model, x_test, y_test, model_name, dataset_name,
               dataset_labels)

    return nn_model


def _train_task(constructor_fn: Callable[..., ModelType],
                default_params: Dict[str, Any], param_grid: Dict[str, Any],
                x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                y_val: np.ndarray, train_sizes: List[float], param_name: str,
                dataset_name: str, model_name: str, grid_search_fn=grid_search,
                log_scale=False):
    """Template function for the training tasks for different models

    Args:
        log_scale:
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
        dataset_name: Dataset name for plot titles and logging purposes
        model_name: Model name for plot titles and logging purposes
        log_scale: If True, log scale is used when plotting validation curves

    Returns:
        None.

    """

    saved_model_path = model_file_path(model_name, dataset_name)
    json_path = gs_results_filepath(model_name, dataset_name)

    if not (os.path.exists(saved_model_path) and os.path.exists(json_path)):
        # Grid search
        logging.info(f'{dataset_name} - {model_name} - Grid search')
        gs = grid_search_fn(constructor_fn, default_params, param_grid, x_train,
                            y_train, x_val, y_val)

        # Save results into JSON
        _gs_results_to_json(gs, model_name, dataset_name)

        # Save best model
        model = gs.best_model
        joblib.dump(model, saved_model_path)

    else:
        logging.info(f'{saved_model_path} found. Loading model from disk.')
        model = joblib.load(saved_model_path)

    # Training size curve
    _sync_training_size_curves(model, x_train, y_train, x_val, y_val,
                               train_sizes, dataset_name, model_name)

    # Validation curve
    _sync_validation_curves(param_name, dataset_name, model_name, log_scale)

    return model


def _sync_training_size_curves(model: SklearnModel, x_train: np.ndarray,
                               y_train: np.ndarray, x_val: np.ndarray,
                               y_val: np.ndarray, train_sizes: List[float],
                               dataset_name: str, model_name: str):
    """Synchronizes training size curves for a given model

    To synchronize means:
      - If the training size curves image file is not found,
        it writes one. This involves training the `model`
        multiple times with the `training_sizes`.
      - Otherwise, this function does nothing.

    Args:
        model: A model
        x_train: Training features
        y_train: Training labels
        x_val: Validation features
        y_val: Validation labels
        train_sizes: Training set sizes
        dataset_name: Dataset name
        model_name: Model name

    Returns:
        None. This function writes a file to the file system.

    """
    logging.info(f'{dataset_name} - {model_name} - Training size curve')
    fig_path = training_fig_path(model_name, dataset_name)

    if not os.path.exists(fig_path):
        fig = training_size_curve(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            train_sizes,
            title=f'{model_name} - {dataset_name} Training Size Curve')
        fig.write_image(fig_path)


def _sync_validation_curves(param_name: str,
                            dataset_name: str,
                            model_name: str,
                            log_scale: bool = False):
    """Synchronizes validation curves for a given model

    To synchronize means:
      - If the validation curves image file does not exist,
        create one
      - Otherwise, this function does nothing

    Args:
        param_name: Parameter name to plot
        dataset_name: Name of the dataset
        model_name: Name of the model
        log_scale: If True, log scale is used along x-axis

    Returns:
        None. This function writes a file to the file system.

    """

    logging.info(f'{dataset_name} - {model_name} - Validation curve')

    fig_path = validation_fig_path(model_name, dataset_name, param_name)
    gs = GridSearchResults(**_gs_load(model_name, dataset_name),
                           best_model=None)
    if not os.path.exists(fig_path):
        plot_title = f'{model_name} {dataset_name} Validation Curve'
        fig_learning_rate = gs_results_validation_curve(gs, param_name,
                                                        plot_title, log_scale)
        fig_learning_rate.write_image(fig_path)


def _sync_other_validation_curves(param_name: str,
                                  dataset_name: str,
                                  model_name: str,
                                  other_param_name: str,
                                  other_param_values: list,
                                  log_scale: bool = False):

    if len(other_param_values) != 2:
        raise ValueError

    fig_path = validation_fig_path(model_name + '_other', dataset_name,
                                   param_name)
    gs = GridSearchResults(**_gs_load(model_name, dataset_name),
                           best_model=None)

    if not os.path.exists(fig_path):
        if gs.best_kwargs[other_param_name] == other_param_values[0]:
            val = other_param_values[1]
        else:
            val = other_param_values[0]

        plot_title = f'{model_name} {dataset_name} Validation Curve - ' \
                     f'{other_param_name} {val}'
        other_params = {other_param_name: val}
        fig_learning_rate = gs_results_validation_curve(
            gs,
            param_name,
            plot_title,
            log_scale=log_scale,
            other_params=other_params)
        fig_learning_rate.write_image(fig_path)


def _test_task(model: ModelType, x_test: np.ndarray, y_test: np.ndarray,
               model_name: str, dataset_name: str, dataset_labels: List[str]):
    """The task to test the performance of the final model

    Args:
        model: A trained model object
        x_test: Test features
        y_test: Test labels
        model_name: Name of the model
        dataset_name: Name of the dataset
        dataset_labels: Ordered labels of the dataset e.g.
            ['negative', 'positive'] or ['cat', 'dob', 'bird']

    Returns:
        Test accuracy

    """

    y_pred = model.predict(x_test)
    cm: np.ndarray = confusion_matrix(y_test, y_pred)

    plot_title = f'{model_name} {dataset_name} confusion matrix'
    fig = model_confusion_matrix(cm, dataset_labels, plot_title)
    cm_fig_path = confusion_matrix_fig_path(model_name, dataset_name)
    fig.write_image(cm_fig_path)

    test_accuracy = accuracy_score(y_test, y_pred)

    test_json = {
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': cm.astype(int).tolist()
    }

    test_json_file_path = test_json_path(model_name, dataset_name)
    with open(test_json_file_path, 'w') as file:
        json.dump(test_json, file, indent=4)

    return test_accuracy


def _gs_results_to_json(gs: GridSearchResults, model_name: str,
                        dataset_name: str):
    """Saves grid search results JSON file"""
    gs_dict = {
        'best_accuracy': gs.best_accuracy,
        'best_kwargs': gs.best_kwargs,
        'table': gs.table,
        'best_fit_time': gs.best_fit_time
    }

    json_path = gs_results_filepath(model_name, dataset_name)
    with open(json_path, 'w') as file:
        json.dump(gs_dict, file, indent=4)


def _gs_load(model_name: str, dataset_name: str):
    """Saves grid search results JSON file"""
    json_path = gs_results_filepath(model_name, dataset_name)
    with open(json_path) as file:
        d = json.load(file)

    return d

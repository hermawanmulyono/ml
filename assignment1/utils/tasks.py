from typing import List, Callable, Dict, Any, Iterable
import json
import logging
import os
import joblib

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

import utils.nnestimator as nnestimator
from utils.grid_search import GridSearchResults, grid_search, grid_search_nn, \
    ModelType
from utils.models import get_decision_tree, get_knn, get_svm, \
    get_boosting
from utils.output_files import training_fig_path, validation_fig_path, \
    gs_results_filepath, model_file_path, model_state_dict_path, \
    loss_fig_path, acc_fig_path, confusion_matrix_fig_path, test_json_path
from utils.plots import training_size_curve, training_size_curve_nn, \
    svm_training_curve_iteration, gs_results_validation_curve, \
    model_confusion_matrix


def dt_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
            dataset_name: str, dataset_labels, n_jobs: int, retrain: bool):
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
        n_jobs: Number of jobs
        retrain: If True, the decision tree will be retrained

    Returns:

    """
    constructor_fn = get_decision_tree
    default_params = {'ccp_alpha': 0.001}
    ccp_alphas: List[float] = np.logspace(0, -5, num=10).tolist()
    param_grid = {'ccp_alpha': ccp_alphas}
    param_name = 'ccp_alpha'
    dataset_name = dataset_name
    model_name = 'DT'

    model = _train_task(constructor_fn, default_params, param_grid, x_train,
                        y_train, x_val, y_val, train_sizes, param_name,
                        dataset_name, model_name, n_jobs, retrain)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def knn_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
             y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
             dataset_name: str, dataset_labels, n_jobs: int, retrain: bool):
    constructor_fn = get_knn
    default_params = {'n_neighbors': 9}
    k_values = [2**p + 1 for p in range(6)]
    param_grid = {'n_neighbors': k_values}
    param_name = 'n_neighbors'
    dataset_name = dataset_name
    model_name = 'KNN'

    model = _train_task(constructor_fn, default_params, param_grid, x_train,
                        y_train, x_val, y_val, train_sizes, param_name,
                        dataset_name, model_name, n_jobs, retrain)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def svm_poly_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                  dataset_name: str, dataset_labels, n_jobs: int,
                  retrain: bool):
    constructor_fn = get_svm
    default_params = {'kernel': 'poly'}
    degree_values = [1, 2, 3, 4, 5]
    param_grid = {'C': [0.1, 1.0, 10.0], 'degree': [1, 2, 3, 4, 5]}
    param_name = 'degree'
    dataset_name = dataset_name
    model_name = 'SVM-Polynomial'

    model: SVC = _train_task(constructor_fn, default_params, param_grid,
                             x_train, y_train, x_val, y_val, train_sizes,
                             param_name, dataset_name, model_name, n_jobs,
                             retrain)

    best_params = _gs_load(model_name, dataset_name)['best_kwargs']

    # Plot accuracy training curve
    svm_accuracy_fig_path = acc_fig_path(model_name, dataset_name)
    if retrain or (not os.path.exists(svm_accuracy_fig_path)):
        fig = svm_training_curve_iteration(best_params, x_train, y_train, x_val,
                                           y_val)
        fig.update_layout(
            {'title': f'{model_name} {dataset_name} training curve'})
        fig.write_image(svm_accuracy_fig_path)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def svm_rbf_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                 dataset_name: str, dataset_labels, n_jobs: int, retrain: bool):
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
                             param_name, dataset_name, model_name, n_jobs,
                             retrain)

    best_params = _gs_load(model_name, dataset_name)['best_kwargs']

    # Plot accuracy training curve
    svm_accuracy_fig_path = acc_fig_path(model_name, dataset_name)
    if retrain or (not os.path.exists(svm_accuracy_fig_path)):
        fig = svm_training_curve_iteration(best_params, x_train, y_train, x_val,
                                           y_val)
        fig.update_layout(
            {'title': f'{model_name} {dataset_name} training curve'})
        fig.write_image(svm_accuracy_fig_path)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def boosting_task(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, x_test, y_test, train_sizes: List[float],
                  dataset_name: str, dataset_labels, n_jobs: int,
                  retrain: bool):
    constructor_fn = get_boosting
    default_params = {'n_estimators': 256, 'ccp_alpha': 0.001}
    param_grid = {'n_estimators': [256]}
    param_name = 'n_estimators'
    dataset_name = dataset_name
    model_name = 'Boosting'

    model: AdaBoostClassifier = _train_task(constructor_fn, default_params,
                                            param_grid, x_train, y_train, x_val,
                                            y_val, train_sizes, param_name,
                                            dataset_name, model_name, n_jobs,
                                            retrain)

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
        'xaxis_title': 'Number of weak learners',
        'yaxis_title': 'Accuracy'
    })

    fig.update_layout({'title': 'Boosting Validation curve'})

    # Overwrite the complexity figure path
    fig_path = validation_fig_path(model_name, dataset_name, param_name)
    fig.write_image(fig_path)

    _test_task(model, x_test, y_test, model_name, dataset_name, dataset_labels)

    return model


def neural_network_task(x_train: np.ndarray, y_train: np.ndarray,
                        x_val: np.ndarray, y_val: np.ndarray, x_test, y_test,
                        train_sizes: List[float], dataset_name: str,
                        dataset_labels, n_jobs: int, retrain: bool):
    in_features = x_train.shape[1]

    model_name = 'NN'

    path_to_state_dict = model_state_dict_path(model_name, dataset_name)
    if os.path.exists(path_to_state_dict):
        nn_model = nnestimator.NeuralNetworkEstimator.from_state_dict(
            path_to_state_dict)
    else:
        default_params = {
            'in_features': in_features,
            'num_classes': 10,
            'hidden_layers': [24] * 4,
            'learning_rate': 1e-5,
            'batch_size': 128,
            'epochs': 200,
            'verbose': True
        }

        param_grid = {
            'hidden_layers': [[16] * n for n in [2, 4, 6, 8, 10, 12, 14, 16]],
            'learning_rate': [3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            'batch_size': [32, 64, 128, 256, 512]
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
        fig = training_size_curve_nn(gs.best_kwargs,
                                     x_train,
                                     y_train,
                                     x_val,
                                     y_val,
                                     train_sizes,
                                     title=training_size_fig_title)
        fig_path = training_fig_path(model_name, dataset_name)
        fig.write_image(fig_path)

    # Validation curve
    fig_path = validation_fig_path(model_name, dataset_name, 'learning_rate')
    gs = GridSearchResults(**_gs_load(model_name, dataset_name),
                           best_model=nn_model)
    if not os.path.exists(fig_path):
        plot_title = f'{model_name} {dataset_name} Validation Curve'
        fig_learning_rate = gs_results_validation_curve(gs, 'learning_rate',
                                                        plot_title)
        fig_learning_rate.write_image(fig_path)

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
                dataset_name: str, model_name: str, n_jobs: int, retrain: bool):
    """Template function for the training tasks for different models

    Args:
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

    else:
        logging.info(f'{saved_model_path} found. Loading model from disk.')
        model = joblib.load(saved_model_path)

    # Training size curve
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
            title=f'{model_name} - {dataset_name} Training Size Curve',
            n_jobs=n_jobs)
        fig.write_image(fig_path)

    # Validation curve
    logging.info(f'{dataset_name} - {model_name} - Validation curve')

    fig_path = validation_fig_path(model_name, dataset_name, param_name)
    gs = GridSearchResults(**_gs_load(model_name, dataset_name),
                           best_model=model)
    if not os.path.exists(fig_path):
        plot_title = f'{model_name} {dataset_name} Validation Curve'
        fig_learning_rate = gs_results_validation_curve(gs, param_name,
                                                        plot_title)
        fig_learning_rate.write_image(fig_path)

    return model


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

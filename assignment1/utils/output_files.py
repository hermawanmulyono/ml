"""
This module contains the paths to output files
"""

import os


OUTPUT_DIRECTORY = 'outputs'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def training_fig_path(model_name: str, dataset_name: str):
    """Training curve (accuracy vs training size)"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}' \
           f'_{dataset_name.lower()}_training_size.png'


def validation_fig_path(model_name: str, dataset_name: str, param_name: str):
    """Validation curve (accuracy vs hyperparameter)"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_{param_name}_validation.png'


def gs_results_filepath(model_name: str, dataset_name: str):
    """Grid search results JSON"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_gs.json'


def model_file_path(model_name: str, dataset_name: str):
    """Scikit-Learn model"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}.joblib'


def model_state_dict_path(model_name: str, dataset_name: str):
    """Pytorch model state_dict"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_{dataset_name.lower()}.pt'


def loss_fig_path(model_name: str, dataset_name: str):
    """Training curve (loss vs iteration) """
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_loss.png'


def acc_fig_path(model_name: str, dataset_name: str):
    """Training curve (accuracy vs iteration)"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_accuracy.png'


def confusion_matrix_fig_path(model_name: str, dataset_name: str):
    """Confusion matrix"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_confusion_matrix.png'


def dataset2d_fig_path():
    """Overview of `Dataset2D`"""
    return f'{OUTPUT_DIRECTORY}/Dataset2D.png'


def test_json_path(model_name: str, dataset_name: str):
    """Test JSON which contains test_accuracy and confusion
    matrix"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_test.json'


def decision_boundary_fig_path(model_name: str, dataset_name: str):
    """Decision boundary visualization"""
    return f'{OUTPUT_DIRECTORY}/{model_name.lower()}_' \
           f'{dataset_name.lower()}_decision_boundary.png'


def fashion_mnist_samples_fig_path():
    return f'{OUTPUT_DIRECTORY}/Fashion-MNIST_samples.png'

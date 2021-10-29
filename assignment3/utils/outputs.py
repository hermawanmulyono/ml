"""
This module contains the paths to output files
"""

import os


OUTPUT_DIRECTORY = 'outputs'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def clusterer_joblib(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_clusterer.joblib'


def clustering_score_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_clustering_score.png'


def clustering_visualization_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{alg_name}_clustering_visualization.png'


def clustering_evaluation_json(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{alg_name}_clustering_evaluation.json'


def reduction_alg_joblib(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_reduction_alg.joblib'


def reconstruction_error_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{alg_name}_reconstruction_error.png'


def reduction_json(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_reduction.json'


def kurtosis_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_kurtosis.png'


def feature_importances_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{alg_name}_feature_importances.png'

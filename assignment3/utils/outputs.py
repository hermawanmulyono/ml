"""
This module contains the paths to output files
"""

import os
from typing import Optional

OUTPUT_DIRECTORY = 'outputs'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

############################################################
# Clustering
############################################################


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


############################################################
# Dimensionality reduction
############################################################


def reduction_alg_joblib(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_reduction_alg.joblib'


def reconstruction_error_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{alg_name}_reconstruction_error.png'


def reduction_json(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_reduction.json'


def vector_visualization_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{alg_name}_vector_visualization.png'


def kurtosis_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_kurtosis.png'


def feature_importances_png(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{alg_name}_feature_importances.png'


############################################################
# Reduction -> Clustering -> NN
############################################################


def reduction_and_nn_joblib(dataset_name: str,
                            reduction_alg_name: Optional[str],
                            clustering_alg_name: Optional[str]):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{reduction_alg_name}_{clustering_alg_name}.joblib'


def training_curve(dataset_name: str, reduction_alg_name: Optional[str],
                   clustering_alg_name: Optional[str]):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_' \
           f'{reduction_alg_name}_{clustering_alg_name}_training_curve.png'


def grid_search_json(dataset_name: str, reduction_alg_name: Optional[str],
                     clustering_alg_name: Optional[str]):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{reduction_alg_name}_' \
           f'{clustering_alg_name}_grid_search.json'

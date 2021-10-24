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


def reduction_alg_joblib(dataset_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{dataset_name}_{alg_name}_reduction_alg.joblib'


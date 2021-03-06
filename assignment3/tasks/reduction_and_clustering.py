"""
This module contains the tasks of running:
  1. Dimensionality reduction algorithm
  2. Clustering on the data after dimensionality reduction

There are 4 dimensionality reduction algorithms and 2
clustering algorithms. There are 16 possible combinations.
"""

import numpy as np

from tasks.clustering import ReducedClusteringVisualizationFunc, run_clustering
from tasks.dims_reduction import reduce_pca, reduce_ica, reduce_rp, reduce_dt, \
    VectorVisualizationFunction


def run_reduction_and_clustering(
        dataset_name: str,
        x_data: np.ndarray,
        y_data: np.ndarray,
        clustering_visualization_fn: ReducedClusteringVisualizationFunc,
        vector_visualization_fn: VectorVisualizationFunction,
        n_jobs=1):

    # Assume all dimensionality reduction algorithms have been run
    sync = False

    dims_reduction_steps = [reduce_pca, reduce_ica, reduce_rp, reduce_dt]

    for dims_reduction_step in dims_reduction_steps:
        x_reduced, reduction_alg = dims_reduction_step(dataset_name, x_data,
                                                       y_data, sync, n_jobs,
                                                       vector_visualization_fn)
        postfix = reduction_alg.__class__.__name__
        run_clustering(f'{dataset_name}-{postfix}', x_reduced, y_data,
                       clustering_visualization_fn, n_jobs, is_reduced=True)

import json
import os
from multiprocessing import Pool
from typing import Union, Callable, List, Type

import joblib
import numpy as np
from plotly import graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score
from sklearn.mixture import GaussianMixture

from utils.outputs import clusterer_joblib, clustering_score_png, \
    clustering_visualization_png, clustering_evaluation_json
from utils.plots import simple_line_plot

ClusteringAlg = Union[KMeans, GaussianMixture]
ClusteringVisualizationFunc = Callable[[np.ndarray, np.ndarray, List[str]],
                                       go.Figure]

# A visualization function arguments are:
#   1. x_data: Features of shape (N, n_features)
#   2. y_data: Labels of shape (N, )
#   3. categories: An ordered list of category names of length n_categories
#
# It returns a go.Figure containing the visualization of the data.
VisualizationFunction = Callable[[np.ndarray, np.ndarray, List[str]], go.Figure]


def run_clustering(dataset_name: str, x_data: np.ndarray, y_data: np.ndarray,
                   visualization_fn: ClusteringVisualizationFunc,
                   n_jobs: int = 1):
    clustering_algs = [KMeans, GaussianMixture]

    for alg in clustering_algs:
        clusterer = _sync_clusterer(alg, dataset_name, x_data, n_jobs)
        _sync_clustering_visualization(dataset_name, x_data, clusterer,
                                       visualization_fn)
        _sync_cluster_evaluation(dataset_name, x_data, y_data, clusterer)


def _sync_clusterer(alg: Type, dataset_name: str, x_train: np.ndarray,
                    n_jobs: int):
    """Synchronizes clusterer

    To synchronize means to create when a file does not
    exist yet.

    The following files are synchronized:
      - The serialized clusterer file. This contains the
        best clusterer, which is the one that maximizes
        the clustering score (e.g. silhouette).
      - The clustering score figure (a .png file).

    Args:
        alg: Algorithm class that can be used for
            clustering. Valid examples are sklearn
            KMeans or GaussianMixture
        dataset_name: Dataset name
        x_train: Training data
        n_jobs: Number of Python processes to use

    Returns:
        A clusterer instance. This is the clusterer that
            maximizes the clustering score on `x_train`.

    """
    alg_name = alg.__name__
    clusterer_joblib_path = clusterer_joblib(dataset_name, alg_name)
    clustering_score_plot_path = clustering_score_png(dataset_name, alg_name)

    joblib_exists = os.path.exists(clusterer_joblib_path)
    png_exists = os.path.exists(clustering_score_plot_path)

    if joblib_exists and png_exists:
        best_clusterer = joblib.load(clusterer_joblib_path)

    else:
        # Create a clusterer which maximizes the clustering score.
        n_clusters = list(range(2, 17))

        def args_generator():
            for n_cluster in n_clusters:
                yield alg, n_cluster, x_train

        with Pool(n_jobs) as pool:
            # tuples = [..., (clusterer, score), ...]
            tuples = pool.starmap(_cluster_data, args_generator())

        clusterers, clustering_scores = zip(*tuples)

        # best_index = int(np.argmax(clustering_scores))
        best_index = _best_clustering_index(clustering_scores)
        best_clusterer = clusterers[best_index]

        if not joblib_exists:
            joblib.dump(best_clusterer, clusterer_joblib_path)

        if not png_exists:
            fig = simple_line_plot(n_clusters, clustering_scores, 'n_clusters',
                                   'silhouette')
            fig.write_image(clustering_score_plot_path)

    return best_clusterer


def _best_clustering_index(clustering_scores: List[float]) -> int:
    """Chooses best clustering index

    Try to choose the maximum local maximum. Otherwise,
    choose the one with max score.

    Args:
        clustering_scores: A list of clustering scores. The
            higher, the better.

    Returns:
        Best clustering index

    """
    indices = np.arange(len(clustering_scores))

    x = np.array(clustering_scores)

    positive_grad = np.concatenate([[False], (x[1:] >= x[:-1])])
    negative_grad = np.concatenate([(x[1:] <= x[:-1]), [False]])
    local_max = positive_grad & negative_grad

    if len(local_max) > 0:
        x_local_max = x[local_max]
        i_local_max = indices[local_max]

        # Find index which maximizes x_local_max, but we convert it to the
        # original index
        best_index = i_local_max[np.argmax(x_local_max)]
    else:
        # There's no local maxima. Just return the maximum.
        best_index = int(np.argmax(clustering_scores))

    return int(best_index)


def _cluster_data(alg, n_cluster, x_train):
    """Function for clustering with multiprocessing purposes"""
    clusterer = alg(n_cluster)
    cluster_labels = clusterer.fit_predict(x_train)

    score = silhouette_score(x_train, cluster_labels)

    return clusterer, score


def _sync_clustering_visualization(dataset_name: str, x_train: np.ndarray,
                                   clusterer: ClusteringAlg,
                                   visualization_fn: VisualizationFunction):
    """Synchronizes clustering visualization

    If it does not exist, a new figure will be created.

    Args:
        dataset_name: Dataset name
        x_train: Training data of shape (N, n_dims)
        clusterer: A clustering algorithm that has been
            fitted.
        visualization_fn: A visualization function. See the
            type hint definition for more information.

    Returns:
        None. This function writes data to the file system.

    """
    alg_name = clusterer.__class__.__name__
    png_path = clustering_visualization_png(dataset_name, alg_name)

    if not os.path.exists(png_path):
        pred_labels = clusterer.predict(x_train)
        all_labels = sorted(set(pred_labels))
        categories = [f'cluster_{c}' for c in all_labels]
        fig = visualization_fn(x_train, pred_labels, categories)
        fig.write_image(png_path)


def _sync_cluster_evaluation(dataset_name: str, x_data: np.ndarray,
                             y_data: np.ndarray,
                             clusterer: ClusteringAlg):

    alg_name = clusterer.__class__.__name__
    json_path = clustering_evaluation_json(dataset_name, alg_name)

    if not os.path.exists(json_path):
        y_pred = clusterer.predict(x_data)
        random_index = rand_score(y_data, y_pred)

        d = {'random_index': random_index}

        with open(json_path, 'w') as fs:
            json.dump(d, fs)

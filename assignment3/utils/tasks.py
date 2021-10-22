import os
from typing import Type, Union, Callable, List

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

from utils.outputs import clusterer_joblib, silhouette_png, \
    clustering_visualization_png
from utils.plots import silhouette_plot

ClusteringAlg = Union[KMeans, GaussianMixture]

ClusteringVisualizationFunc = Callable[[np.ndarray, np.ndarray, List[str]],
                                       go.Figure]


def run_clustering(dataset_name: str, x_train: np.ndarray,
                   visualization_fn: ClusteringVisualizationFunc):
    clustering_algs = [KMeans, GaussianMixture]

    for alg in clustering_algs:
        clusterer = sync_clusterer(alg, dataset_name, x_train)
        synchronize_visualization(dataset_name, x_train, clusterer, visualization_fn)


def sync_clusterer(alg: Type, dataset_name: str, x_train: np.ndarray):
    """Synchronizes clusterer

    To synchronize means to create when a file does not
    exist yet.

    The following files are synchronized:
      - The serialized clusterer file. This contains the
        best clusterer, which is the one that maximizes
        the silhouette score.
      - The silhouette score figure (a .png file).

    Args:
        alg: Algorithm class that can be used for
            clustering. Valid examples are sklearn
            KMeans or GaussianMixture
        dataset_name: Dataset name
        x_train: Training data

    Returns:
        A clusterer instance. This is the clusterer that
            maximizes the silhouette score on `x_train`.

    """
    alg_name = alg.__name__
    clusterer_joblib_path = clusterer_joblib(dataset_name, alg_name)
    silhouette_plot_path = silhouette_png(dataset_name, alg_name)

    joblib_exists = os.path.exists(clusterer_joblib_path)
    png_exists = os.path.exists(silhouette_plot_path)

    if joblib_exists and png_exists:
        best_clusterer = joblib.load(clusterer_joblib_path)

    else:
        # Create a clusterer which maximizes the silhouette score.
        n_clusters = list(range(2, 17))
        silhouette_scores = []
        clusterers = []

        for n_cluster in n_clusters:
            clusterer = alg(n_cluster)
            cluster_labels = clusterer.fit_predict(x_train)

            score = silhouette_score(x_train, cluster_labels)
            silhouette_scores.append(score)
            clusterers.append(clusterer)

        best_index = int(np.argmax(silhouette_scores))
        best_clusterer = clusterers[best_index]

        if not joblib_exists:
            joblib.dump(best_clusterer, clusterer_joblib_path)

        if not png_exists:
            fig = silhouette_plot(n_clusters, silhouette_scores)
            fig.write_image(silhouette_plot_path)

    return best_clusterer


def synchronize_visualization(dataset_name: str, x_train: np.ndarray,
                                        clusterer: ClusteringAlg,
                              visualization_fn):
    alg_name = clusterer.__class__.__name__
    png_path = clustering_visualization_png(dataset_name, alg_name)

    if not os.path.exists(png_path):
        pred_labels = clusterer.predict(x_train)
        categories = [f'cluster_{c}' for c in pred_labels]
        fig = visualization_fn(x_train, pred_labels, categories)
        fig.write_image(png_path)


import os
from typing import Type, Union, Callable, List, Optional
from multiprocessing import Pool

import joblib
import numpy as np
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import kurtosis
import plotly.graph_objects as go

from utils.gaussian_rp import GaussianRP
from utils.outputs import clusterer_joblib, clustering_score_png, \
    clustering_visualization_png
from utils.plots import clustering_score_plot, visualize_3d_data

ClusteringAlg = Union[KMeans, GaussianMixture]

ClusteringVisualizationFunc = Callable[[np.ndarray, np.ndarray, List[str]],
                                       go.Figure]


def run_clustering(dataset_name: str,
                   x_train: np.ndarray,
                   visualization_fn: ClusteringVisualizationFunc,
                   n_jobs: int = 1):
    clustering_algs = [KMeans, GaussianMixture]

    for alg in clustering_algs:
        clusterer = sync_clusterer(alg, dataset_name, x_train, n_jobs)
        synchronize_visualization(dataset_name, x_train, clusterer,
                                  visualization_fn)


def sync_clusterer(alg: Type, dataset_name: str, x_train: np.ndarray,
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

        best_index = int(np.argmax(clustering_scores))
        best_clusterer = clusterers[best_index]

        if not joblib_exists:
            joblib.dump(best_clusterer, clusterer_joblib_path)

        if not png_exists:
            fig = clustering_score_plot(n_clusters, clustering_scores)
            fig.write_image(clustering_score_plot_path)

    return best_clusterer


def _cluster_data(alg, n_cluster, x_train):
    """Function for clustering with multiprocessing purposes"""
    clusterer = alg(n_cluster)
    cluster_labels = clusterer.fit_predict(x_train)

    score = calinski_harabasz_score(x_train, cluster_labels)

    return clusterer, score


def synchronize_visualization(dataset_name: str, x_train: np.ndarray,
                              clusterer: ClusteringAlg, visualization_fn):
    alg_name = clusterer.__class__.__name__
    png_path = clustering_visualization_png(dataset_name, alg_name)

    if not os.path.exists(png_path):
        pred_labels = clusterer.predict(x_train)
        categories = [f'cluster_{c}' for c in pred_labels]
        fig = visualization_fn(x_train, pred_labels, categories)
        fig.write_image(png_path)


def run_dim_reduction(x_data: np.ndarray, y_data: np.ndarray, sync=False):
    alg_names = ['pca', 'ica', 'rp', 'forward']
    funcs = []

    visualize_3d_data(x_data, y_data, ['0', '1']).show()

    for n_dims in range(1, 4):
        _reduce_rp(x_data, y_data, n_dims)


def _reduce_pca(x_data: np.ndarray, y_data: np.ndarray, dataset_name: str,
                n_dims: int):
    """Reduces the `x_data` to `n_dims` dimensions.

    This function uses the PCA algorithm.

    Args:
        x_data: An array of shape `(N, original_dims)`
        y_data: The corresponding labels of shape `(N, )`
        n_dims: Target dimension

    Returns:
        `x_reduced`, which is an array of shape `(N, n_dims)`

    """
    pca = PCA(n_dims)
    x_reduced = pca.fit_transform(x_data)
    print(pca.explained_variance_)

    return x_reduced, pca


def _reduce_ica(x_data: np.ndarray, y_data: np.ndarray, n_dims: int):
    ica = FastICA(n_dims)
    x_reduced = ica.fit_transform(x_data)

    w = ica.components_
    s_t = np.dot(w, x_data.T)
    s = s_t.T  # Sources of shape (N, n_dims)
    print(s[:5])
    k = kurtosis(s)
    print(k)
    k_ave = np.mean(k)

    print(k_ave)

    # Need to choose optimal N

    mixing = ica.mixing_

    categories = sorted(set(y_data))
    fig = visualize_3d_data(x_data, y_data, [f'{c}' for c in categories])

    x_mean = x_data.mean(axis=0)

    for vector in mixing.T:
        vector = vector / np.linalg.norm(vector) * 5
        x_arrow = np.array([0, vector[0]]) + x_mean[0]
        y_arrow = np.array([0, vector[1]]) + x_mean[1]
        z_arrow = np.array([0, vector[2]]) + x_mean[2]
        fig.add_trace(go.Scatter3d(x=x_arrow, y=y_arrow, z=z_arrow,
                                   mode='lines'))

    fig.show()

    if n_dims == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.flatten(), y=np.zeros((len(s),)),
                                 mode='markers',
                                 marker={'opacity': 0.5}))
        fig.show()
    elif n_dims == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s[:, 0], y=s[:, 1], mode='markers',
                                 marker={'opacity': 0.5}))
        fig.show()
    elif n_dims == 3:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2],
                                   mode='markers', marker={'opacity': 0.5}))
        fig.show()

    return x_reduced, ica


def _reduce_rp(x_data: np.ndarray, y_data: np.ndarray, n_dims: int):
    rp = GaussianRP(n_dims)
    x_reduced = rp.fit_transform(x_data)

    components = rp.components_

    x_proj = np.dot(x_data, components.T)
    x_rec = np.dot(x_proj, components)

    categories = sorted(set(y_data))
    fig = visualize_3d_data(x_rec, y_data, [f'{c}' for c in categories])
    fig.show()

    return x_reduced, rp


def _reduce_forward(x_data: np.ndarray, y_data: np.ndarray, n_dims: int):
    pass

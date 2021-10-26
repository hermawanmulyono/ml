import json
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

from utils.data import check_input
from utils.gaussian_rp import GaussianRP
from utils.outputs import clusterer_joblib, clustering_score_png, \
    clustering_visualization_png, reduction_alg_joblib, \
    reconstruction_error_png, reduction_json
from utils.plots import simple_line_plot, visualize_3d_data

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


# A visualization function arguments are:
#   1. x_data: Features of shape (N, n_features)
#   2. y_data: Labels of shape (N, )
#   3. categories: An ordered list of category names of length n_categories
#
# It returns a go.Figure containing the visualization of the data.
VisualizationFunction = Callable[[np.ndarray, np.ndarray, List[str]], go.Figure]


def synchronize_visualization(dataset_name: str, x_train: np.ndarray,
                              clusterer: ClusteringAlg,
                              visualization_fn: VisualizationFunction):
    """

    Args:
        dataset_name:
        x_train:
        clusterer: A clustering algorithm that has been
            fitted.
        visualization_fn: A visualization function. See the
            type hint definition for more information.

    Returns:

    """
    alg_name = clusterer.__class__.__name__
    png_path = clustering_visualization_png(dataset_name, alg_name)

    if not os.path.exists(png_path):
        pred_labels = clusterer.predict(x_train)
        all_labels = sorted(set(pred_labels))
        categories = [f'cluster_{c}' for c in all_labels]
        fig = visualization_fn(x_train, pred_labels, categories)
        fig.write_image(png_path)


def run_dim_reduction(dataset_name: str,
                      x_data: np.ndarray,
                      y_data: np.ndarray,
                      visualization_fn: VisualizationFunction,
                      sync=False):

    _reduce_pca(dataset_name, x_data, y_data, sync)


def _reduce_pca(dataset_name: str, x_data: np.ndarray, y_data: np.ndarray,
                sync: bool):
    """Reduces the `x_data` to `n_dims` dimensions.

    This function uses the PCA algorithm.

    Args:
        dataset_name: Dataset name
        x_data: An array of shape `(N, original_dims)`
        y_data: The corresponding labels of shape `(N, )`.
            This variable is unused.
        sync: If True, synchronization is enabled. This
            function may run PCA from scratch. This
            includes synchronizing necessary plots. If
            `False`, it expects that there is an existing
            serialized PCA object.

    Returns:
        `x_reduced`, which is an array of shape `(N, n_dims)`

    """

    check_input(x_data)

    alg_name = PCA.__name__

    joblib_path = reduction_alg_joblib(dataset_name, alg_name)
    rec_error_path = reconstruction_error_png(dataset_name, alg_name)
    json_path = reduction_json(dataset_name, alg_name)
    files_exist = os.path.exists(joblib_path) and os.path.exists(
        rec_error_path) and os.path.exists(json_path)

    if (not sync) and (not files_exist):
        raise FileNotFoundError(
            f'{joblib_path} or {rec_error_path} does not exist')

    if not files_exist:
        # Run PCA
        n_features = x_data.shape[1]

        pcas = []
        errors = []

        n_dims_list = list(range(1, n_features + 1))

        for n_dims in n_dims_list:
            pca = PCA(n_dims)
            pca.fit(x_data)
            print(pca.explained_variance_)
            pcas.append(pca)

            x_rec = reconstruct_pca(pca, x_data)
            error = reconstruction_error(x_data, x_rec)
            errors.append(error)

        # Store PCA joblib. Just pick 2nd dimension.
        assert len(pcas) >= 2
        joblib.dump(pcas[1], joblib_path)

        # Save reconstruction error plot
        fig = simple_line_plot(n_dims_list, errors, 'n_components',
                               'reconstruction_error')
        fig.write_image(rec_error_path)

        # Save raw data as JSON
        d = {'reconstruction_error': errors,
             'explained_variance': pcas[-1].explained_variance_}
        with open(json_path, 'w') as fstream:
            json.dump(d, fstream)

    dim_red_alg: PCA = joblib.load(joblib_path)
    x_reduced = dim_red_alg.transform(x_data)

    return x_reduced, dim_red_alg


def reconstruct_pca(pca: PCA, X: np.ndarray):
    x_proj = pca.transform(X)
    x_rec = np.dot(x_proj, pca.components_) + pca.mean_
    return x_rec


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
        fig.add_trace(
            go.Scatter3d(x=x_arrow, y=y_arrow, z=z_arrow, mode='lines'))

    fig.show()

    if n_dims == 1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=s.flatten(),
                       y=np.zeros((len(s),)),
                       mode='markers',
                       marker={'opacity': 0.5}))
        fig.show()
    elif n_dims == 2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=s[:, 0],
                       y=s[:, 1],
                       mode='markers',
                       marker={'opacity': 0.5}))
        fig.show()
    elif n_dims == 3:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(x=s[:, 0],
                         y=s[:, 1],
                         z=s[:, 2],
                         mode='markers',
                         marker={'opacity': 0.5}))
        fig.show()

    return x_reduced, ica


def _reduce_rp(x_data: np.ndarray, y_data: np.ndarray, n_dims: int):
    rp = GaussianRP(n_dims)
    x_reduced = rp.fit_transform(x_data)

    x_rec = rp.reconstruct(x_data)

    error = reconstruction_error(x_data, x_rec)
    print(f'error {error}')

    categories = sorted(set(y_data))
    fig = visualize_3d_data(x_rec, y_data, [f'{c}' for c in categories])
    fig.show()

    return x_reduced, rp


def reconstruction_error(x_data: np.ndarray, x_rec: np.ndarray):
    delta = np.linalg.norm(x_data - x_rec, axis=1)
    error = np.mean(np.power(delta, 2))
    return error


def _reduce_forward(x_data: np.ndarray, y_data: np.ndarray, n_dims: int):
    pass

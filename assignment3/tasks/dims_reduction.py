import json
import logging
import os
from multiprocessing import Pool

import joblib
import numpy as np
from plotly import graph_objects as go
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA

from utils.data import check_input
from utils.dtfilter import DTFilter
from utils.gaussian_rp import GaussianRP
from utils.outputs import reduction_alg_joblib, reconstruction_error_png, \
    reduction_json, kurtosis_png, clustering_visualization_png, \
    feature_importances_png
from utils.plots import simple_line_plot, visualize_3d_data, \
    feature_importance_chart


def run_dim_reduction(dataset_name: str,
                      x_data: np.ndarray,
                      y_data: np.ndarray,
                      sync=False,
                      n_jobs=1):

    reduce_pca(dataset_name, x_data, y_data, sync, n_jobs)
    reduce_ica(dataset_name, x_data, y_data, sync, n_jobs)
    reduce_rp(dataset_name, x_data, y_data, sync, n_jobs)
    reduce_dt(dataset_name, x_data, y_data, sync, n_jobs)


def reduce_pca(dataset_name: str, x_data: np.ndarray, y_data: np.ndarray,
               sync: bool, n_jobs):
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
        n_jobs: Number of Python processes to use in
            parallel.

    Returns:
        `(x_reduced, pca)` tuple. `x_reduced` is the
            transformed `x_data`, while `pca` is the
            corresponding PCA object.

    """

    logging.info(f'PCA - {dataset_name}')

    check_input(x_data)

    alg_name = PCA.__name__

    joblib_path = reduction_alg_joblib(dataset_name, alg_name)
    rec_error_path = reconstruction_error_png(dataset_name, alg_name)
    json_path = reduction_json(dataset_name, alg_name)
    files_exist = os.path.exists(joblib_path) and os.path.exists(
        rec_error_path) and os.path.exists(json_path)

    if (not sync) and (not files_exist):
        raise FileNotFoundError(
            f'{joblib_path} or {rec_error_path} or {json_path} does not exist')

    if not files_exist:
        # Run PCA
        n_features = x_data.shape[1]

        n_dims_list = list(range(1, min(64, n_features) + 1))

        def args_generator():
            for n_dims in n_dims_list:
                yield x_data, n_dims

        with Pool(n_jobs) as pool:
            tuples = pool.starmap(_fit_pca, args_generator())

        pcas, errors = zip(*tuples)

        # Store PCA joblib. Just pick 2nd dimension.
        assert len(pcas) >= 2
        chosen_index = 1
        joblib.dump(pcas[chosen_index], joblib_path)

        # Save reconstruction error plot
        fig = simple_line_plot(n_dims_list, errors, 'n_components',
                               'reconstruction_error')
        fig.write_image(rec_error_path)

        # Save raw data as JSON
        explained_variance = pcas[-1].explained_variance_.tolist()
        d = {
            'reconstruction_error': errors[chosen_index],
            'explained_variance': explained_variance
        }

        with open(json_path, 'w') as fstream:
            json.dump(d, fstream, indent=4)

    dim_red_alg: PCA = joblib.load(joblib_path)
    x_reduced = dim_red_alg.transform(x_data)

    return x_reduced, dim_red_alg


def _fit_pca(x_data, n_dims):
    """Fits a PCA model

    This function can be parallelized with multiprocessing.

    Args:
        x_data: (N, n_features) feature array
        n_dims: Number of PCA dimensions

    Returns:
        (pca, reconstruction_error) tuple

    """
    print(f'Running ICA {n_dims} dimensions')

    pca = PCA(n_dims)
    pca.fit(x_data)

    x_rec = _reconstruct_pca(pca, x_data)
    reconstruction_error = _reconstruction_error(x_data, x_rec)

    return pca, reconstruction_error


def _reconstruct_pca(pca: PCA, X: np.ndarray):
    x_proj = pca.transform(X)
    x_rec = np.dot(x_proj, pca.components_) + pca.mean_
    return x_rec


def reduce_ica(dataset_name: str, x_data: np.ndarray, y_data: np.ndarray,
               sync: bool, n_jobs: int):
    logging.info(f'ICA - {dataset_name}')

    check_input(x_data)

    alg_name = FastICA.__name__

    joblib_path = reduction_alg_joblib(dataset_name, alg_name)
    kurtosis_path = kurtosis_png(dataset_name, alg_name)
    json_path = reduction_json(dataset_name, alg_name)
    files_exist = os.path.exists(joblib_path) and os.path.exists(
        kurtosis_path) and os.path.exists(json_path)

    if (not sync) and (not files_exist):
        raise FileNotFoundError(
            f'{joblib_path} or {kurtosis_path} or {json_path} does not exist')

    if files_exist:
        ica: FastICA = joblib.load(joblib_path)
    else:
        # Run ICA
        n_features = x_data.shape[1]

        n_dims_list = list(range(1, min(64, n_features) + 1))

        def args_generator():
            for _n_dims in n_dims_list:
                yield x_data, _n_dims

        with Pool(n_jobs) as pool:
            tuples = pool.starmap(_fit_ica, args_generator())

        icas, mean_abs_kurtosis = zip(*tuples)

        # Find the n_dims with highest kurtosis
        best_index = np.argmax(mean_abs_kurtosis)
        ica = icas[best_index]
        n_dims = n_dims_list[best_index]

        # Store ICA joblib
        joblib.dump(ica, joblib_path)

        # Save kurtosis plot
        fig = simple_line_plot(n_dims_list, mean_abs_kurtosis, 'n_components',
                               'kurtosis')
        fig.write_image(kurtosis_path)

        # Save raw data as JSON
        d = {'n_dims': n_dims, 'abs_kurtosis': mean_abs_kurtosis}

        with open(json_path, 'w') as fstream:
            json.dump(d, fstream, indent=4)

        # Visualize the independent components for Dataset3D
        if x_data.shape[1] == 3:
            mixing = ica.mixing_

            categories = sorted(set(y_data))
            fig = visualize_3d_data(x_data, y_data,
                                    [f'{c}' for c in categories])

            x_mean = x_data.mean(axis=0)

            for vector in mixing.T:
                vector = vector / np.linalg.norm(vector) * 5
                x_arrow = np.array([0, vector[0]]) + x_mean[0]
                y_arrow = np.array([0, vector[1]]) + x_mean[1]
                z_arrow = np.array([0, vector[2]]) + x_mean[2]
                fig.add_trace(
                    go.Scatter3d(x=x_arrow, y=y_arrow, z=z_arrow, mode='lines'))

            png_path = clustering_visualization_png(dataset_name, alg_name)
            fig.write_image(png_path)

    x_reduced = ica.transform(x_data)

    return x_reduced, ica


def _fit_ica(x_data: np.ndarray, n_dims: int):
    """Fits an ICA model

    This function can be parallelized with multiprocessing.

    Args:
        x_data: (N, n_features) feature array
        n_dims: Number of ICA dimensions

    Returns:
        (ica, mean_abs_kurtosis) tuple

    """
    print(f'Running ICA {n_dims} dimensions')

    ica = FastICA(n_dims, max_iter=1000)
    ica.fit(x_data)

    # Find kurtosis of the source vectors
    w = ica.components_
    s_t = np.dot(w, x_data.T)
    s = s_t.T  # Sources of shape (N, n_dims)
    k = kurtosis(s)
    assert len(k) == n_dims, \
        f'Expecting kurtosis of length {n_dims}, got {len(k)}'

    mean_abs_kurtosis = float(np.mean(np.abs(k)))

    return ica, mean_abs_kurtosis


def reduce_rp(dataset_name: str, x_data: np.ndarray, y_data: np.ndarray,
              sync: bool, n_jobs):
    logging.info(f'Random Projection - {dataset_name}')

    check_input(x_data)

    alg_name = GaussianRP.__name__

    joblib_path = reduction_alg_joblib(dataset_name, alg_name)
    rec_error_path = reconstruction_error_png(dataset_name, alg_name)
    json_path = reduction_json(dataset_name, alg_name)

    files_exist = os.path.exists(joblib_path) and os.path.exists(
        rec_error_path) and os.path.exists(json_path)

    if (not sync) and (not files_exist):
        raise FileNotFoundError(
            f'{joblib_path} or {rec_error_path} or {json_path} does not exist')

    if not files_exist:
        # Run GaussianRP
        n_features = x_data.shape[1]

        n_dims_list = list(range(1, n_features + 1))

        def args_generator():
            for n_dims in n_dims_list:
                yield x_data, n_dims

        with Pool(n_jobs) as pool:
            tuples = pool.starmap(_fit_rp, args_generator())

        rps, errors = zip(*tuples)

        # Store RP joblib. Just pick the 2nd dimension.
        assert len(rps) >= 2
        chosen_index = 1
        joblib.dump(rps[chosen_index], joblib_path)

        # Save reconstruction error plot
        fig = simple_line_plot(n_dims_list, errors, 'n_components',
                               'reconstruction_error')
        fig.write_image(rec_error_path)

        # Save raw data as JSON
        d = {
            'reconstruction_error': errors[chosen_index]
        }

        with open(json_path, 'w') as fstream:
            json.dump(d, fstream, indent=4)

    dim_red_alg: GaussianRP = joblib.load(joblib_path)
    x_reduced = dim_red_alg.transform(x_data)

    return x_reduced, dim_red_alg


def _fit_rp(x_data, n_dims):
    """Fits a random projection model

    This function can be parallelized with multiprocessing.

    Args:
        x_data: (N, n_features) feature array
        n_dims: Number of random projection vectors

    Returns:
        (rp, reconstruction_error) tuple

    """
    print(f'Running RP {n_dims} dimensions')

    rp = GaussianRP(n_dims)
    rp.fit(x_data)

    x_rec = rp.reconstruct(x_data)
    reconstruction_error = _reconstruction_error(x_data, x_rec)

    return rp, reconstruction_error


def _reconstruction_error(x_data: np.ndarray, x_rec: np.ndarray):

    abs_data = np.linalg.norm(x_data, axis=1)
    abs_rec = np.linalg.norm(x_rec, axis=1)

    if not np.all(abs_data >= abs_rec - 1E-5):
        raise ValueError('Reconstructed vector must be smaller than the '
                         'original')

    delta = np.linalg.norm(x_data - x_rec, axis=1) / abs_data
    error = np.mean(np.power(delta, 2))
    return error


def reduce_dt(dataset_name: str, x_data: np.ndarray, y_data: np.ndarray,
              sync: bool, n_jobs):

    logging.info(f'DT - {dataset_name}')

    check_input(x_data)

    alg_name = DTFilter.__name__

    joblib_path = reduction_alg_joblib(dataset_name, alg_name)
    bar_chart_path = feature_importances_png(dataset_name, alg_name)
    json_path = reduction_json(dataset_name, alg_name)

    files_exist = os.path.exists(joblib_path) and os.path.exists(
        bar_chart_path) and os.path.exists(json_path)

    if (not sync) and (not files_exist):
        raise FileNotFoundError(f'{joblib_path} or {bar_chart_path} or '
                                f'{json_path} does not exist')

    if not files_exist:
        dt_filter = DTFilter()
        dt_filter.fit(x_data, y_data)

        # Save DTFilter joblib
        joblib.dump(dt_filter, joblib_path)

        # Save bar chart
        importances = dt_filter.dt.feature_importances_
        fig = feature_importance_chart(importances)
        fig.write_image(bar_chart_path)

        # Save JSON file
        d = {'n_dims': len(dt_filter.selected_features),
             'selected_features': dt_filter.selected_features}

        with open(json_path, 'w') as fstream:
            json.dump(d, fstream, indent=4)

    dt_filter = joblib.load(joblib_path)
    x_transformed = dt_filter.transform(x_data)

    return x_transformed, dt_filter
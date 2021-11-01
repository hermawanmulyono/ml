import json
import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

from tasks.clustering import ClusteringAlg, _get_clusterer
from tasks.dims_reduction import reduce_pca, reduce_ica, reduce_rp, reduce_dt, \
    VectorVisualizationFunction, ReductionAlgorithm
from utils.grid_search import grid_search_nn
from utils.nnestimator import NeuralNetworkEstimator, training_curves, one_hot
from utils.outputs import reduction_clustering_nn_joblib, training_curve, \
    grid_search_json

HIDDEN_LAYERS = [16] * 4


class ReductionClusteringNN:

    def __init__(self,
                 reduction_alg: ReductionAlgorithm = None,
                 clustering_alg: ClusteringAlg = None,
                 add_to_reduced_features: bool = False):
        """Constructs a ReductionClusteringNN object

        Args:
            reduction_alg:
            clustering_alg:
            add_to_reduced_features: Only used when
                `clustering_alg` is not `None`. If `False`,
                the clustering features are added to the
                original 
        """

        self.reduction_alg = reduction_alg
        self.clustering_alg = clustering_alg

        self.add_to_reduced_features = add_to_reduced_features

        # These will be initialized in `fit()`
        self.n_classes: Optional[int] = None
        self.nn: Optional[NeuralNetworkEstimator] = None
        self.grid_search_table: Optional[dict] = None

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray = None,
            y_val: np.ndarray = None):

        # Infer n_classes from y_train and y_val
        max1 = np.max(y_train)
        max2 = np.max(y_val)

        num_classes = int(np.max([max1, max2]) + 1)
        self.n_classes = num_classes

        x_train_red = self.transform(x_train)
        x_val_red = self.transform(x_val)
        in_features = int(x_train_red.shape[1])

        param_grid = {
            'in_features': [in_features],  # Constant
            'num_classes': [num_classes],  # Constant
            'nn_size': [4],
            'learning_rate': [1e-6, 1e-5],
            'batch_size': [min(len(x_train), 1024)],
            'epochs': [200],
            'verbose': [True]
        }

        gs = grid_search_nn(param_grid, x_train_red, y_train, x_val_red, y_val)

        self.nn = gs.best_model
        self.grid_search_table = {
            'best_accuracy': gs.best_accuracy,
            'best_kwargs': gs.best_kwargs,
            'table': gs.table,
            'best_fit_time': gs.best_fit_time
        }

    def predict(self, X: np.ndarray):
        x_reduced = self.transform(X)
        predicted = self.nn.predict(x_reduced)
        return predicted

    def transform(self, X: np.ndarray):
        """Transforms feature matrix

        There are two steps:
          - Dimensionality reduction
          - Clustering features are added

        Depending on the arguments given during
        construction, either of these steps can be omitted.

        Args:
            X: Feature matrix of shape `(N, n_features)`

        Returns:
            The transformed feature matrix of shape
                `(N, n_appended_features)`

        """
        # Dimensionality reduction
        if self.reduction_alg is not None:
            x_reduced = self.reduction_alg.transform(X)
        else:
            x_reduced = X.copy()

        # Clustering
        if self.clustering_alg is None:
            x_concat = x_reduced.copy()
        else:
            clust_labels = self.clustering_alg.predict(x_reduced)

            try:
                n_clusters = self.clustering_alg.n_clusters
            except AttributeError:
                n_clusters = self.clustering_alg.n_components

            oh_features = one_hot(clust_labels, n_clusters)

            if self.add_to_reduced_features:
                x_concat = np.concatenate([x_reduced, oh_features], axis=1)
            else:
                x_concat = np.concatenate([X, oh_features], axis=1)

        return x_concat


def run_reduction_and_nn(dataset_name: str,
                         x_train: np.ndarray,
                         y_train: np.ndarray,
                         x_val: np.ndarray,
                         y_val: np.ndarray,
                         n_jobs=1):
    """Applies dimensionality reduction and trains NN

    There are two steps which will be repeated 4 times for
    different dimensionality reduction algorithms:

      - Apply dimensionality reduction to obtain
        `x_reduced`
      - Train NN with `x_reduced`

    The trained reduction-NN are stored in an object
    encapsulating the reduction algorithm and NN.

    Returns:
        None.
    """
    logging.info(f'run_reduction_and_nn() - {dataset_name}')

    # Assume all dimensionality reduction algorithms have been run
    sync = False

    dims_reduction_steps = [reduce_pca, reduce_ica, reduce_rp, reduce_dt]

    for dims_reduction_step in dims_reduction_steps:

        x_reduced, reduction_alg = dims_reduction_step(dataset_name, x_train,
                                                       y_train, sync, n_jobs,
                                                       None)
        _sync_reduction_clustering_nn(dataset_name, x_train, y_train, x_val,
                                      y_val, reduction_alg, None)


def _sync_reduction_clustering_nn(dataset_name: str,
                                  x_train: np.ndarray,
                                  y_train: np.ndarray,
                                  x_val: np.ndarray,
                                  y_val: np.ndarray,
                                  reduction_alg,
                                  clustering_alg,
                                  add_to_reduced_features: bool = False):
    """Synchronizes dimensionality reduction and NN outputs

    The following files are synchronized:
      - The serialized ReductionClusteringNN object
      - The corresponding training curve i.e. acc vs iter

    Args:
        dataset_name: Dataset name
        x_train: Training features of shape
            `(n_train, n_features)`
        y_train: Training labels of shape `(n_train, )`
        x_val: Validation features of shape
            `(n_val, n_features)`
        y_val: Validation labels of shape `(n_val, )`
        reduction_alg: An optional reduction algorithm
            object that implements `.transform()` method.
        clustering_alg: An optional clustering algorithm
            that implements `.transform()` method.
        add_to_reduced_features: Only used when
            `clustering_alg` is not `None`. If `False`,
            the clustering features are added to the
            original

    Returns:
        A ReductionAndNN object. In addition, this function
            may write files to system.

    """
    reduction_alg_name = reduction_alg.__class__.__name__
    clustering_alg_name = clustering_alg.__class__.__name__

    logging.info(f'_sync_reduction_clustering_nn() - {dataset_name} - '
                 f'{reduction_alg_name} - {clustering_alg_name}')

    joblib_path = reduction_clustering_nn_joblib(dataset_name, reduction_alg_name,
                                                 clustering_alg_name)
    training_curve_path = training_curve(dataset_name, reduction_alg_name,
                                         clustering_alg_name)
    gs_json_path = grid_search_json(dataset_name, reduction_alg_name,
                                    clustering_alg_name)

    files_exist = os.path.exists(joblib_path) and os.path.exists(
        training_curve_path) and os.path.exists(gs_json_path)

    if not files_exist:
        # ReductionClusteringNN object
        red_nn = ReductionClusteringNN(reduction_alg, clustering_alg,
                                       add_to_reduced_features)
        red_nn.fit(x_train, y_train, x_val, y_val)
        joblib.dump(red_nn, joblib_path)

        # Training curve
        loss_fig, acc_fig = training_curves(red_nn.nn.training_log)
        acc_fig.write_image(training_curve_path)

        # Grid search JSON
        with open(gs_json_path, 'w') as fs:
            json.dump(red_nn.grid_search_table, fs, indent=4)

    red_nn = joblib.load(joblib_path)
    train_acc = accuracy_score(y_train, red_nn.predict(x_train))
    val_acc = accuracy_score(y_val, red_nn.predict(x_val))

    print(f'{reduction_alg_name} {clustering_alg_name} train_acc {train_acc}, '
          f'val_acc {val_acc}')

    return red_nn


def run_reduction_clustering_nn(dataset_name: str,
                                x_train: np.ndarray,
                                y_train: np.ndarray,
                                x_val: np.ndarray,
                                y_val: np.ndarray,
                                n_jobs=1):
    logging.info(f'run_reduction_clustering_nn() - {dataset_name}')

    # Assume all dimensionality reduction algorithms have been run
    sync = False

    dims_reduction_steps = [reduce_pca, reduce_ica, reduce_rp, reduce_dt]
    clustering_classes = [KMeans, GaussianMixture]

    for dims_reduction_step in dims_reduction_steps:
        x_reduced, reduction_alg = dims_reduction_step(dataset_name, x_train,
                                                       y_train, sync, n_jobs,
                                                       None)

        for clustering_class in clustering_classes:
            postfix = reduction_alg.__class__.__name__
            clustering_alg = _get_clusterer(clustering_class,
                                            f'{dataset_name}-{postfix}',
                                            x_reduced, n_jobs, False)

            for add_to_reduced_features in [False, True]:
                dataset_name_ = dataset_name
                if add_to_reduced_features:
                    dataset_name_ += '_add_to_reduced_features'

                _sync_reduction_clustering_nn(dataset_name_, x_train, y_train,
                                              x_val, y_val, reduction_alg,
                                              clustering_alg,
                                              add_to_reduced_features)

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from tasks.dims_reduction import reduce_pca, reduce_ica, reduce_rp, reduce_dt, \
    VectorVisualizationFunction, ReductionAlgorithm
from utils.nnestimator import NeuralNetworkEstimator, training_curves
from utils.outputs import reduction_and_nn_joblib, training_curve

HIDDEN_LAYERS = [16] * 4


class ReductionAndNN:

    def __init__(self, reduction_alg: ReductionAlgorithm):
        self.reduction_alg = reduction_alg
        self.nn: Optional[NeuralNetworkEstimator] = None

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray = None,
            y_val: np.ndarray = None,
            learning_rate=1e-6,
            batch_size=64,
            epochs=50,
            weight_decay=1e-6,
            verbose=False):

        x_train_red = self.reduction_alg.transform(x_train)
        x_val_red = self.reduction_alg.transform(x_val)
        in_features = x_train_red.shape[1]

        # Infer n_classes from y_train and y_val
        max1 = np.max(y_train)
        max2 = np.max(y_val)

        num_classes = np.max([max1, max2]) + 1

        nn = NeuralNetworkEstimator(in_features, num_classes, HIDDEN_LAYERS)
        nn.fit(x_train_red, y_train, x_val_red, y_val, learning_rate,
               batch_size, epochs, weight_decay, verbose)
        self.nn = nn

    def predict(self, X: np.ndarray):
        x_reduced = self.reduction_alg.transform(X)
        predicted = self.nn.predict(x_reduced)
        return predicted


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

    # Assume all dimensionality reduction algorithms have been run
    sync = False

    dims_reduction_steps = [reduce_pca, reduce_ica, reduce_rp, reduce_dt]

    for dims_reduction_step in dims_reduction_steps:

        x_reduced, reduction_alg = dims_reduction_step(dataset_name, x_train,
                                                       y_train, sync, n_jobs,
                                                       None)
        _sync_reduction_and_nn(dataset_name, x_train, y_train, x_val, y_val,
                               reduction_alg)


def _sync_reduction_and_nn(dataset_name: str, x_train: np.ndarray,
                           y_train: np.ndarray, x_val: np.ndarray,
                           y_val: np.ndarray, reduction_alg):
    """Synchronizes dimensionality reduction and NN outputs

    The following files are synchronized:
      - The serialized ReductionAndNN object
      - The corresponding training curve i.e. acc vs iter

    Args:
        dataset_name: Dataset name
        x_train: Training features of shape
            `(n_train, n_features)`
        y_train: Training labels of shape `(n_train, )`
        x_val: Validation features of shape
            `(n_val, n_features)`
        y_val: Validation labels of shape `(n_val, )`
        reduction_alg: Reduction algorithm object
            which implements `.transform()` method.

    Returns:
        A ReductionAndNN object. In addition, this function
            may write files to system.

    """
    alg_name = reduction_alg.__class__.__name__
    joblib_path = reduction_and_nn_joblib(dataset_name, alg_name)
    training_curve_path = training_curve(dataset_name, alg_name)

    files_exist = os.path.exists(joblib_path) and os.path.exists(
        training_curve_path)

    if not files_exist:
        red_nn = ReductionAndNN(reduction_alg)
        red_nn.fit(x_train,
                   y_train,
                   x_val,
                   y_val,
                   learning_rate=1e-5,
                   epochs=2000,
                   batch_size=min(len(x_train), 1024),
                   verbose=True)

        joblib.dump(red_nn, joblib_path)
        loss_fig, acc_fig = training_curves(red_nn.nn.training_log)
        acc_fig.write_image(training_curve_path)

    red_nn = joblib.load(joblib_path)
    train_acc = accuracy_score(y_train, red_nn.predict(x_train))
    val_acc = accuracy_score(y_val, red_nn.predict(x_val))
    print(f'{reduce_pca.__name__} train_acc {train_acc}, val_acc {val_acc}')

    return red_nn

import os
import contextlib

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split


@contextlib.contextmanager
def temp_seed(seed):
    """Temporarily sets Numpy seed

    Code taken from
    https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed

    Args:
        seed: Input seed

    Returns:
        None. This is a context manager.

    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def gen_3d_data(x1_size: float = 5,
                x2_size: float = 2,
                n_train: int = 5000,
                n_val: int = 500,
                n_test: int = 500,
                noise_prob: float = 0.01):
    # _gen_3d_examples() is a deterministic function, so we cannot call it three
    # times to produce train, val, and test sets. The correct way to do it
    # is by producing all samples, then split to train, val, and test.
    with temp_seed(1234):
        x_all, y_all = _gen_3d_examples(x1_size, x2_size,
                                        n_train + n_val + n_test, noise_prob)

        x_train, x_valtest, y_train, y_valtest = train_test_split(
            x_all, y_all, train_size=n_train)

        x_val, x_test, y_val, y_test = train_test_split(x_valtest,
                                                        y_valtest,
                                                        test_size=n_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def _gen_3d_examples(x1_size: float, x2_size: float, num_examples: int,
                     noise_prob: float):
    """Generates 3D examples

    Each example has coordinates [x1, x2] where:
      - x1 is on interval [0, x1]
      - x2 is on interval [0, x2]

    Args:
        x1_size: Size of first axis.
        x2_size: Size of second axis.
        num_examples: Number of examples to generate
        noise_prob: Probability of label flip

    Returns:
        A tuple `(x_data, y_data)` of shapes `(num_points, 3)` and
        `(num_points, )`, respectively.

    """

    mean_ = np.array([x1_size / 2, x2_size / 2])
    std1 = (x1_size / 3.5)
    std2 = (x2_size / 3.5)
    corr = 0.4 * std1 * std2
    cov_ = np.array([[std1**2, corr], [corr, std2**2]])

    # x1x2_data = np.random.multivariate_normal(mean_, cov_, size=num_examples)

    x1_data = np.random.uniform(-x1_size / 2, x1_size / 2, size=num_examples)
    x2_data = np.random.uniform(-x2_size / 2, x2_size / 2, size=num_examples)
    x1x2_data = np.stack([x1_data, x2_data], axis=-1)

    T = np.array([[5, 1], [1, 5]])
    T = T / np.linalg.norm(T, axis=1).reshape((-1, 1))

    x1x2_data = np.dot(T, x1x2_data.T)
    x1x2_data = x1x2_data.T + [x1_size / 2, x2_size / 2]

    # Construct labels
    y_data = np.zeros((num_examples,))

    prob = _ground_truth_proba(x1x2_data, x1_size, x2_size)
    rand = np.random.uniform(low=0.1, high=1.0, size=(num_examples,))
    y_data[rand <= prob] = 1

    # Noise
    noise_flags: np.ndarray = np.random.binomial(
        n=2, p=noise_prob, size=(num_examples,)).astype(bool)
    y_data[noise_flags] = 1 - y_data[noise_flags]

    y_data = y_data.astype(np.int)

    x3_mean = (y_data - 0.5)  # Probability [0, 1] -> [-0.5 +0.5]
    # x3_mean = np.zeros((num_examples, ))
    # std = 2 * np.sqrt(0.25 - np.power(np.power(prob, 10000) - 0.5, 2))
    std = 0.1
    x3 = np.random.normal(x3_mean, std)

    x_data = np.concatenate([x1x2_data, x3.reshape((-1, 1))], axis=-1)

    return x_data, y_data


def _ground_truth_proba(x_data: np.ndarray, x1_size: float, x2_size: float):
    """Generates Dataset3D ground truth logits

    Args:
        x_data: Features
        x1_size: Size of x1 parameter
        x2_size: Size of x2 parameter

    Returns:
        Logits of `x_data`. They can be converted to
            probabilities or labels.

    """
    x1 = x_data[:, 0]
    x2 = x_data[:, 1]

    # Conditions for positive labels
    cond1 = x1 / (x1_size / 2) + x2 / x2_size - 1
    cond2 = np.square(
        (x1 - x1_size) / (x1_size / 2)) + np.square(x2 / (x2_size / 2)) - 1
    cond3 = np.max(np.stack([
        x1_size * 0.3 - x1, x2_size * 0.5 - x2, x1 - 0.75 * x1_size,
        x2 - 0.9 * x2_size
    ],
                            axis=-1) * 2,
                   axis=-1)
    logits = np.minimum(np.minimum(cond1, cond2), cond3)

    proba = 1 / (1 + np.exp(logits / 0.1))

    return proba


class Dataset3DGroundTruth:
    """The Dataset3D ground truth

    The interface follows the Scikit-Learn estimator with
    the following methods:

      1. `predict_proba()`
      2. `predict()`

    """

    def __init__(self, x1_size: float, x2_size: float):
        """Initializes a ground truth object

        Args:
            x1_size: `x1_size` parameter of the Dataset3D
            x2_size: `x2_size` parameter of the Dataset3D
        """
        self.x1_size = x1_size
        self.x2_size = x2_size

    def predict_proba(self, x_data: np.ndarray):
        """Predicts probabilities of the given features

        Args:
            x_data: A `(num_examples, 2)` Dataset3D feature
                array

        Returns:
            A `(num_examples, 2)` numpy array, where the
                first and second columns correspond to the
                negative and positive probabilities,
                respectively. Each row sums up to 1.

        """
        p = _ground_truth_proba(x_data, self.x1_size, self.x2_size)
        proba = np.stack([1 - p, p], axis=-1)
        return proba

    def predict(self, x_data: np.ndarray):
        """Predicts the label of the given features

        Args:
            x_data: A `(num_examples, 2)` Dataset3D feature
                array

        Returns:
            A `(num_examples, )` numpy array whose elements
            are the predicted labels

        """
        proba = self.predict_proba(x_data)
        labels = (proba[:, 1] > 0.5).astype(float)
        return labels


def get_fashion_mnist_data():
    with temp_seed(1234):
        mnist_x_train, mnist_y_train = _get_fashion_mnist_examples(train=True)
        x_test, y_test = _get_fashion_mnist_examples(train=False)

        x_train, x_val, y_train, y_val = train_test_split(mnist_x_train,
                                                          mnist_y_train,
                                                          test_size=0.2)
    return x_train, y_train, x_val, y_val, x_test, y_test


def _get_fashion_mnist_examples(train: bool):
    """Gets the Fashion MNIST 10% examples

    Args:
        train: If True, this function gets the training set. Otherwise,
            it gets the validation set.

    Returns:
        Tuple (x_data, y_data) where
            1. `x_data` is an array of shape (num_examples, 784)
            2. `y_data` is an array of shape (num_examples, )

    """

    dir_name = 'mnist'
    os.makedirs(dir_name, exist_ok=True)
    mnist = torchvision.datasets.FashionMNIST(dir_name, train, download=True)
    x = np.stack([np.array(x).flatten().copy() for x, _ in mnist])
    y = np.array([y for _, y in mnist])

    assert len(x) == len(y)
    num_examples = len(x)

    # Adjust these lines if need a smaller dataset
    indices = np.arange(0, num_examples, 10)
    x_resampled = x[indices]
    y_resampled = y[indices]

    return x_resampled, y_resampled


def check_input(X: np.ndarray):
    """Checks if input array X is valid

    A valid `X`:
      - is 2-dimensional
      - has non-zero length
      - non-empty features

    Args:
        X: A feature matrix of shape `(N, n_features)`.

    Returns:
        None. This function may raise an exception.

    Raises:
        ValueError if any condition fails.

    """
    if len(X.shape) != 2:
        raise ValueError('Input X must be 2-dimensional')

    if len(X) < 1:
        raise ValueError('Input X must be non-empty')

    if X.shape[1] <= 0:
        raise ValueError('Input X must have at least a feature.')

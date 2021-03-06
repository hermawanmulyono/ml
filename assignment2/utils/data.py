import contextlib

import numpy as np
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


def gen_2d_data(x1_size: float, x2_size: float, n_train: int, n_val: int,
                n_test: int, noise_prob: float):
    # _gen_2d_examples() is a deterministic function, so we cannot call it three
    # times to produce train, val, and test sets. The correct way to do it
    # is by producing all samples, then split to train, val, and test.
    with temp_seed(1234):
        x_all, y_all = _gen_2d_examples(x1_size, x2_size,
                                        n_train + n_val + n_test, noise_prob)

        x_train, x_valtest, y_train, y_valtest = train_test_split(
            x_all, y_all, train_size=n_train)

        x_val, x_test, y_val, y_test = train_test_split(x_valtest,
                                                        y_valtest,
                                                        test_size=n_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def _gen_2d_examples(x1_size: float, x2_size: float, num_examples: int,
                     noise_prob: float):
    """Generates 2D examples

    Each example has coordinates [x1, x2] where:
      - x1 is on interval [0, x1]
      - x2 is on interval [0, x2]

    Args:
        x1_size: Size of first axis.
        x2_size: Size of second axis.
        num_examples: Number of examples to generate
        noise_prob: Probability of label flip

    Returns:
        A tuple `(x_data, y_data)` of shapes `(num_points, 2)` and
        `(num_points, )`, respectively.

    """

    mean_ = np.array([x1_size / 2, x2_size / 2])
    std1 = (x1_size / 3.5)
    std2 = (x2_size / 3.5)
    corr = 0.4 * std1 * std2
    cov_ = np.array([[std1**2, corr], [corr, std2**2]])

    x_data = np.random.multivariate_normal(mean_, cov_, size=num_examples)

    # Construct labels
    y_data = np.zeros((num_examples,))

    prob = _ground_truth_proba(x_data, x1_size, x2_size)
    rand = np.random.uniform(low=0.1, high=1.0, size=(num_examples,))
    y_data[rand <= prob] = 1

    # Noise
    noise_flags: np.ndarray = np.random.binomial(
        n=2, p=noise_prob, size=(num_examples,)).astype(bool)
    y_data[noise_flags] = 1 - y_data[noise_flags]

    y_data = y_data.astype(np.int)

    return x_data, y_data


def _ground_truth_proba(x_data: np.ndarray, x1_size: float, x2_size: float):
    """Generates Dataset2D ground truth logits

    Args:
        x_data: Features
        x1_size: Size of x1 parameter
        x2_size: Size of x2 parameter

    Returns:
        Logits of `x_data`. They can be converted to
            qprobabilities or labels.

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


class Dataset2DGroundTruth:
    """The Dataset2D ground truth

    The interface follows the Scikit-Learn estimator with
    the following methods:

      1. `predict_proba()`
      2. `predict()`

    """

    def __init__(self, x1_size: float, x2_size: float):
        """Initializes a ground truth object

        Args:
            x1_size: `x1_size` parameter of the Dataset2D
            x2_size: `x2_size` parameter of the Dataset2D
        """
        self.x1_size = x1_size
        self.x2_size = x2_size

    def predict_proba(self, x_data: np.ndarray):
        """Predicts probabilities of the given features

        Args:
            x_data: A `(num_examples, 2)` Dataset2D feature
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
            x_data: A `(num_examples, 2)` Dataset2D feature
                array

        Returns:
            A `(num_examples, )` numpy array whose elements
            are the predicted labels

        """
        proba = self.predict_proba(x_data)
        labels = (proba[:, 1] > 0.5).astype(float)
        return labels


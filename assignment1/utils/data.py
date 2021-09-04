import os

import numpy as np
import torchvision
from torchvision.transforms import Resize


def gen_2d_data(x1_max: float, x2_max: float, num_examples: int,
                noise_prob: float):
    """Generates 2D dataset

    Each example has coordinates [x1, x2] where:
      - x1 is on interval [0, x1]
      - x2 is on interval [0, x2]

    Args:
        x1_max: Maximum value of first axis.
        x2_max: Maximum value of second axis
        num_examples: Number of examples to generate
        noise_prob: Probability of label flip

    Returns:
        A tuple `(x_data, y_data)` of shapes `(num_points, 2)` and
        `(num_points, )`, respectively.

    """
    x_data = np.random.rand(num_examples, 2) * np.array([[x1_max, x2_max]])

    x1 = x_data[:, 0]
    x2 = x_data[:, 1]

    # Construct labels
    y_data = np.zeros((num_examples,))

    # Conditions for positive labels
    cond1 = x1 / (x1_max / 2) + x2 / x2_max - 1
    cond2 = np.square(
        (x1 - x1_max) / (x1_max / 2)) + np.square(x2 / (x2_max / 2)) - 1
    cond3 = np.max(np.stack(
        [x1_max / 2 - x1, x2_max * 0.7 - x2, x1 - 0.75 * x1_max], axis=-1),
                   axis=-1)

    cond = np.minimum(np.minimum(cond1, cond2), cond3)
    prob = 1 / (1 + np.exp(cond / 0.1))
    rand = np.random.uniform(low=0.1, high=1.0, size=(num_examples,))
    y_data[rand <= prob] = 1

    # Noise
    noise_flags: np.ndarray = np.random.binomial(
        n=2, p=noise_prob, size=(num_examples,)).astype(bool)
    y_data[noise_flags] = 1 - y_data[noise_flags]

    y_data = y_data.astype(np.int)

    return x_data, y_data


def get_mnist(train: bool):
    """Gets x_data, y_data of the MNIST dataset, resampled 5%

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
    mnist = torchvision.datasets.MNIST(dir_name, train, download=True)
    x = np.stack([np.array(x).flatten().copy() for x, _ in mnist])
    y = np.array([y for _, y in mnist])

    assert len(x) == len(y)
    num_examples = len(x)

    # Resample 5% only
    indices = np.arange(0, num_examples, 5)
    x_resampled = x[indices]
    y_resampled = y[indices]

    return x_resampled, y_resampled


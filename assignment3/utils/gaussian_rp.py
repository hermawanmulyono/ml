from typing import Optional

import numpy as np


class GaussianRP:

    def __init__(self, n_components: int):
        self.R: Optional[np.ndarray] = None
        self.n_components = n_components
        self.n_features: Optional[int] = None

    def fit(self, X: np.ndarray):
        _check_input(X)
        self.n_features = X.shape[1]
        self.R = _make_proj_matrix(self.n_components, self.n_features)

        assert self.R.shape == (self.n_components, self.n_features)
        assert np.allclose(1.0, np.linalg.norm(self.R, axis=1))

    def transform(self, X: np.ndarray):
        _check_input(X)

        assert self.R is not None

        if X.shape[1] != self.R.shape[1]:
            raise ValueError(f'Invalid X shape, expecting (N, {self.n_features})')

        x_proj = np.dot(X, self.R.T)

        return x_proj

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    @property
    def components_(self):
        return self.R


def _check_input(X: np.ndarray):
    """Checks if input array X is valid

    A valid `X`:
      - is 2-dimensional
      - has non-zero length

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


def _make_proj_matrix(n_components: int, n_features: int):
    """Makes a projection matrix

    A projection matrix is defined as an orthonormal matrix
    where each row is a unit vector, and any pair of rows
    are orthogonal vectors. The shape is
    `(n_components, n_features)`.

    Args:
        n_components: Number of components
        n_features: Number of features

    Returns:
        Projection matrix of shape
            `(n_components, n_features)`.

    """

    # From Scikit-Learn
    components = np.random.normal(loc=0.0,
                                  scale=1.0 / np.sqrt(n_components),
                                  size=(n_components, n_features))

    # Gram-Schmidt process from
    # http://mlwiki.org/index.php/Gram-Schmidt_Process
    # It works on columns, so need to transpose later.

    A = components.T

    def normalize(v):
        return v / np.sqrt(v.dot(v))

    n = A.shape[1]

    A[:, 0] = normalize(A[:, 0])

    for i in range(1, n):
        Ai = A[:, i]  # i'th column

        for j in range(0, i):
            # j'th column is already normalized and orthonormal
            Aj = A[:, j]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj  # Residual

        A[:, i] = normalize(Ai)

    components = A.T

    return components

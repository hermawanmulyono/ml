from typing import Optional

import numpy as np

from utils.data import check_input


class GaussianRP:

    def __init__(self, n_components: int):
        self.R: Optional[np.ndarray] = None
        self.n_components = n_components
        self.n_features: Optional[int] = None
        self.mean: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        check_input(X)
        self.n_features = X.shape[1]
        self.R = _make_proj_matrix(self.n_components, self.n_features)
        self.mean = np.mean(X, axis=0)

        assert self.R.shape == (self.n_components, self.n_features)
        # assert np.allclose(1.0, np.linalg.norm(self.R, axis=1))

    def transform(self, X: np.ndarray):
        check_input(X)

        if self.R is None:
            raise RuntimeError('fit() must be called before transform()')

        if X.shape[1] != self.R.shape[1]:
            raise ValueError(
                f'Invalid X shape, expecting (N, {self.n_features})')

        x_centered = X - self.mean
        pseudo_inverse = np.linalg.inv(np.dot(self.R, self.R.T))
        x_proj = np.dot(np.dot(x_centered, self.R.T), pseudo_inverse)

        return x_proj

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    @property
    def components_(self):
        return self.R

    @property
    def mean_(self):
        return self.mean

    def reconstruct(self, X: np.ndarray):
        check_input(X)

        x_proj = self.transform(X)
        R = self.R
        x_rec = np.dot(x_proj, R) + self.mean

        return x_rec


def _make_proj_matrix(n_components: int, n_features: int,
                      orthogonalize: bool = False):
    """Makes a projection matrix

    A projection matrix is defined as a random matrix
    from a Gaussian distribution. The shape is
    `(n_components, n_features)`.

    Args:
        n_components: Number of components
        n_features: Number of features
        orthogonalize: If True, Gram-Schmidt process is
            applied.

    Returns:
        Projection matrix of shape
            `(n_components, n_features)`.

    """

    # From Scikit-Learn
    components = np.random.normal(loc=0.0,
                                  scale=1.0 / np.sqrt(n_components),
                                  size=(n_components, n_features))

    if not orthogonalize:
        return components

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

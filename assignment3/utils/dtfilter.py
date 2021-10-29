import copy
from typing import List, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from utils.data import check_input


class DTFilter:
    def __init__(self, threshold=0.5):
        self._threshold = threshold
        self._selected_features: Optional[List[int]] = None
        self._dt: Optional[DecisionTreeClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        check_input(X)
        dt = DecisionTreeClassifier()
        dt.fit(X, y)

        self._dt = dt

        importances = dt.feature_importances_

        assert np.isclose(np.sum(dt.feature_importances_), 1.0)

        selected_features = []

        sorted_features = np.argsort(-importances)
        for feature in sorted_features:
            selected_features.append(int(feature))

            if np.sum(importances[selected_features]) >= self._threshold:
                break

        self._selected_features = selected_features

    def transform(self, X: np.ndarray):
        check_input(X)

        if not self._selected_features:
            raise RuntimeError('fit() must be called before transform()')

        return X[:, self._selected_features]

    @property
    def dt(self):
        return copy.deepcopy(self._dt)

    @property
    def selected_features(self):
        return copy.deepcopy(self._selected_features)

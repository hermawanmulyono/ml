from typing import List, Union

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from utils.nnestimator import NeuralNetworkEstimator

SklearnModel = Union[KNeighborsClassifier, SVC, DecisionTreeClassifier,
                     AdaBoostClassifier]


def get_decision_tree(ccp_alpha: float) -> DecisionTreeClassifier:
    """Constructs a decision tree with pruning

    Args:
        ccp_alpha: Pruning minimal cost-complexity parameter alpha

    Returns:
        A decision tree classifier object

    """
    dt = DecisionTreeClassifier(criterion='entropy',
                                splitter='random',
                                ccp_alpha=ccp_alpha)
    return dt


def get_boosting(n_estimators: int, ccp_alpha: float) -> AdaBoostClassifier:
    base_estimator = DecisionTreeClassifier(criterion='entropy',
                                            splitter='best',
                                            ccp_alpha=ccp_alpha)

    adaboost = AdaBoostClassifier(base_estimator, n_estimators=n_estimators)

    return adaboost


def get_svm(kernel: str, **kwargs) -> SVC:
    return SVC(kernel=kernel, **kwargs)


def get_knn(n_neighbors: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors)


def get_nn(in_features: int, num_classes: int,
           layer_width: int, num_layers: int) -> NeuralNetworkEstimator:

    hidden_layers = [layer_width] * num_layers
    nn_est = NeuralNetworkEstimator(in_features, num_classes, hidden_layers)

    return nn_est

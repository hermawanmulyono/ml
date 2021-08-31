from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import torch
import numpy as np

from utils.nnestimator import NeuralNetworkEstimator


def get_decision_tree(ccp_alpha: float) -> DecisionTreeClassifier:
    """Constructs a decision tree with pruning

    Args:
        ccp_alpha: Pruning minimal cost-complexity paramether alpha

    Returns:
        A decision tree classifier object

    """
    dt = DecisionTreeClassifier(criterion='entropy',
                                splitter='best',
                                ccp_alpha=ccp_alpha)
    return dt


def get_boosting(n_estimators: int, ccp_alpha: float) -> AdaBoostClassifier:
    base_estimator = DecisionTreeClassifier(criterion='entropy',
                                            splitter='best',
                                            ccp_alpha=ccp_alpha)

    adaboost = AdaBoostClassifier(base_estimator, n_estimators=n_estimators)

    return adaboost


def get_svm(kernel: str, **kernel_params) -> SVC:
    return SVC(kernel=kernel, **kernel_params)


def get_nn(in_features: int, num_classes: int, num_hidden_layers: int,
           hidden_layer_size: int) -> NeuralNetworkEstimator:

    nn_est = NeuralNetworkEstimator(in_features, num_classes,
                                    num_hidden_layers, hidden_layer_size)

    return nn_est


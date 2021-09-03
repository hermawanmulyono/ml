import logging
import os

import numpy as np
import torch.random

from utils.data import gen_2d_data, get_mnist
from utils.plots import visualize_2d_data, visualize_2d_decision_boundary, \
    training_size_curve
from utils.models import get_decision_tree, get_boosting, get_svm, get_nn, \
    get_knn
from utils.nnestimator import training_curves, NeuralNetworkEstimator


def dataset1():
    x1_max = 5
    x2_max = 2
    n_train = 9000
    n_val = 1000
    n_test = 1000
    noise_prob = 0.005
    x_train, y_train = gen_2d_data(x1_max, x2_max, n_train, noise_prob)
    x_val, y_val = gen_2d_data(x1_max, x2_max, n_val, noise_prob)
    x_test, y_test = gen_2d_data(x1_max, x2_max, n_test, noise_prob)

    # visualize_2d_data(x_train, y_train, '2D Data').show()

    dt = get_decision_tree(ccp_alpha=0.001)

    training_size_curve(dt, x_train, y_train, x_val, y_val,
                        [0.2, 0.4, 0.6, 0.8, 1.0], title='Decision Tree').show()

    # dt.fit(x_train, y_train)

    # visualize_2d_decision_boundary(dt, x1_max, x2_max, x_train, y_train,
    #                                'Decision tree').show()

    # adaboost = get_boosting(200, 0.002)
    # adaboost.fit(x_train, y_train)
    #
    # visualize_2d_decision_boundary(adaboost, x1_max, x2_max, x_train, y_train,
    #                                'Adaboost').show()

    # svm_linear = get_svm('linear')
    # svm_linear.fit(x_train, y_train)
    # visualize_2d_decision_boundary(svm_linear, x1_max, x2_max, x_train, y_train,
    #                                'SVM-Linear').show()
    #
    # svm_linear = get_svm('rbf', gamma=0.5)
    # svm_linear.fit(x_train, y_train)
    # visualize_2d_decision_boundary(svm_linear, x1_max, x2_max, x_train, y_train,
    #                                'SVM-Linear').show()

    # knn = get_knn(50)
    # knn.fit(x_train, y_train)
    #
    # fig = visualize_2d_decision_boundary(knn, x1_max, x2_max, x_train, y_train,
    #                                      'knn')
    # fig.show()

    # nn = get_nn(in_features=2,
    #             num_classes=2,
    #             hidden_layers=[8, 16, 16, 8])
    #
    # path_to_state_dict = 'nn.pt'
    # if os.path.exists(path_to_state_dict):
    #     nn.load(path_to_state_dict)
    # else:
    #     nn.fit(x_train, y_train, x_test, y_test, learning_rate=5e-4,
    #            batch_size=1024, epochs=1000, verbose=True)
    #     nn.save(path_to_state_dict)
    #
    # loss_fig, acc_fig = training_curves(nn.training_log)
    # loss_fig.show()
    # acc_fig.show()
    #
    # fig = visualize_2d_decision_boundary(nn, x1_max, x2_max, x_train, y_train,
    #                                      'Neural network')
    # fig.show()


def dataset2():
    x_train, y_train = get_mnist(train=True)
    x_test, y_test = get_mnist(train=False)

    in_features = x_train.shape[1]

    path_to_state_dict = 'nn_mnist.pt'
    if os.path.exists(path_to_state_dict):
        nn = NeuralNetworkEstimator.from_state_dict(path_to_state_dict)
    else:
        nn = get_nn(in_features=in_features,
                    num_classes=10,
                    hidden_layers=[8, 16, 16, 8])
        nn.fit(x_train, y_train, x_test, y_test, learning_rate=1e-5,
               batch_size=128, epochs=1000, verbose=True)
        nn.save(path_to_state_dict)

    loss_fig, acc_fig = training_curves(nn.training_log)
    loss_fig.show()
    acc_fig.show()


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    dataset1()
    # dataset2()

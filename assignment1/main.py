import logging
import multiprocessing

import numpy as np
import torch.random
from sklearn.model_selection import train_test_split

from utils.data import gen_2d_data, get_fashion_mnist
from utils.tasks import dt_task, knn_task, svm_poly_task, svm_rbf_task, \
    boosting_task, neural_network_task


def dataset1(n_jobs):
    x1_max = 5
    x2_max = 2
    n_train = 9000
    n_val = 1000
    n_test = 1000
    noise_prob = 0.005
    x_train, y_train = gen_2d_data(x1_max, x2_max, n_train, noise_prob)
    x_val, y_val = gen_2d_data(x1_max, x2_max, n_val, noise_prob)
    x_test, y_test = gen_2d_data(x1_max, x2_max, n_test, noise_prob)

    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    # visualize_2d_data(x_train, y_train, '2D Data').show()

    dt_task(x_train, y_train, x_val, y_val, train_sizes, 'Dataset2D', n_jobs)

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


def dataset2(n_jobs):
    mnist_x_train, mnist_y_train = get_fashion_mnist(train=True)
    x_test, y_test = get_fashion_mnist(train=False)

    x_train, x_val, y_train, y_val = train_test_split(mnist_x_train,
                                                      mnist_y_train,
                                                      test_size=0.2)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert x_train.shape[1] == x_val.shape[1] == x_test.shape[1]

    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    # dt_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
    #         n_jobs)
    # knn_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
    #          n_jobs)
    # svm_poly_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
    #               n_jobs)
    # svm_rbf_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
    #              n_jobs)
    # boosting_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
    #               n_jobs)
    neural_network_task(x_train, y_train, x_val, y_val, train_sizes,
                        'Fashion-MNIST', n_jobs)


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    num_jobs = multiprocessing.cpu_count()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    # dataset1(num_jobs)
    dataset2(num_jobs)

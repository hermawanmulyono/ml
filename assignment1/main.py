"""Assignment 1 - Supervised Learning

Author: Hermawan Mulyono
GT Username: hmulyono3

This script is the entry point of assignment 1.

Examples:

$ # Show help
$ python3 main --help
$ # Run with cache files in the `outputs` directory
$ python3 main.py
$ # Run with svm and neural network retrained
$ python3 main.py --svm --nn

"""

import argparse
import logging
import multiprocessing

import numpy as np
import torch.random
from sklearn.model_selection import train_test_split

from utils.data import gen_2d_data, get_fashion_mnist
from utils.plots import visualize_2d_data
from utils.tasks import dt_task, knn_task, svm_poly_task, svm_rbf_task, \
    boosting_task, neural_network_task, OUTPUT_DIRECTORY


def dataset1(train_dt: bool, train_boosting: bool, train_svm: bool,
             train_knn: bool, train_nn: bool, n_jobs: int):
    x1_size = 5
    x2_size = 2
    n_train = 5000
    n_val = 200
    n_test = 200
    noise_prob = 0.01
    x_train, y_train = gen_2d_data(x1_size, x2_size, n_train, noise_prob)
    x_val, y_val = gen_2d_data(x1_size, x2_size, n_val, noise_prob)
    x_test, y_test = gen_2d_data(x1_size, x2_size, n_test, noise_prob)

    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    dataset2d_file_path = f'{OUTPUT_DIRECTORY}/Dataset2D.png'
    visualize_2d_data(x_train, y_train,
                      '2D Data').write_image(dataset2d_file_path)

    exit()

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


def dataset2(train_dt: bool, train_boosting: bool, train_svm: bool,
             train_knn: bool, train_nn: bool, n_jobs: int):
    mnist_x_train, mnist_y_train = get_fashion_mnist(train=True)
    x_test, y_test = get_fashion_mnist(train=False)

    x_train, x_val, y_train, y_val = train_test_split(mnist_x_train,
                                                      mnist_y_train,
                                                      test_size=0.2)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert x_train.shape[1] == x_val.shape[1] == x_test.shape[1]

    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    dt_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
            n_jobs, train_dt)
    knn_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
             n_jobs, train_knn)
    # svm_poly_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
    #               n_jobs, train_svm)
    svm_rbf_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
                 n_jobs, train_svm)
    boosting_task(x_train, y_train, x_val, y_val, train_sizes, 'Fashion-MNIST',
                  n_jobs, train_boosting)
    neural_network_task(x_train, y_train, x_val, y_val, train_sizes,
                        'Fashion-MNIST', n_jobs, train_nn)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    models = [
        ('dt', 'decision tree'),
        ('boosting', 'boosting'),
        ('svm', 'support vector machine'),
        ('knn', 'k-nearest neighbors'),
        ('nn', 'neural network'),
    ]

    for flag, model_name in models:
        parser.add_argument(f'--{flag}',
                            action='store_true',
                            help=f'If given, {model_name} will be retrained.')

    parser.add_argument('-n',
                        '--num-jobs',
                        type=int,
                        default=0,
                        help='Number of jobs to be deployed. If num-jobs is'
                        ' 0, all available CPUs will be used.')

    args = parser.parse_args()

    if args.num_jobs == 0:
        num_jobs = multiprocessing.cpu_count()
    else:
        num_jobs = args.num_jobs

    kwargs = {
        'train_dt': args.dt,
        'train_boosting': args.boosting,
        'train_svm': args.svm,
        'train_knn': args.knn,
        'train_nn': args.nn,
        'n_jobs': num_jobs
    }

    return kwargs


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    kwargs = parse_args()

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    # dataset1(**kwargs)
    dataset2(**kwargs)

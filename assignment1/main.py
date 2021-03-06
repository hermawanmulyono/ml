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

from typing import NamedTuple
import argparse
import logging
import os

import numpy as np
import skimage.io
import torch.random
from sklearn.metrics import accuracy_score

from utils.data import gen_2d_data, _get_fashion_mnist_examples, Dataset2DGroundTruth, \
    get_fashion_mnist_data
from utils.plots import visualize_2d_data, visualize_2d_decision_boundary, \
    sample_mnist_dataset
from utils.tasks import dt_task, knn_task, svm_poly_task, svm_rbf_task, \
    boosting_task, neural_network_task
from utils.output_files import dataset2d_fig_path, \
    decision_boundary_fig_path, fashion_mnist_samples_fig_path


class TrainFlags(NamedTuple):
    """A named tuple to contain which model needs to
    be retrained.

    The flags are:
      - train_dt: Train decision tree
      - train_boosting: Train boosting
      - train_svm: Train SVM
      - train_knn: Train KNN
      - train_nn: Train Neural Network
    """
    train_dt: bool
    train_boosting: bool
    train_svm: bool
    train_knn: bool
    train_nn: bool


def dataset1():
    x1_size = 5
    x2_size = 2
    n_train = 5000
    n_val = 500
    n_test = 500
    noise_prob = 0.01

    x_train, y_train, x_val, y_val, x_test, y_test = gen_2d_data(
        x1_size, x2_size, n_train, n_val, n_test, noise_prob)

    assert len(x_train) == len(y_train) == n_train
    assert len(x_val) == len(y_val) == n_val
    assert len(x_test) == len(y_test) == n_test

    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    dataset_labels = ['negative', 'positive']
    dataset_name = 'Dataset2D'

    # Generate dataset visualization
    dataset2d_fig = visualize_2d_data(x_train, y_train, f'Train {dataset_name}')
    dataset2d_fig.write_image(dataset2d_fig_path())

    dt, knn, svm_poly, svm_rbf, boosting, nn = _run_all_models(
        x_train, y_train, x_val, y_val, x_test, y_test, train_sizes,
        dataset_name, dataset_labels)

    # Plot decision boundary figures
    models = [dt, knn, svm_poly, svm_rbf, boosting, nn]
    model_names = ['DT', 'KNN', 'SVM-Polynomial', 'SVM-RBF', 'Boosting', 'NN']

    for model, model_name in zip(models, model_names):
        fig_path = decision_boundary_fig_path(model_name, dataset_name)
        if not os.path.exists(fig_path):
            plot_title = f'{model_name} {dataset_name} Decision Boundary'

            fig = visualize_2d_decision_boundary(model, x1_size, x2_size,
                                                 x_train, y_train, plot_title)
            fig.write_image(fig_path)

    fig_path = decision_boundary_fig_path('SVM-Polynomial-Zoomed out',
                                          dataset_name)
    if not os.path.exists(fig_path):
        plot_title = f'SVM-Polynomial-Zoomed out {dataset_name} Decision Boundary'

        fig = visualize_2d_decision_boundary(svm_poly, 10 * x1_size,
                                             10 * x2_size, x_train, y_train,
                                             plot_title)
        fig.write_image(fig_path)

    # Ground truth model
    gt_model = Dataset2DGroundTruth(x1_size, x2_size)

    # Plot Dataset2D ground truth decision boundary
    fig_path = decision_boundary_fig_path('Dataset2D-Ground-Truth',
                                          dataset_name)
    if not os.path.exists(fig_path):
        plot_title = f'{dataset_name}'
        fig = visualize_2d_decision_boundary(gt_model,
                                             x1_size,
                                             x2_size,
                                             x_train,
                                             y_train,
                                             plot_title,
                                             scatter_size=5)
        fig.write_image(fig_path)

    # Evaluate the best possible accuracy for train, val and test
    y_pred = gt_model.predict(x_train)
    print(f'Best train accuracy {accuracy_score(y_train, y_pred)}')
    y_pred = gt_model.predict(x_val)
    print(f'Best train accuracy {accuracy_score(y_val, y_pred)}')
    y_pred = gt_model.predict(x_test)
    print(f'Best train accuracy {accuracy_score(y_test, y_pred)}')


def dataset2():
    x_train, y_train, x_val, y_val, x_test, y_test = get_fashion_mnist_data()

    dataset_labels = [
        'T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal',
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    dataset_name = 'Fashion-MNIST'

    # Get some samples
    samples = sample_mnist_dataset(x_train, y_train, 10)
    skimage.io.imsave(fashion_mnist_samples_fig_path(), samples)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert x_train.shape[1] == x_val.shape[1] == x_test.shape[1]

    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    _run_all_models(x_train, y_train, x_val, y_val, x_test, y_test, train_sizes,
                    dataset_name, dataset_labels)


def _run_all_models(x_train, y_train, x_val, y_val, x_test, y_test, train_sizes,
                    dataset_name, dataset_labels):

    dt = dt_task(x_train, y_train, x_val, y_val, x_test, y_test, train_sizes,
                 dataset_name, dataset_labels)
    knn = knn_task(x_train, y_train, x_val, y_val, x_test, y_test, train_sizes,
                   dataset_name, dataset_labels)
    svm_poly = svm_poly_task(x_train, y_train, x_val, y_val, x_test, y_test,
                             train_sizes, dataset_name, dataset_labels)
    svm_rbf = svm_rbf_task(x_train, y_train, x_val, y_val, x_test, y_test,
                           train_sizes, dataset_name, dataset_labels)
    boosting = boosting_task(x_train, y_train, x_val, y_val, x_test, y_test,
                             train_sizes, dataset_name, dataset_labels)
    nn = neural_network_task(x_train, y_train, x_val, y_val, x_test, y_test,
                             train_sizes, dataset_name, dataset_labels)

    return dt, knn, svm_poly, svm_rbf, boosting, nn


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

    args = parser.parse_args()

    flags = TrainFlags(train_dt=args.dt,
                       train_boosting=args.boosting,
                       train_svm=args.svm,
                       train_knn=args.knn,
                       train_nn=args.nn)

    return flags


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    train_flags = parse_args()

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    dataset1()
    dataset2()

import argparse
import logging

from tasks.reduction_and_clustering import run_reduction_and_clustering
from tasks.train_nn import run_reduction_and_nn, run_reduction_clustering_nn
from utils.data import gen_3d_data, get_fashion_mnist_data
from utils.plots import visualize_3d_data, visualize_fashion_mnist, \
    visualize_reduced_dataset3d, visualize_dataset3d_vectors, \
    visualize_fashionmnist_vectors, get_visualize_reduced_fashion_mnist_fn
from tasks.dims_reduction import run_dim_reduction
from tasks.clustering import run_clustering


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',
                        '--n_jobs',
                        default=1,
                        type=int,
                        help='Number of Python processes to use')
    args = parser.parse_args()
    return args.n_jobs


def dataset1(n_jobs: int):
    x1_size = 5
    x2_size = 2
    n_train = 5000
    n_val = 500
    n_test = 500
    noise_prob = 0.01

    x_train, y_train, x_val, y_val, x_test, y_test = gen_3d_data(
        x1_size, x2_size, n_train, n_val, n_test, noise_prob)
    dataset_name = 'Dataset3D'
    run_clustering(dataset_name, x_train, y_train, visualize_3d_data, n_jobs)
    run_dim_reduction(dataset_name,
                      x_train,
                      y_train,
                      visualize_dataset3d_vectors,
                      sync=True,
                      n_jobs=n_jobs)
    run_reduction_and_clustering(dataset_name, x_train, y_train,
                                 visualize_reduced_dataset3d,
                                 visualize_dataset3d_vectors, n_jobs)


def dataset2(n_jobs: int):
    x_train, y_train, x_val, y_val, x_test, y_test = get_fashion_mnist_data()
    dataset_name = 'Fashion-MNIST'

    run_clustering(dataset_name, x_train, y_train, visualize_fashion_mnist,
                   n_jobs)
    run_dim_reduction(dataset_name,
                      x_train,
                      y_train,
                      visualize_fashionmnist_vectors,
                      sync=True,
                      n_jobs=n_jobs)
    run_reduction_and_nn(dataset_name, x_train, y_train, x_val, y_val, n_jobs)
    run_reduction_and_clustering(
        dataset_name, x_train, y_train,
        get_visualize_reduced_fashion_mnist_fn(x_train),
        visualize_fashionmnist_vectors, n_jobs)
    run_reduction_clustering_nn(dataset_name, x_train, y_train, x_val, y_val,
                                n_jobs)


def main():
    n_jobs = parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    dataset1(n_jobs)
    dataset2(n_jobs)


if __name__ == '__main__':
    main()

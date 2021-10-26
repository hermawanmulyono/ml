import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

from utils.data import gen_3d_data, get_fashion_mnist_data
from utils.plots import visualize_3d_data, visualize_fashion_mnist
from utils.tasks import run_clustering, run_dim_reduction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_jobs', default=1,
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
    run_clustering(dataset_name, x_train, visualize_3d_data, n_jobs)
    run_dim_reduction(dataset_name, x_train, y_train, sync=True)


def dataset2(n_jobs: int):
    x_train, y_train, x_val, y_val, x_test, y_test = get_fashion_mnist_data()
    dataset_name = 'Fasihon-MNIST'
    run_clustering(dataset_name, x_train, visualize_fashion_mnist, n_jobs)


def main():
    n_jobs = parse_args()
    dataset1(n_jobs)
    dataset2(n_jobs)


if __name__ == '__main__':
    main()

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

from utils.data import gen_3d_data, get_fashion_mnist_data
from utils.plots import visualize_3d_data, visualize_fashion_mnist
from utils.tasks import run_clustering


def dataset1():
    x1_size = 5
    x2_size = 2
    n_train = 5000
    n_val = 500
    n_test = 500
    noise_prob = 0.01

    x_train, y_train, x_val, y_val, x_test, y_test = gen_3d_data(
        x1_size, x2_size, n_train, n_val, n_test, noise_prob)
    dataset_name = 'Dataset3D'
    run_clustering(dataset_name, x_train, visualize_3d_data)

    # fig = visualize_3d_data(x_train, y_train, ['negative', 'positive'])
    # fig.show()
    #
    # # Need to cache the trained kmeans
    # n_clusters = list(range(2, 17))
    # silhouette_scores = []
    # for n_cluster in n_clusters:
    #     kmeans = KMeans(n_cluster)
    #     cluster_labels = kmeans.fit_predict(x_train)
    #
    #     score = silhouette_score(x_train, cluster_labels)
    #     silhouette_scores.append(score)
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=n_clusters,
    #                          y=silhouette_scores))
    # fig.show()
    #
    # best_n_cluster = n_clusters[int(np.argmax(silhouette_scores))]
    # kmeans = KMeans(best_n_cluster)
    # cluster_labels = kmeans.fit_predict(x_train)
    # categories = [f'cluster_{c}' for c in kmeans.labels_]
    #
    # visualize_3d_data(x_train, cluster_labels, categories).show()


def dataset2():
    x_train, y_train, x_val, y_val, x_test, y_test = get_fashion_mnist_data()
    dataset_name = 'Fasihon-MNIST'
    run_clustering(dataset_name, x_train, visualize_fashion_mnist)


def main():
    dataset1()
    dataset2()


if __name__ == '__main__':
    main()

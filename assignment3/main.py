from utils.data import gen_3d_data
from utils.plots import visualize_3d_data


def dataset1():
    x1_size = 5
    x2_size = 2
    n_train = 5000
    n_val = 500
    n_test = 500
    noise_prob = 0.01

    x_train, y_train, x_val, y_val, x_test, y_test = gen_3d_data(
        x1_size, x2_size, n_train, n_val, n_test, noise_prob)

    fig = visualize_3d_data(x_train, y_train, 'Dataset3D')
    fig.show()


def main():
    dataset1()


if __name__ == '__main__':
    main()

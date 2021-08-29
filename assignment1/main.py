import plotly.graph_objects as go
import numpy as np

from utils.data import gen_2d_data, visualize_2d_data


def main():
    x_train, y_train = gen_2d_data(x1_max=5,
                                   x2_max=2,
                                   num_examples=9000,
                                   noise_prob=0.005)

    visualize_2d_data(x_train, y_train).show()


if __name__ == '__main__':
    main()

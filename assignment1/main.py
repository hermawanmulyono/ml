import plotly.graph_objects as go
import numpy as np

from utils.data import gen_2d_data, visualize_2d_data, \
    visualize_2d_decision_boundary
from utils.models import get_decision_tree


def main():
    x1_max = 5
    x2_max = 2
    n_train = 9000
    n_test = 1000
    noise_prob = 0.005
    x_train, y_train = gen_2d_data(x1_max, x2_max, n_train, noise_prob)

    # visualize_2d_data(x_train, y_train).show()

    dt = get_decision_tree(ccp_alpha=0.000)
    dt.fit(x_train, y_train)

    visualize_2d_decision_boundary(dt, x_train, y_train, x1_max, x2_max).show()

    x_test, y_test = gen_2d_data(x1_max, x2_max, n_test, noise_prob)




if __name__ == '__main__':
    main()

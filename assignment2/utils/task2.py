import logging

import six
import sys

from sklearn.metrics import accuracy_score

sys.modules['sklearn.externals.six'] = six
import mlrose
import plotly.express as px

from utils.data import gen_2d_data


def task2():
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

    hidden_nodes = [16] * 4
    nn_model = mlrose.NeuralNetwork(hidden_nodes,
                                    algorithm='simulated_annealing',
                                    # learning_rate=1e-5,
                                    max_iters=10000,
                                    curve=True)

    logging.info('Training neural network')
    nn_model.fit(x_train, y_train)

    y_pred = nn_model.predict(x_train)
    train_acc = accuracy_score(y_train, y_pred)
    logging.info(f'Train accuracy: {train_acc}')

    y_pred = nn_model.predict(x_val)
    val_acc = accuracy_score(y_val, y_pred)
    logging.info(f'Val accuracy: {val_acc}')

    fitness_curve = nn_model.fitness_curve
    fig = px.line(x=list(range(len(fitness_curve))), y=fitness_curve)
    fig.show()

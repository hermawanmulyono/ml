import logging

from tasks.optimization import run_optimization
from tasks.nn_weights import run_nn_weights


def main():
    logging.basicConfig(level=logging.INFO)

    run_optimization()
    run_nn_weights()


if __name__ == '__main__':
    main()

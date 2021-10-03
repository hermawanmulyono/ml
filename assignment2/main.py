import logging

from utils.task1 import task1
from utils.task2 import task2


def main():
    logging.basicConfig(level=logging.INFO)

    task1()
    task2()


if __name__ == '__main__':
    main()

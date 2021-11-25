import logging

from utils import forest, frozenlake

logging.basicConfig(level=logging.INFO)

forest.run_all()
frozenlake.run_all()

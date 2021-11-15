import copy
import json
import logging
import os
from typing import List, Tuple, Callable, Dict, Iterable

import numpy as np
from mdptoolbox.mdp import ValueIteration, PolicyIteration, MDP, QLearning
from scipy.stats import entropy
import mdptoolbox.example
import joblib

from utils.outputs import table_score, table_joblib


class ParamGridGenerator:

    def __init__(self, param_grid):
        self.param_grid = param_grid

    def generator(self):
        param_names: List[str] = list(self.param_grid.keys())
        grid_index = [0] * len(param_names)
        while True:
            kwargs = {
                key: self.param_grid[key][gi]
                for gi, key in zip(grid_index, param_names)
            }

            yield kwargs

            grid_index = self._increase_grid_index(grid_index)

            if all([gi == 0 for gi in grid_index]):
                break

    def _increase_grid_index(self, grid_index: List[int]):
        """Gets the next set of parameters for grid search

        Note:
            This function assumes the default ordered dictionary
            keys in Python 3.7.

        Args:
            grid_index: Current grid index. Each element
                indicates the corresponding element in
                param_grid.

        Returns:
            The next grid_index.

        """
        param_grid = self.param_grid
        param_names: List[str] = list(param_grid.keys())

        grid_index = copy.deepcopy(grid_index)

        for i in range(len(grid_index)):
            grid_index[i] += 1

            if grid_index[i] >= len(param_grid[param_names[i]]):
                grid_index[i] = 0
            else:
                break

        return grid_index


def run_all():
    task_policy_iteration()
    task_value_iteration()
    task_q_learning()


def task_policy_iteration():
    problem_name = 'forest'
    alg_name = 'policy_iteration'

    param_grid = {
        'S': [5, 10, 100, 1000],
        'r2': [2, 4, 8, 16],
        'p': [0.1, 0.2, 0.3],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [100000]
    }

    def single_policy_iteration(S, r2, p, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r2=r2, p=p)
        pi = PolicyIteration(P, R, discount=discount, max_iter=max_iter)
        pi.run()
        return pi

    _task_template(problem_name, alg_name, param_grid, single_policy_iteration)


def task_value_iteration():
    problem_name = 'forest'
    alg_name = 'value_iteration'

    param_grid = {
        'S': [5, 10, 100, 1000],
        'r2': [2, 4, 8, 16],
        'p': [0.1, 0.2, 0.3],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [100000]
    }

    def single_value_iteration(S, r2, p, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r2=r2, p=p)
        vi = ValueIteration(P, R, discount=discount, max_iter=max_iter)
        vi.run()
        return vi

    _task_template(problem_name, alg_name, param_grid, single_value_iteration)


def task_q_learning():
    problem_name = 'forest'
    alg_name = 'q_learning'

    param_grid = {
        'S': [5, 10, 100, 1000],
        'r2': [2, 4, 8, 16],
        'p': [0.1, 0.2, 0.3],
        'discount': [0.9, 0.99, 0.999],
        'n_iter': [100000]
    }

    def single_value_iteration(S, r2, p, discount, n_iter):
        P, R = mdptoolbox.example.forest(S, r2=r2, p=p)
        vi = QLearning(P, R, discount=discount, n_iter=n_iter)
        vi.run()
        return vi

    _task_template(problem_name, alg_name, param_grid, single_value_iteration)


def _task_template(problem_name: str, alg_name: str,
                   param_grid: Dict[str, Iterable], single_run_fn: Callable):
    joblib_table_path = table_joblib(problem_name, alg_name)

    if not os.path.exists(joblib_table_path):
        param_grid_generator = ParamGridGenerator(param_grid)

        joblib_table: List[Tuple[dict, MDP]] = []

        for kwargs in param_grid_generator.generator():
            logging.info(f'{problem_name} {alg_name} {kwargs}')
            pi = single_run_fn(**kwargs)
            joblib_table.append((kwargs, pi))

        joblib.dump(joblib_table, joblib_table_path)

    joblib_table = joblib.load(joblib_table_path)

    json_table_path = table_score(problem_name, alg_name)

    if not os.path.exists(json_table_path):
        score_table = [(kwargs, {
            'score': eval_mdp(mdp),
            'time': mdp.time,
            'policy': mdp.policy
        }) for kwargs, mdp in joblib_table]

        score_table.sort(key=lambda x: x[1]['score'], reverse=True)

        with open(json_table_path, 'w') as fs:
            json.dump(score_table, fs, indent=4)

    with open(json_table_path, 'r') as fs:
        score_table = json.load(fs)

    return joblib_table, score_table


def eval_mdp(pi: MDP):
    policy = pi.policy
    actions = set(policy)
    counts = {a: 0 for a in actions}
    for a in policy:
        counts[a] += 1
    e = entropy(list(counts.values()))

    return e

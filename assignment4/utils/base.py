import copy
import json
import logging
import os
from typing import List, Dict, Iterable, Callable, Tuple

import joblib
from mdptoolbox.mdp import MDP

from utils.outputs import table_joblib, table_score


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


def task_template(problem_name: str,
                  alg_name: str,
                  param_grid: Dict[str, Iterable],
                  single_run_fn: Callable,
                  eval_mdp: Callable[..., float]):
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
            'score': eval_mdp(mdp, **kwargs),
            'time': mdp.time,
            'policy': mdp.policy
        }) for kwargs, mdp in joblib_table]

        score_table.sort(key=lambda x: x[1]['score'], reverse=True)

        with open(json_table_path, 'w') as fs:
            json.dump(score_table, fs, indent=4)

    with open(json_table_path, 'r') as fs:
        score_table = json.load(fs)

    return joblib_table, score_table

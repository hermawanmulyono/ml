import copy
import json
import logging
import os
from typing import List, Dict, Iterable, Callable, Tuple, Any

import joblib
from matplotlib import pyplot as plt
from mdptoolbox.mdp import MDP

from utils.outputs import table_joblib, table_score


class ParamGridGenerator:

    def __init__(self, param_grid):
        if type(param_grid) == dict:
            param_grid = [param_grid]
        self.param_grids = param_grid

    def generator(self):
        for param_grid in self.param_grids:
            param_names: List[str] = list(param_grid.keys())
            grid_index = [0] * len(param_names)
            while True:
                kwargs = {
                    key: param_grid[key][gi]
                    for gi, key in zip(grid_index, param_names)
                }

                yield kwargs

                grid_index = self._increase_grid_index(grid_index, param_grid)

                if all([gi == 0 for gi in grid_index]):
                    break

    @classmethod
    def _increase_grid_index(cls, grid_index: List[int], param_grid):
        """Gets the next set of parameters for grid search

        Note:
            This function assumes the default ordered dictionary
            keys in Python 3.7.

        Args:
            grid_index: Current grid index. Each element
                indicates the corresponding element in
                param_grid.
            param_grid: parameter grid dictionary of interest

        Returns:
            The next grid_index.

        """
        param_grid = param_grid
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
                  eval_mdp: Callable[..., float],
                  group_problems_by: List[str]):

    joblib_table_path = table_joblib(problem_name, alg_name)

    if not os.path.exists(joblib_table_path):
        param_grid_generator = ParamGridGenerator(param_grid)

        joblib_table: List[Tuple[dict, MDP]] = []

        for kwargs in param_grid_generator.generator():
            logging.info(f'{problem_name} {alg_name} {kwargs}')
            pi = single_run_fn(**kwargs)
            joblib_table.append((kwargs, pi))

            # For debugging frozen lake
            # from utils.frozenlake import plot_policy
            # plot_policy(pi.policy, kwargs['size'], kwargs['p'], 'eval.png')
            # plt.show()

            # For debugging forest
            plt.figure(figsize=(6, 2))
            x = list(range(len(pi.policy)))
            plt.plot(x, pi.policy)
            plt.show()

        joblib.dump(joblib_table, joblib_table_path)

    joblib_table = joblib.load(joblib_table_path)

    # all_json_tables has the following structure
    # {..., 'problem_name_string': [(kwargs, metrics)] ,...}
    all_json_tables: Dict[str, list] = {}

    for kwargs, mdp in joblib_table:
        # Construct problem name 'problem_name_param1_val1_param2_val2'
        params_to_append = [(param_name, kwargs[param_name]) for param_name
                            in group_problems_by]
        problem_name_with_params = append_problem_name(problem_name, params_to_append)

        if problem_name_with_params not in all_json_tables:
            all_json_tables[problem_name_with_params] = []

        all_json_tables[problem_name_with_params].append((kwargs, mdp))

    json_table_path = table_score(problem_name, alg_name)

    if not os.path.exists(json_table_path):
        all_score_tables = {}
        for problem_name_with_params, table in all_json_tables.items():
            score_table = [(kwargs, {
                'score': eval_mdp(mdp, **kwargs),
                'time': mdp.time,
                'policy': mdp.policy,
                'iter': mdp.iter if hasattr(mdp, 'iter') else mdp.max_iter
            }) for kwargs, mdp in table]

            score_table.sort(key=lambda x: x[1]['score'], reverse=True)

            all_score_tables[problem_name_with_params] = score_table

        with open(json_table_path, 'w') as fs:
            json.dump(all_score_tables, fs, indent=4)

    with open(json_table_path, 'r') as fs:
        all_score_tables: Dict[str, list] = json.load(fs)

    return joblib_table, all_score_tables


def append_problem_name(problem_name: str, params: List[Tuple[str, Any]]):
    problem_name_with_params = problem_name
    for param_name, val in params:
        problem_name_with_params += f'_{param_name}_{val}'
    return problem_name_with_params




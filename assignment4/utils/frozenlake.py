import math
import os

import gym
import joblib
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.wrappers import TimeLimit
from mdptoolbox.mdp import ValueIteration, PolicyIteration, MDP
import matplotlib.pyplot as plt

from utils.base import task_template, append_problem_name
from utils.outputs import frozen_lake_map, frozen_lake_policy_path
from utils.algs import QLearning


def run_all():
    task_policy_iteration()
    task_value_iteration()
    task_q_learning()


def get_pr_matrices(env: TimeLimit):
    num_states = env.env.nS
    num_actions = env.env.nA

    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_actions, num_states, num_states))

    gym_p: dict = env.env.P
    for s0, transition in gym_p.items():
        for action, next_states in transition.items():
            for next_state in next_states:
                prob, s1, reward, is_terminal = next_state

                P[action, s0, s1] = prob
                R[action, s0, s1] = reward

    for action in range(num_actions):
        P[action] = P[action] / P[action].sum(axis=1).reshape((-1, 1))

    return P, R


def plot_policy(policy, size, p, filename: str):
    # state = row * ncol + col
    policy = np.array(policy).reshape((size, size))

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    # [..., (dx, dy), ...]
    directions = [(-2 / 3, 0), (0, -2 / 3), (2 / 3, 0), (0, 2 / 3)]

    desc = get_map(size, p)

    plt.figure(figsize=(size / 2, size / 2))

    for row in range(size):
        for col in range(size):
            x0 = col + 0.5
            y0 = (size - (row + 0.5))
            if desc[row, col] == b'H':
                plt.text(x0, y0, 'H')
            elif desc[row, col] == b'S':
                plt.text(x0, y0, 'S')
            elif desc[row, col] == b'G':
                plt.text(x0, y0, 'G')
            else:
                action_index = policy[row, col]
                action = directions[action_index]
                x = col + 0.5 - action[0] / 2
                y = (size - (row + 0.5)) - action[1] / 2
                dx, dy = action
                plt.arrow(x, y, dx, dy, head_width=0.1)

    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.xticks(list(range(size + 1)), visible=False)
    plt.yticks(list(range(size + 1)), visible=False)
    plt.grid()
    # plt.show()
    plt.savefig(filename)


def get_frozen_lake(size: int, p: float, is_slippery: bool):
    desc = get_map(size, p)
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery)
    return env


def eval_mdp(mdp: MDP, size: int, p: float, is_slippery: bool, repeats=1000):
    env: TimeLimit = get_frozen_lake(size, p, is_slippery)
    policy = mdp.policy

    steps = []

    for _ in range(repeats):
        state = env.reset()
        step = 0
        max_steps = 1000
        while step < max_steps:
            action = policy[state]
            step += 1
            state, reward, done, info = env.step(action)
            # env.render()

            if done:
                if reward > 0:
                    break
                else:
                    state = env.reset()
                    continue

        steps.append(step)

    print(f'{float(np.mean(steps))}')
    return 1.0 / float(np.mean(steps))


def get_map(size: int, p: float) -> np.ndarray:
    path_to_map = frozen_lake_map(size, p)

    if not os.path.exists(path_to_map):
        lake_map = generate_random_map(size, p)
        lake_map = np.asarray(lake_map, dtype='c')

        joblib.dump(lake_map, path_to_map)

    lake_map = joblib.load(path_to_map)
    return lake_map


def get_param_grid():
    return {
        'size': [4, 6, 8],
        'p': [0.9, 0.7, 0.5],
        'is_slippery': [True, False],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [100000]
    }


def task_policy_iteration():
    problem_name = 'frozenlake'
    alg_name = 'policy_iteration'

    param_grid = get_param_grid()

    group_problems_by = ['size', 'p', 'is_slippery']

    def single_policy_iteration(size, p, is_slippery, discount, max_iter):
        env = get_frozen_lake(size, p, is_slippery)
        P, R = get_pr_matrices(env)
        pi = PolicyIteration(P, R, discount=discount, max_iter=max_iter)
        pi.run()
        return pi

    def eval_fn(mdp, size, p, is_slippery, **kwargs):
        return eval_mdp(mdp, size, p, is_slippery)

    _, all_scores_tables = task_template(problem_name, alg_name, param_grid,
                                         single_policy_iteration, eval_fn,
                                         group_problems_by)

    print('Varying size')
    for size in param_grid['size']:
        p = 0.7
        is_slippery = False
        discount = 0.9
        max_iter = param_grid['max_iter'][0]

        params_to_append = [('size', size), ('p', p),
                            ('is_slippery', is_slippery)]
        problem_name_with_params = append_problem_name(problem_name,
                                                       params_to_append)
        table = all_scores_tables[problem_name_with_params]
        kwargs = {
            'size': size,
            'p': p,
            'is_slippery': is_slippery,
            'discount': discount,
            'max_iter': max_iter
        }

        fig_file_path = frozen_lake_policy_path(problem_name_with_params)

        # Linear search
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)
                plot_policy(results['policy'], kwargs['size'], kwargs['p'],
                            fig_file_path)

    print('Varying p')
    for p in param_grid['p']:
        size = 8
        is_slippery = False
        discount = 0.9

        max_iter = param_grid['max_iter'][0]

        params_to_append = [('size', size), ('p', p),
                            ('is_slippery', is_slippery)]
        problem_name_with_params = append_problem_name(problem_name,
                                                       params_to_append)
        table = all_scores_tables[problem_name_with_params]
        kwargs = {
            'size': size,
            'p': p,
            'is_slippery': is_slippery,
            'discount': discount,
            'max_iter': max_iter
        }

        fig_file_path = frozen_lake_policy_path(problem_name_with_params)

        # Linear search
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)
                plot_policy(results['policy'], kwargs['size'], kwargs['p'],
                            fig_file_path)

    print('Varying is_slippery')
    for is_slippery in param_grid['is_slippery']:
        size = 8
        p = 0.7
        discount = 0.9
        max_iter = param_grid['max_iter'][0]

        params_to_append = [('size', size), ('p', p),
                            ('is_slippery', is_slippery)]
        problem_name_with_params = append_problem_name(problem_name,
                                                       params_to_append)
        table = all_scores_tables[problem_name_with_params]
        kwargs = {
            'size': size,
            'p': p,
            'is_slippery': is_slippery,
            'discount': discount,
            'max_iter': max_iter
        }

        fig_file_path = frozen_lake_policy_path(problem_name_with_params)

        # Linear search
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)
                plot_policy(results['policy'], kwargs['size'], kwargs['p'],
                            fig_file_path)


def task_value_iteration():
    problem_name = 'frozenlake'
    alg_name = 'value_iteration'

    param_grid = get_param_grid()

    group_problems_by = ['size', 'p', 'is_slippery']

    def single_value_iteration(size, p, is_slippery, discount, max_iter):
        env = get_frozen_lake(size, p, is_slippery)
        P, R = get_pr_matrices(env)
        vi = ValueIteration(P, R, discount=discount, max_iter=max_iter)
        vi.run()
        return vi

    def eval_fn(mdp, size, p, is_slippery, **kwargs):
        return eval_mdp(mdp, size, p, is_slippery)

    task_template(problem_name, alg_name, param_grid, single_value_iteration,
                  eval_fn, group_problems_by)


def epsilon_schedule(n):
    return 0.5


def learning_rate_schedule(n):
    lr = 1 / math.log(n + math.exp(0))
    return max(lr, 1E-2)


def task_q_learning():
    problem_name = 'frozenlake'
    alg_name = 'q_learning'

    param_grid = {
        'size': [4, 6, 8],
        'p': [0.9, 0.7, 0.5],
        'is_slippery': [False, True],
        'discount': [0.9, 0.99, 0.999],
        'n_iter': [300000]  # or 300000 if using small states <= 8
    }

    group_problems_by = ['size', 'p', 'is_slippery']

    def single_value_iteration(size, p, is_slippery, discount, n_iter):
        env = get_frozen_lake(size, p, is_slippery)
        P, R = get_pr_matrices(env)
        vi = QLearning(P,
                       R,
                       discount=discount,
                       n_iter=n_iter,
                       restart_steps=100,
                       learning_rate_schedule=learning_rate_schedule,
                       epsilon_schedule=epsilon_schedule)
        vi.run()
        return vi

    def eval_fn(mdp, size, p, is_slippery, **kwargs):
        return eval_mdp(mdp, size, p, is_slippery)

    task_template(problem_name, alg_name, param_grid, single_value_iteration,
                  eval_fn, group_problems_by)

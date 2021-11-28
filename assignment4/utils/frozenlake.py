import copy
import math
import os

import gym
import joblib
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.wrappers import TimeLimit
from mdptoolbox.mdp import MDP
import matplotlib.pyplot as plt

from utils.base import task_template, append_problem_name
from utils.outputs import frozen_lake_map, frozen_lake_policy_path, \
    convergence_plot, frozen_lake_map_png
from utils.algs import PolicyIteration, ValueIteration, QLearning


def run_all():
    # plot size = 20, p = 0.9
    for size in [8, 20]:
        p = 0.9
        plot_grid_map(size, p, frozen_lake_map_png(size, p))

    joblib_table1, all_scores_table1 = task_policy_iteration()
    joblib_table2, all_scores_table2 = task_value_iteration()

    compare_joblib_tables(joblib_table1, joblib_table2)

    joblib_table3, all_scores_table3 = task_q_learning()

    from utils.forest import print_wall_time
    print('Printing PI time')
    print_wall_time(joblib_table1)
    print('Printing VI time')
    print_wall_time(joblib_table2)
    print('Printing Q-Learning time')
    print_wall_time(joblib_table3)


def compare_joblib_tables(joblib_table1, joblib_table2):
    n_experiments = 0
    n_policy_same = 0
    n_iter1_smaller = 0
    n_iter1_faster = 0
    n_almost_equal = 0

    table1 = copy.deepcopy(joblib_table1)
    table2 = copy.deepcopy(joblib_table2)

    table1.sort(key=lambda x: str(x[0]))
    table2.sort(key=lambda x: str(x[0]))

    for (key1, mdp1), (key2, mdp2) in zip(table1, table2):
        assert key1 == key2

        n_experiments += 1

        if mdp1.policy == mdp2.policy:
            n_policy_same += 1

        if mdp1.iter <= mdp2.iter:
            n_iter1_smaller += 1

        if mdp1.time <= mdp2.time:
            n_iter1_faster += 1

        if np.allclose(mdp1.V, mdp2.V, rtol=0.1):
            n_almost_equal += 1

    print(f'n_experiments {n_experiments}')
    print(f'n_policy_same {n_policy_same}')
    print(f'n_iter1_smaller {n_iter1_smaller}')
    print(f'n_iter1_faster {n_iter1_faster}')
    print(f'n_almost_equal {n_almost_equal}')

    # Find the best of of PI and VI, then plot their policies
    sizes1 = {kwargs['size'] for kwargs, mdp in table1}
    sizes2 = {kwargs['size'] for kwargs, mdp in table2}
    all_sizes = sizes1.union(sizes2)

    for size in all_sizes:
        for mode in ['min', 'max']:
            # Find best of table 1
            plt.close('all')
            relevant_table1 = [entry for entry in table1 if entry[0]['size'] ==
                               size]
            scores1 = [np.mean(mdp.V) for kwargs, mdp in relevant_table1]

            if mode == 'min':
                index = np.argmin(scores1)
            else:
                index = np.argmax(scores1)
            kwargs1, mdp1 = relevant_table1[index]
            p = kwargs1['p']

            for kwargs, mdp in table2:
                if kwargs == kwargs1:
                    plt.close('all')
                    filename = frozen_lake_policy_path(f'frozenlake_{mode}_vi_pi_{size}')
                    plot_policy(mdp1.policy, size, p, filename, mdp.policy)

                    print(f'Mode {mode} {scores1[index]} vs {np.mean(mdp.V)}')
                    print(kwargs)

                    break

        # relevant_table2 = [entry for entry in table2 if entry[0]['size'] ==
        #                    size]
        # scores2 = [np.mean(mdp.V) for kwargs, mdp in relevant_table2]
        # argmax2 = np.argmax(scores2)
        # kwargs2, mdp2 = relevant_table1[argmax2]


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


def plot_grid_map(size, p, filename: str):
    desc = get_map(size, p)
    plt.figure(figsize=(size / 2, size / 2))
    for row in range(size):
        for col in range(size):
            x0 = col + 0.5
            y0 = (size - (row + 0.5))

            if desc[row, col] == b'S':
                plt.text(x0, y0, 'S')

            if desc[row, col] == b'H':
                plt.text(x0, y0, 'H')
            elif desc[row, col] == b'G':
                plt.text(x0, y0, 'G')

    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.xticks(list(range(size + 1)), visible=False)
    plt.yticks(list(range(size + 1)), visible=False)
    plt.grid()
    plt.savefig(filename)


def plot_policy(policy, size, p, filename: str, policy2=None):
    # state = row * ncol + col
    policy = np.array(policy).reshape((size, size))

    if policy2 is not None:
        policy2 = np.array(policy2).reshape((size, size))

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

            if desc[row, col] == b'S':
                plt.text(x0, y0, 'S')

            if desc[row, col] == b'H':
                plt.text(x0, y0, 'H')
            elif desc[row, col] == b'G':
                plt.text(x0, y0, 'G')
            else:
                action_index = policy[row, col]
                action = directions[action_index]
                x = col + 0.5 - action[0] / 2
                y = (size - (row + 0.5)) - action[1] / 2
                dx, dy = action
                plt.arrow(x, y, dx, dy, head_width=0.1)

                if (policy2 is not None) and (policy2[row, col] !=
                                              action_index):
                    plt.arrow(x, y, dx, dy, head_width=0.1, color='blue')

                    action_index = policy2[row, col]
                    action = directions[action_index]
                    x = col + 0.5 - action[0] / 2
                    y = (size - (row + 0.5)) - action[1] / 2
                    dx, dy = action
                    plt.arrow(x, y, dx, dy, head_width=0.1, color='red')

                else:
                    plt.arrow(x, y, dx, dy, head_width=0.1)


    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.xticks(list(range(size + 1)), visible=False)
    plt.yticks(list(range(size + 1)), visible=False)
    plt.grid()
    # plt.show()
    plt.savefig(filename)


def plot_value(V, size, p, filename: str):
    V = np.array(V).reshape((size, size))

    desc = get_map(size, p)

    for row in range(size):
        for col in range(size):
            x0 = col + 0.5
            y0 = (size - (row + 0.5))

            if desc[row, col] == b'S':
                plt.text(x0, y0, 'S')
            if desc[row, col] == b'H':
                plt.text(x0, y0, 'H')
            elif desc[row, col] == b'G':
                plt.text(x0, y0, 'G')

            plt.text(x0, y0, f'{np.around(V[row, col], decimals=3)}')

    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.xticks(list(range(size + 1)), visible=False)
    plt.yticks(list(range(size + 1)), visible=False)
    plt.grid()
    plt.show()
    plt.savefig(filename)


def get_frozen_lake(size: int, p: float, is_slippery: bool):
    desc = get_map(size, p)
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery)
    return env


def eval_mdp(mdp: MDP, size: int, p: float, is_slippery: bool, repeats=300):
    return np.mean(mdp.V)
    env: TimeLimit = get_frozen_lake(size, p, is_slippery)
    policy = mdp.policy

    steps = []

    for r in range(repeats):
        if r % 10 == 0:
            print(f'Evaluating {r}')
        state = env.reset()
        step = 0
        max_steps = 1E10
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
    return [{
        'size': [8],
        'p': [0.9],
        'is_slippery': [True],
        'epsilon': [0.1, 0.01, 0.001, 0.0001],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [1000]
    }, {
        'size': [20],
        'p': [0.9],
        'is_slippery': [True],
        'epsilon': [0.1, 0.01, 0.001, 0.0001],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [10000]
    }]


def task_policy_iteration():
    problem_name = 'frozenlake'
    alg_name = 'policy_iteration'

    param_grid = get_param_grid()

    group_problems_by = ['size', 'p', 'is_slippery']

    def single_policy_iteration(size, p, is_slippery, epsilon, discount, \
                                                             max_iter):
        env = get_frozen_lake(size, p, is_slippery)
        P, R = get_pr_matrices(env)
        pi = PolicyIteration(P,
                             R,
                             epsilon=epsilon,
                             discount=discount,
                             max_iter=max_iter)
        pi.run()
        return pi

    def eval_fn(mdp, size, p, is_slippery, **kwargs):
        return eval_mdp(mdp, size, p, is_slippery)

    joblib_table, all_scores_tables = task_template(problem_name, alg_name,
                                                    param_grid,
                                                    single_policy_iteration,
                                                    eval_fn, group_problems_by)

    generate_convergence_plots(joblib_table, problem_name, alg_name)

    return joblib_table, all_scores_tables


def generate_convergence_plots(joblib_table, problem_name: str, alg_name: str):

    for grid in get_param_grid():
        plt.figure()

        for epsilon in grid['epsilon'][::-1]:
            size = grid['size'][0]
            p = grid['p'][0]
            is_slippery = grid['is_slippery'][0]
            discount = 0.99
            max_iter = grid['max_iter'][0]

            kwargs = {
                'size': size,
                'p': p,
                'is_slippery': is_slippery,
                'discount': discount,
                'max_iter': max_iter,
                'epsilon': epsilon
            }

            pi: PolicyIteration
            evaluations = None
            for kwargs_, pi in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = pi.evaluations
                    break

            assert evaluations is not None

            steps, vmean = zip(*evaluations)
            print(evaluations)
            plt.plot(steps, vmean, label=f'{epsilon}')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')

        size = grid['size'][0]
        problem_name_w_size = f'{problem_name}_{size}'
        filename = convergence_plot(problem_name_w_size, alg_name, 'epsilon')
        plt.savefig(filename)

        plt.figure()
        for discount in grid['discount'][::-1]:
            epsilon = 0.001
            size = grid['size'][0]
            p = grid['p'][0]
            is_slippery = grid['is_slippery'][0]
            max_iter = grid['max_iter'][0]

            kwargs = {
                'size': size,
                'p': p,
                'is_slippery': is_slippery,
                'discount': discount,
                'max_iter': max_iter,
                'epsilon': epsilon
            }

            pi: PolicyIteration
            evaluations = None
            for kwargs_, pi in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = pi.evaluations
                    break

            assert evaluations is not None

            steps, vmean = zip(*evaluations)
            print(evaluations)
            plt.plot(steps, vmean, label=f'{discount}')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')

        filename = convergence_plot(problem_name_w_size, alg_name, 'gamma')
        plt.savefig(filename)


def task_value_iteration():
    problem_name = 'frozenlake'
    alg_name = 'value_iteration'

    param_grid = get_param_grid()

    group_problems_by = ['size', 'p', 'is_slippery']

    def single_value_iteration(size, p, is_slippery, epsilon, discount, \
                                                             max_iter):
        env = get_frozen_lake(size, p, is_slippery)
        P, R = get_pr_matrices(env)
        vi = ValueIteration(P,
                            R,
                            epsilon=epsilon,
                            discount=discount,
                            max_iter=max_iter)
        vi.run()
        return vi

    def eval_fn(mdp, size, p, is_slippery, **kwargs):
        return eval_mdp(mdp, size, p, is_slippery)

    joblib_table, all_scores_tables = task_template(problem_name, alg_name, param_grid,
                                    single_value_iteration, eval_fn,
                                    group_problems_by)

    generate_convergence_plots(joblib_table, problem_name, alg_name)

    return joblib_table, all_scores_tables


def epsilon_schedule(n):
    e = max(0.1, 1 / (1 + n / 1E7))
    if n % 10000 == 0:
        print(f'e = {e}')
    return e


def learning_rate_schedule(n):
    lr = 1 / math.log(n + math.exp(0))
    return max(lr, 1E-3)


def task_q_learning():
    problem_name = 'frozenlake'
    alg_name = 'q_learning'

    # param_grid = {
    #     'size': [16, 4],
    #     'p': [0.8],
    #     'is_slippery': [True],
    #     'discount': [0.9, 0.99, 0.999],
    #     'n_iter': [3000000]  # or 300000 if using small states <= 8
    # }

    param_grid = [{
        'size': [8],
        'p': [0.9],
        'is_slippery': [True],
        'discount': [0.9, 0.99, 0.999],
        'epsilon_schedule': [0.9, epsilon_schedule, None],
        'learning_rate_schedule': [0.1, learning_rate_schedule, None],
        'n_iter': [1000000]
    }, {
        'size': [20],
        'p': [0.9],
        'is_slippery': [True],
        'discount': [0.9, 0.99, 0.999],
        'epsilon_schedule': [0.9, epsilon_schedule, None],
        'learning_rate_schedule': [0.1, learning_rate_schedule, None],
        'n_iter': [5000000]
    }][::-1]

    group_problems_by = ['size', 'p', 'is_slippery']

    def single_value_iteration(size, p, is_slippery, epsilon_schedule,
                               learning_rate_schedule, discount, n_iter):
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

    joblib_table, all_scores_table = task_template(problem_name, alg_name,
                                              param_grid,
                                    single_value_iteration, eval_fn,
                                    group_problems_by)

    for grid in param_grid:
        plt.close('all')
        plt.figure()

        size = grid['size'][0]
        p = grid['p'][0]
        is_slippery = grid['is_slippery'][0]
        n_iter = grid['n_iter'][0]
        discount = grid['discount'][:2][-1]

        ####################################################
        # Epsilon schedule
        ####################################################

        for epsilon_schedule_ in grid['epsilon_schedule']:
            kwargs = {
                'size': size,
                'p': p,
                'is_slippery': is_slippery,
                'discount': discount,
                'learning_rate_schedule': learning_rate_schedule,
                'epsilon_schedule': epsilon_schedule_,
                'n_iter': n_iter
            }

            if callable(epsilon_schedule_):
                eps_sch_string = 'custom_eps_schedule'
            elif epsilon_schedule_ is None:
                eps_sch_string = 'default'
            else:
                eps_sch_string = f'{epsilon_schedule_}'

            problem_name_w_params = f'{problem_name}_{size}_epsilon_' \
                                  f'schedule_{eps_sch_string}'
            ql: QLearning
            evaluations = None
            for kwargs_, ql in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = ql.evaluations

                    if size == 8:
                        if epsilon_schedule_ == epsilon_schedule:
                            print(f'With {kwargs} {ql.evaluations[-1]}')
                            p1 = ql.policy
                        elif epsilon_schedule_ is None:
                            print(f'With {kwargs} {ql.evaluations[-1]}')
                            p2 = ql.policy

                    policy_file_name = frozen_lake_policy_path(problem_name_w_params)
                    # plot_policy(ql.policy, size, p, policy_file_name)
                    break
            assert evaluations is not None

            steps, vmean = zip(*evaluations)
            plt.plot(steps, vmean, label=eps_sch_string)

        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')

        problem_name_w_size = f'{problem_name}_{size}'
        filename = convergence_plot(problem_name_w_size, alg_name,
                                    'epsilon_schedule')
        plt.savefig(filename)

        plt.close('all')

        if size == 8:
            policy_file_name = frozen_lake_policy_path('frozen_lake_8_ql_comparison')
            plot_policy(p1, 8, 0.9, policy_file_name, p2)

        ####################################################
        # Learning rate schedule
        ####################################################

        plt.close('all')
        plt.figure()

        for learning_rate_schedule_ in grid['learning_rate_schedule']:
            kwargs = {
                'size': size,
                'p': p,
                'is_slippery': is_slippery,
                'discount': discount,
                'learning_rate_schedule': learning_rate_schedule_,
                'epsilon_schedule': epsilon_schedule,
                'n_iter': n_iter
            }

            if callable(learning_rate_schedule_):
                lr_sch_string = 'custom_lr_schedule'
            elif learning_rate_schedule_ is None:
                lr_sch_string = 'default'
            else:
                lr_sch_string = f'{learning_rate_schedule_}'

            problem_name_w_params = f'{problem_name}_{size}_learning_rate_' \
                                    f'schedule_{lr_sch_string}'
            ql: QLearning
            evaluations = None
            for kwargs_, ql in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = ql.evaluations

                    if size == 20:
                        if learning_rate_schedule_ == learning_rate_schedule:
                            print(f'With {kwargs} {ql.evaluations[-1]}')
                            p1 = ql.policy
                        elif learning_rate_schedule_ is None:
                            print(f'With {kwargs} {ql.evaluations[-1]}')
                            p2 = ql.policy

                    policy_file_name = frozen_lake_policy_path(
                        problem_name_w_params)
                    # plot_policy(ql.policy, size, p, policy_file_name)
                    break
            assert evaluations is not None

            steps, vmean = zip(*evaluations)
            plt.plot(steps, vmean, label=lr_sch_string)

        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')

        problem_name_w_size = f'{problem_name}_{size}'
        filename = convergence_plot(problem_name_w_size, alg_name,
                                    'learning_rate')
        plt.savefig(filename)

        if size == 20:
            policy_file_name = frozen_lake_policy_path(
                'frozen_lake_20_ql_comparison')
            plot_policy(p1, size, 0.9, policy_file_name, p2)

    return joblib_table, all_scores_table


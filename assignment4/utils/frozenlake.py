import os

import gym
import joblib
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.wrappers import TimeLimit
from mdptoolbox.mdp import ValueIteration, PolicyIteration, MDP, QLearning

from utils.base import task_template
from utils.outputs import frozen_lake_map


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


def get_frozen_lake(size: int, p: float):
    desc = get_map(size, p)
    env = gym.make('FrozenLake-v1', desc=desc)
    return env


def eval_mdp(mdp: MDP, size: int, p: float, repeats=1000):
    env: TimeLimit = get_frozen_lake(size, p)
    policy = mdp.policy

    steps = []

    for _ in range(repeats):
        state = env.reset()
        step = 0
        while step < 10000:
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


def task_policy_iteration():
    problem_name = 'frozenlake'
    alg_name = 'policy_iteration'

    param_grid = {
        'size': [16, 18, 20],
        'p': [0.9, 0.7, 0.5],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [10000]
    }

    def single_policy_iteration(size, p, discount, max_iter):
        env = get_frozen_lake(size, p)
        P, R = get_pr_matrices(env)
        pi = PolicyIteration(P, R, discount=discount, max_iter=max_iter)
        pi.run()
        return pi

    def eval_fn(mdp, size, p, **kwargs):
        return eval_mdp(mdp, size, p)

    task_template(problem_name, alg_name, param_grid, single_policy_iteration,
                  eval_fn)


def task_value_iteration():
    problem_name = 'frozenlake'
    alg_name = 'value_iteration'

    param_grid = {
        'size': [16, 18, 20],
        'p': [0.9, 0.7, 0.5],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [10000]
    }

    def single_value_iteration(size, p, discount, max_iter):
        env = get_frozen_lake(size, p)
        P, R = get_pr_matrices(env)
        vi = ValueIteration(P, R, discount=discount, max_iter=max_iter)
        vi.run()
        return vi

    def eval_fn(mdp, size, p, **kwargs):
        return eval_mdp(mdp, size, p)

    task_template(problem_name, alg_name, param_grid, single_value_iteration,
                  eval_fn)


def task_q_learning():
    problem_name = 'forest'
    alg_name = 'q_learning'

    param_grid = {
        'size': [16, 18, 20],
        'p': [0.9, 0.7, 0.5],
        'discount': [0.9, 0.99, 0.999],
        'n_iter': [10000]
    }

    def single_value_iteration(size, p, discount, n_iter):
        env = get_frozen_lake(size, p)
        P, R = get_pr_matrices(env)
        vi = QLearning(P, R, discount=discount, n_iter=n_iter)
        vi.run()
        return vi

    task_template(problem_name, alg_name, param_grid, single_value_iteration,
                  eval_mdp)


if __name__ == '__main__':
    pass
    # param_grid = {'discount': [0.9, 0.99], 'max_iter': [10000]}

    # def single_run(discount, max_iter):
    #     env = gym.make('FrozenLake-v1')
    #     P, R = get_pr_matrices(env)
    #     vi = ValueIteration(P, R, discount=discount, max_iter=max_iter)
    #     vi.run()
    #     return vi
    #
    # joblib_table, score_table = task_template('frozen-lake', 'value_iteration',
    #                                           param_grid, single_run)
    # vi = joblib_table[0][1]
    #
    # env_: TimeLimit = gym.make('FrozenLake-v1')
    # state = env_.reset()
    # policy = vi.policy
    #
    # while True:
    #     action = policy[state]
    #     observation, reward, done, info = env_.step(action)
    #     env_.render()
    #
    #     if done:
    #         break
    #
    #     state = observation

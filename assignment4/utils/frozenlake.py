import gym
import numpy as np
from gym.wrappers import TimeLimit
from mdptoolbox.mdp import ValueIteration

from utils.forest import _task_template


class FrozenLakeAdapter:
    def __init__(self):
        self.env: TimeLimit = gym.make('FrozenLake-v1')

    @property
    def mdp_matrices(self):
        num_states = self.env.env.nS
        num_actions = self.env.env.nA

        P = np.zeros((num_actions, num_states, num_states))
        R = np.zeros((num_actions, num_states, num_states))

        gym_p: dict = self.env.env.P
        for s0, transition in gym_p.items():
            for action, next_states in transition.items():
                for next_state in next_states:
                    prob, s1, reward, is_terminal = next_state

                    P[action, s0, s1] = prob
                    R[action, s0, s1] = reward

        return P, R


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


if __name__ == '__main__':
    param_grid = {
        'discount': [0.9, 0.99],
        'max_iter': [10000]
    }

    def single_run(discount, max_iter):
        env = gym.make('FrozenLake-v1')
        P, R = get_pr_matrices(env)
        vi = ValueIteration(P, R, discount=discount, max_iter=max_iter)
        vi.run()
        return vi

    joblib_table, score_table = _task_template('frozen-lake',
                                             'value_iteration', param_grid, single_run)
    vi = joblib_table[0][1]

    env: TimeLimit = gym.make('FrozenLake-v1')
    state = env.reset()
    policy = vi.policy

    while True:
        action = policy[state]
        observation, reward, done, info = env.step(action)
        env.render()

        if done:
            break

        state = observation

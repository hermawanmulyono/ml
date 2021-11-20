import time as _time
import math as _math

import numpy as _np
from mdptoolbox.mdp import QLearning as MDPQLearning


def default_epsilon_schedule(n):
    return 1 / _math.log(n + 2)


def default_learning_rate_schedule(n):
    return 1 / _math.sqrt(n + 2)


class ConstantFunction:
    """Pickleable and callable constant function"""

    def __init__(self, value):
        self.value = value

    def __call__(self, n):
        return self.value


class QLearning(MDPQLearning):

    def __init__(self,
                 transitions,
                 reward,
                 discount,
                 n_iter=10000,
                 restart_steps=100,
                 epsilon_schedule=None,
                 learning_rate_schedule=None):
        super().__init__(transitions, reward, discount, n_iter)

        self.restart_steps = restart_steps

        if epsilon_schedule is None:
            self.epsilon_schedule = default_epsilon_schedule
        elif type(epsilon_schedule) in [int, float]:
            self.epsilon_schedule = ConstantFunction(epsilon_schedule)
        else:
            self.epsilon_schedule = epsilon_schedule

        if learning_rate_schedule is None:
            self.learning_rate_schedule = default_learning_rate_schedule
        elif type(learning_rate_schedule) in [int, float]:
            self.learning_rate_schedule = ConstantFunction(
                learning_rate_schedule)
        else:
            self.learning_rate_schedule = learning_rate_schedule

    def run(self):
        # Run the Q-learning algoritm.
        discrepancy = []

        self.time = _time.time()

        # initial state choice
        s = _np.random.randint(0, self.S)

        for n in range(1, self.max_iter + 1):

            # Reinitialisation of trajectories every 100 transitions
            if (n % self.restart_steps) == 0:
                s = _np.random.randint(0, self.S)

            # Action choice : greedy with increasing probability
            pn = _np.random.random()
            epsilon = self.epsilon_schedule(n)
            if pn < (1 - epsilon):
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()
            else:
                a = _np.random.randint(0, self.A)

            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = _np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            # Updating the value of Q
            # Decaying update coefficient (1/sqrt(n+2)) can be changed
            delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            dQ = self.learning_rate_schedule(n) * delta
            self.Q[s, a] = self.Q[s, a] + dQ

            # current state is updated
            s = s_new

            # Computing and saving maximal values of the Q variation
            discrepancy.append(_np.absolute(dQ))

            # Computing means all over maximal Q variations values
            if len(discrepancy) == 100:
                self.mean_discrepancy.append(_np.mean(discrepancy))
                # print(self.mean_discrepancy[-1])
                discrepancy = []

            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

        self.time = _time.time() - self.time

        # convert V and policy to tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

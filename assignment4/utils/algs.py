import copy
import time as _time
import math as _math

import numpy as np
import scipy.sparse as _sp
import numpy as _np
from mdptoolbox.mdp import QLearning as MDPQLearning, \
    _MSG_STOP_UNCHANGING_POLICY, _MSG_STOP_MAX_ITER, \
    _MSG_STOP_EPSILON_OPTIMAL_POLICY
from mdptoolbox.mdp import PolicyIteration as MDPPolicyIteration
from mdptoolbox.mdp import ValueIteration as MDPValueIteration
import mdptoolbox.util as _util


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

        self.evaluations = []

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
                discrepancy = []

            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

            eval_every = int(np.ceil(self.max_iter // 1000))
            if n % eval_every == 0:
                eval_v = self.eval_policy()
                mean_v = np.mean(eval_v)
                print(mean_v)
                self.evaluations.append((n, mean_v))

        self.time = _time.time() - self.time

        # convert V and policy to tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

    def eval_policy(self):

        reshapedR = self._computeReward(self.R, self.P)

        Ppolicy = _np.empty((self.S, self.S))
        Rpolicy = _np.zeros(self.S)
        for aa in range(self.A):  # avoid looping over S
            # the rows that use action a.
            ind = (self.policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign this
            if ind.size > 0:
                try:
                    Ppolicy[ind, :] = self.P[aa][ind, :]
                except ValueError:
                    Ppolicy[ind, :] = self.P[aa][ind, :].todense()
                Rpolicy[ind] = reshapedR[aa][ind]
        if type(self.R) is _sp.csr_matrix:
            Rpolicy = _sp.csr_matrix(Rpolicy)

        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
        return _np.linalg.solve(
            (_sp.eye(self.S, self.S) - self.discount * Ppolicy), Rpolicy)


class PolicyIteration(MDPPolicyIteration):

    def __init__(self,
                 transitions,
                 reward,
                 discount,
                 policy0=None,
                 max_iter=1000,
                 eval_type=0,
                 epsilon=0.01):

        super().__init__(transitions, reward, discount, policy0, max_iter,
                         eval_type)
        self.evaluations = []
        self.epsilon = epsilon

    def run(self):
        # Run the policy iteration algorithm.
        # If verbose the print a header
        if self.verbose:
            print('  Iteration\t\tNumber of different actions')
        # Set up the while stopping condition and the current time
        done = False
        self.time = _time.time()
        # loop until a stopping condition is reached
        while not done:
            self.iter += 1

            Vprev = copy.deepcopy(self.V)

            # these _evalPolicy* functions will update the classes value
            # attribute
            if self.eval_type == "matrix":
                self._evalPolicyMatrix()
            elif self.eval_type == "iterative":
                self._evalPolicyIterative()
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, null = self._bellmanOperator()
            del null
            # calculate in how many places does the old policy disagree with
            # the new policy
            n_different = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            if self.verbose:
                print(('    %s\t\t  %s') % (self.iter, n_different))
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop

            self.evaluations.append((self.iter, np.mean(self.V)))
            print((self.iter, np.mean(self.V)))

            variation = _util.getSpan(self.V - Vprev)

            # if n_different == 0:
            if variation < self.epsilon:
                done = True
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
            elif self.iter == self.max_iter:
                done = True
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
            else:
                self.policy = policy_next
        # update the time to return th computation time
        self.time = _time.time() - self.time
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())


class ValueIteration(MDPValueIteration):

    def __init__(self,
                 transitions,
                 reward,
                 discount,
                 epsilon=0.01,
                 max_iter=1000,
                 initial_value=0):

        super().__init__(transitions, reward, discount, epsilon, max_iter,
                         initial_value)
        self.evaluations = []

    def run(self):
        # Run the value iteration algorithm.

        if self.verbose:
            print('  Iteration\t\tV-variation')

        self.time = _time.time()
        while True:
            self.iter += 1

            Vprev = self.V.copy()

            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()

            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = _util.getSpan(self.V - Vprev)

            if self.verbose:
                print(("    %s\t\t  %s" % (self.iter, variation)))

            self.evaluations.append((self.iter, np.mean(self.V)))
            print((self.iter, np.mean(self.V)))

            if variation < self.epsilon:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

        self.time = _time.time() - self.time

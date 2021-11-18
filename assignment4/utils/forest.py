from mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning, MDP
import mdptoolbox.example
from scipy.stats import entropy

from utils.base import task_template


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

    task_template(problem_name, alg_name, param_grid, single_policy_iteration,
                  eval_mdp)


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

    task_template(problem_name, alg_name, param_grid, single_value_iteration,
                  eval_mdp)


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

    def single_qlearning(S, r2, p, discount, n_iter):
        P, R = mdptoolbox.example.forest(S, r2=r2, p=p)
        vi = QLearning(P, R, discount=discount, n_iter=n_iter)
        vi.run()
        return vi

    task_template(problem_name, alg_name, param_grid, single_qlearning,
                  eval_mdp)


def eval_mdp(mdp: MDP, **kwargs):
    policy = mdp.policy
    actions = set(policy)
    counts = {a: 0 for a in actions}
    for a in policy:
        counts[a] += 1
    e = entropy(list(counts.values()))

    return e

from mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning, MDP
import mdptoolbox.example
from scipy.stats import entropy

from utils.base import task_template, append_problem_name


def run_all():
    task_policy_iteration()
    task_value_iteration()
    task_q_learning()


def task_policy_iteration():
    problem_name = 'forest'
    alg_name = 'policy_iteration'

    param_grid = {
        'S': [5, 10, 100, 1000],
        'r1': [1, 8, 16],
        'r2': [1, 8, 16],
        'p': [0.1, 0.2, 0.3],
        'discount': [0.1, 0.6, 0.9, 0.99, 0.999],
        'max_iter': [100000]
    }

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_policy_iteration(S, r1, r2, p, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        pi = PolicyIteration(P, R, discount=discount, max_iter=max_iter)
        pi.run()
        return pi

    _, all_scores_table = task_template(problem_name, alg_name, param_grid,
                                        single_policy_iteration, eval_mdp,
                                        group_problems_by)

    # Effects of varying S.
    # The pattern is [0] + [1] * (S-4) + [0] * 3
    print('Varying S')
    for S in param_grid['S']:
        # Fix these variables
        r1 = 1
        r2 = 1
        p = 0.1
        discount = 0.9
        max_iter = 100000

        params_to_append = [('S', S), ('r1', r1), ('r2', r2), ('p', p)]
        problem_name_with_params = append_problem_name(problem_name,
                                                       params_to_append)
        table = all_scores_table[problem_name_with_params]
        kwargs = {'S': S, 'r1': r1, 'r2': r2, 'p': p, 'discount': discount,
                  'max_iter': max_iter}
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)

    # Effects of varying r1
    print('Varying r1')
    for r1 in param_grid['r1']:
        # Fix these variables
        S = 100
        r2 = 8
        p = 0.1
        discount = 0.9
        max_iter = 100000

        params_to_append = [('S', S), ('r1', r1), ('r2', r2), ('p', p)]
        problem_name_with_params = append_problem_name(problem_name,
                                                       params_to_append)
        table = all_scores_table[problem_name_with_params]
        kwargs = {'S': S, 'r1': r1, 'r2': r2, 'p': p, 'discount': discount,
                  'max_iter': max_iter}
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)

    # Effects of varying r2
    print('Varying r2')
    for r2 in param_grid['r2']:
        # Fix these variables
        S = 100
        r1 = 1
        p = 0.1
        discount = 0.9
        max_iter = 100000

        params_to_append = [('S', S), ('r1', r1), ('r2', r2), ('p', p)]
        problem_name_with_params = append_problem_name(problem_name,
                                                       params_to_append)
        table = all_scores_table[problem_name_with_params]
        kwargs = {'S': S, 'r1': r1, 'r2': r2, 'p': p, 'discount': discount,
                  'max_iter': max_iter}
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)

    # Effects of varying discount
    print('Varying discount')
    for discount in param_grid['discount']:
        # Fix these variables
        S = 100
        r1 = 8
        r2 = 8
        p = 0.1
        max_iter = 100000

        params_to_append = [('S', S), ('r1', r1), ('r2', r2), ('p', p)]
        problem_name_with_params = append_problem_name(problem_name,
                                                       params_to_append)
        table = all_scores_table[problem_name_with_params]
        kwargs = {'S': S, 'r1': r1, 'r2': r2, 'p': p, 'discount': discount,
                  'max_iter': max_iter}
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)


def task_value_iteration():
    problem_name = 'forest'
    alg_name = 'value_iteration'

    param_grid = {
        'S': [5, 10, 100, 1000],
        'r1': [1, 8, 16],
        'r2': [1, 8, 16],
        'p': [0.1, 0.2, 0.3],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [100000]
    }

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_value_iteration(S, r1, r2, p, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        vi = ValueIteration(P, R, discount=discount, max_iter=max_iter)
        vi.run()
        return vi

    task_template(problem_name, alg_name, param_grid, single_value_iteration,
                  eval_mdp, group_problems_by)


def task_q_learning():
    problem_name = 'forest'
    alg_name = 'q_learning'

    param_grid = {
        'S': [5, 10, 100, 1000],
        'r1': [1, 8, 16],
        'r2': [1, 8, 16],
        'p': [0.1, 0.2, 0.3],
        'discount': [0.9, 0.99, 0.999],
        'n_iter': [100000]
    }

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_qlearning(S, r1, r2, p, discount, n_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        vi = QLearning(P, R, discount=discount, n_iter=n_iter)
        vi.run()
        return vi

    task_template(problem_name, alg_name, param_grid, single_qlearning,
                  eval_mdp, group_problems_by)


def eval_mdp(mdp: MDP, **kwargs):
    policy = mdp.policy
    actions = set(policy)
    counts = {a: 0 for a in actions}
    for a in policy:
        counts[a] += 1
    e = entropy(list(counts.values()))

    return e

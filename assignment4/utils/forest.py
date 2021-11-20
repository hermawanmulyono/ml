from typing import Dict

from mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning, MDP
import mdptoolbox.example
from scipy.stats import entropy

from utils.base import task_template, append_problem_name


def run_all():
    all_scores_table1 = task_policy_iteration()
    all_scores_table2 = task_value_iteration()

    compare_scores_tables(all_scores_table1, all_scores_table2)

    task_q_learning()


def get_pi_vi_param_grid():
    return {
        'S': [5, 10, 100, 1000],
        'r1': [1, 8, 16],
        'r2': [1, 8, 16],
        'p': [0.1, 0.2, 0.3],
        'discount': [0.1, 0.6, 0.9, 0.99, 0.999],
        'max_iter': [100000]
    }


def task_policy_iteration():
    problem_name = 'forest'
    alg_name = 'policy_iteration'

    param_grid = get_pi_vi_param_grid()

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_policy_iteration(S, r1, r2, p, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        pi = PolicyIteration(P, R, discount=discount, max_iter=max_iter)
        pi.run()
        return pi

    _, all_scores_tables = task_template(problem_name, alg_name, param_grid,
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
        table = all_scores_tables[problem_name_with_params]
        kwargs = {
            'S': S,
            'r1': r1,
            'r2': r2,
            'p': p,
            'discount': discount,
            'max_iter': max_iter
        }
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
        table = all_scores_tables[problem_name_with_params]
        kwargs = {
            'S': S,
            'r1': r1,
            'r2': r2,
            'p': p,
            'discount': discount,
            'max_iter': max_iter
        }
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
        table = all_scores_tables[problem_name_with_params]
        kwargs = {
            'S': S,
            'r1': r1,
            'r2': r2,
            'p': p,
            'discount': discount,
            'max_iter': max_iter
        }
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
        table = all_scores_tables[problem_name_with_params]
        kwargs = {
            'S': S,
            'r1': r1,
            'r2': r2,
            'p': p,
            'discount': discount,
            'max_iter': max_iter
        }
        for kwargs_, results in table:
            if kwargs_ == kwargs:
                print(results)

    return all_scores_tables


def task_value_iteration():
    problem_name = 'forest'
    alg_name = 'value_iteration'

    param_grid = get_pi_vi_param_grid()

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_value_iteration(S, r1, r2, p, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        vi = ValueIteration(P, R, discount=discount, max_iter=max_iter)
        vi.run()
        return vi

    _, all_scores_table = task_template(problem_name, alg_name, param_grid,
                                        single_value_iteration, eval_mdp,
                                        group_problems_by)

    return all_scores_table


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


def compare_scores_tables(all_scores_table1: Dict[str, list],
                          all_scores_table2: Dict[str, list]):
    if set(all_scores_table1.keys()) != set(all_scores_table2.keys()):
        raise ValueError

    n_experiments = 0
    n_policy_same = 0
    n_iter1_smaller = 0
    n_iter1_faster = 0

    for problem1, problem2 in zip(all_scores_table1.keys(),
                                  all_scores_table2.keys()):
        assert problem1 == problem2
        table1 = all_scores_table1[problem1]
        table2 = all_scores_table2[problem2]

        for (key1, metrics1), (key2, metrics2) in zip(table1, table2):
            assert key1 == key2

            n_experiments += 1

            if metrics1['policy'] == metrics2['policy']:
                n_policy_same += 1

            if metrics1['iter'] <= metrics2['iter']:
                n_iter1_smaller += 1

            if metrics1['time'] <= metrics1['time']:
                n_iter1_faster += 1

    print(f'n_experiments {n_experiments}')
    print(f'n_policy_same {n_policy_same}')
    print(f'n_iter1_smaller {n_iter1_smaller}')
    print(f'n_iter1_faster {n_iter1_faster}')


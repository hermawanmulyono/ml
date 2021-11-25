import copy
import math
from typing import Dict

from mdptoolbox.mdp import MDP
import mdptoolbox.example
from scipy.stats import entropy
import matplotlib.pyplot as plt

from utils.algs import QLearning, PolicyIteration, ValueIteration
from utils.base import task_template, append_problem_name
from utils.outputs import convergence_plot


def run_all():
    all_scores_table1 = task_policy_iteration()
    all_scores_table2 = task_value_iteration()

    compare_scores_tables(all_scores_table1, all_scores_table2)

    task_q_learning()


def get_pi_vi_param_grid():
    return {
        'S': [10, 1000],
        'r1': [8],
        'r2': [8],
        'p': [0.1],
        'epsilon': [0.1, 0.01, 0.001, 0.0001],
        'discount': [0.9, 0.99, 0.999],
        'max_iter': [100000]
    }


def print_forest_policy(policy):
    policy_copy = list(policy)

    to_print = []
    while policy_copy:
        p = policy_copy.pop(0)

        if not to_print:
            to_print.append((p, 1))
        else:
            if p == to_print[-1][0]:
                to_print[-1] = (p, to_print[-1][1] + 1)
            else:
                to_print.append((p, 1))

    assert sum([[p] * n for p, n in to_print], []) == list(policy)

    s = ''
    for i in range(len(to_print)):
        if i > 0:
            s += ', '

        p, n = to_print[i]
        s += f'[{p}] x {n}'

    print(s)

    return to_print


def task_policy_iteration():
    problem_name = 'forest'
    alg_name = 'policy_iteration'

    param_grid = get_pi_vi_param_grid()

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_policy_iteration(S, r1, r2, p, epsilon, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        pi = PolicyIteration(P,
                             R,
                             epsilon=epsilon,
                             discount=discount,
                             max_iter=max_iter)
        pi.run()
        return pi

    joblib_table, all_scores_tables = task_template(problem_name, alg_name,
                                                    param_grid,
                                                    single_policy_iteration,
                                                    eval_mdp, group_problems_by)

    generate_convergence_plots(joblib_table, problem_name, alg_name)
    '''
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
    '''

    return all_scores_tables


def task_value_iteration():
    problem_name = 'forest'
    alg_name = 'value_iteration'

    param_grid = get_pi_vi_param_grid()

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_value_iteration(S, r1, r2, p, epsilon, discount, max_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        vi = ValueIteration(P,
                            R,
                            epsilon=epsilon,
                            discount=discount,
                            max_iter=max_iter)
        vi.run()
        return vi

    joblib_table, all_scores_table = task_template(problem_name, alg_name,
                                                   param_grid,
                                                   single_value_iteration,
                                                   eval_mdp, group_problems_by)

    generate_convergence_plots(joblib_table, problem_name, alg_name)

    return all_scores_table


def generate_convergence_plots(joblib_table, problem_name: str, alg_name: str):
    param_grid = get_pi_vi_param_grid()

    for S in param_grid['S']:
        r1 = param_grid['r1'][0]
        r2 = param_grid['r2'][0]
        p = param_grid['p'][0]
        max_iter = param_grid['max_iter'][0]

        plt.figure()
        for epsilon in param_grid['epsilon'][::-1]:
            discount = 0.99
            kwargs = {
                'S': S,
                'r1': r1,
                'r2': r2,
                'p': p,
                'epsilon': epsilon,
                'discount': discount,
                'max_iter': max_iter
            }

            pi: PolicyIteration
            evaluations = None
            for kwargs_, pi in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = pi.evaluations
                    print(kwargs_)
                    print_forest_policy(pi.policy)
                    break

            assert evaluations is not None

            steps, vmean = zip(*evaluations)
            plt.plot(steps, vmean, label=f'{epsilon}')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')

        problem_name_w_size = f'{problem_name}_{S}'
        filename = convergence_plot(problem_name_w_size, alg_name, 'epsilon')
        plt.savefig(filename)

        plt.figure()
        for discount in param_grid['discount']:
            epsilon = 0.001
            kwargs = {
                'S': S,
                'r1': r1,
                'r2': r2,
                'p': p,
                'epsilon': epsilon,
                'discount': discount,
                'max_iter': max_iter
            }

            pi: PolicyIteration
            evaluations = None
            for kwargs_, pi in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = pi.evaluations
                    print(kwargs_)
                    print_forest_policy(pi.policy)
                    break

            assert evaluations is not None

            steps, vmean = zip(*evaluations)
            plt.plot(steps, vmean, label=f'{discount}')

        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')
        filename = convergence_plot(problem_name_w_size, alg_name, 'gamma')
        plt.savefig(filename)


def epsilon_schedule(n):
    return max(1 / math.log(n + math.exp(0)), 0.5)


def learning_rate_schedule(n):
    lr = 1 / math.log(n + math.exp(0))
    return max(lr, 1E-2)


def task_q_learning():
    problem_name = 'forest'
    alg_name = 'q_learning'

    param_grid = [{
        'S': [10],
        'r1': [8],
        'r2': [8],
        'p': [0.1],
        'epsilon_schedule': [None, epsilon_schedule, 0.9],
        'learning_rate_schedule': [None, learning_rate_schedule],
        'discount': [0.9, 0.99, 0.999],
        'n_iter': [10000]
    }, {
        'S': [1000],
        'r1': [8],
        'r2': [8],
        'p': [0.1],
        'epsilon_schedule': [None, epsilon_schedule, 0.9],
        'learning_rate_schedule': [None, learning_rate_schedule],
        'discount': [0.9, 0.99, 0.999],
        'n_iter': [100000]
    }]

    group_problems_by = ['S', 'r1', 'r2', 'p']

    def single_qlearning(S, r1, r2, p, epsilon_schedule, learning_rate_schedule,
                         discount, n_iter):
        P, R = mdptoolbox.example.forest(S, r1=r1, r2=r2, p=p)
        vi = QLearning(P,
                       R,
                       discount=discount,
                       n_iter=n_iter,
                       learning_rate_schedule=learning_rate_schedule,
                       epsilon_schedule=epsilon_schedule)
        vi.run()
        return vi

    joblib_table, _ = task_template(problem_name, alg_name, param_grid,
                                    single_qlearning, eval_mdp,
                                    group_problems_by)

    for grid in param_grid:
        S = grid['S'][0]
        r1 = grid['r1'][0]
        r2 = grid['r2'][0]
        p = grid['p'][0]
        n_iter = grid['n_iter'][0]

        plt.figure()
        for epsilon_schedule_ in grid['epsilon_schedule'][::-1]:
            discount = 0.99
            kwargs = {
                'S': S,
                'r1': r1,
                'r2': r2,
                'p': p,
                'epsilon_schedule': epsilon_schedule_,
                'learning_rate_schedule': None,
                'discount': discount,
                'n_iter': n_iter
            }

            ql: QLearning
            evaluations = None
            for kwargs_, ql in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = ql.evaluations
                    print(kwargs_)
                    print_forest_policy(ql.policy)
                    break

            assert evaluations is not None

            steps, vmean = zip(*evaluations)

            if callable(epsilon_schedule_):
                label = epsilon_schedule_.__name__
            elif epsilon_schedule_ is None:
                label = 'default'
            else:
                label = f'{epsilon_schedule_}'

            plt.plot(steps, vmean, label=label)

        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')

        problem_name_w_size = f'{problem_name}_{S}'
        filename = convergence_plot(problem_name_w_size, alg_name,
                                    'epsilon_schedule')
        plt.savefig(filename)

        plt.close("all")
        plt.figure()
        for learning_rate_schedule_ in grid['learning_rate_schedule'][::-1]:
            discount = 0.99
            kwargs = {
                'S': S,
                'r1': r1,
                'r2': r2,
                'p': p,
                'epsilon_schedule': None,
                'learning_rate_schedule': learning_rate_schedule_,
                'discount': discount,
                'n_iter': n_iter
            }

            ql: QLearning
            evaluations = None
            for kwargs_, ql in joblib_table:
                if kwargs_ == kwargs:
                    evaluations = ql.evaluations
                    print(kwargs_)
                    print_forest_policy(ql.policy)
                    break

            assert evaluations is not None

            steps, vmean = zip(*evaluations)

            if callable(learning_rate_schedule_):
                label = learning_rate_schedule_.__name__
            elif learning_rate_schedule_ is None:
                label = 'default'
            else:
                label = f'{learning_rate_schedule_}'

            plt.plot(steps, vmean, label=label)

        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('V mean')

        problem_name_w_size = f'{problem_name}_{S}'
        filename = convergence_plot(problem_name_w_size, alg_name,
                                    'epsilon_schedule')
        plt.savefig(filename)
        plt.show()


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

        table1 = copy.deepcopy(table1)
        table2 = copy.deepcopy(table2)

        table1.sort(key=lambda x: str(x[0]))
        table2.sort(key=lambda x: str(x[0]))

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

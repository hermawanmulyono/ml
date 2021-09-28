import numpy as np
import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose


def onemax_task():
    fitness = mlrose.OneMax()
    length = 100
    problem_fit = mlrose.DiscreteOpt(length=length,
                                     fitness_fn=fitness,
                                     maximize=True)
    print('OneMaxTask')

    best_state, best_fitness = mlrose.hill_climb(problem_fit)
    print(f'Hill climb: {best_fitness}')

    best_state, best_fitness = mlrose.simulated_annealing(problem_fit,
                                                          max_attempts=length)
    print(f'Simulated annealing: {best_fitness}')

    # Solve problem using the genetic algorithm
    for mutation_prob in np.logspace(-1, -5, 10)[::-1]:
        best_state, best_fitness = mlrose.genetic_alg(
            problem_fit,
            pop_size=200,
            mutation_prob=mutation_prob,
            max_attempts=length)
        print(f'GA: {best_fitness}')

    best_state, best_fitness = mlrose.mimic(problem_fit, max_attempts=length)
    print(f'MIMIC: {best_fitness}')


def fourpeaks_task():
    fitness = mlrose.SixPeaks()
    length = 100
    problem_fit = mlrose.DiscreteOpt(length=length,
                                     fitness_fn=fitness,
                                     maximize=True)
    print('Four peaks')

    best_state, best_fitness = mlrose.hill_climb(problem_fit, restarts=100)
    print(f'Hill climb: {best_fitness}')

    best_state, best_fitness = mlrose.simulated_annealing(problem_fit,
                                                          max_attempts=length)
    print(f'Simulated annealing: {best_fitness}')

    # Solve problem using the genetic algorithm
    for mutation_prob in np.logspace(-1, -5, 10)[::-1]:
        print(f'mutation: {mutation_prob}')
        for _ in range(10):
            best_state, best_fitness = mlrose.genetic_alg(
                problem_fit,
                pop_size=200,
                mutation_prob=mutation_prob,
                max_attempts=length)
            print(f'GA: {best_fitness}')

    best_state, best_fitness = mlrose.mimic(problem_fit, max_attempts=length)
    print(f'MIMIC: {best_fitness}')



def maxkcolor_edges(num_vertices: int, num_edges: int):
    """Generates edges for MaxKColor problems randomly

    Args:
        num_vertices: Number of vertices
        num_edges: Number of edges

    Returns:
        List of randomly generated undirected edges
        `[..., (v1, v2) ,...]`. Note that (v1, v2) and
        (v2, v1) are equivalent.

    """
    if num_edges <= 1 or num_vertices <= 1:
        raise ValueError

    if num_edges > num_vertices * (num_vertices - 1) / 2:
        raise ValueError

    edges = set()
    while len(edges) < num_edges:
        v1 = np.random.randint(0, num_vertices)
        v2 = v1
        while v2 == v1:
            v2 = np.random.randint(0, num_vertices)

        if ((v1, v2) not in edges) and ((v2, v1) not in edges):
            edges.add((v1, v2))

    return list(edges)


def max_kcolor_task():
    num_vertices = 30
    num_edges = 20

    fitness = mlrose.MaxKColor(maxkcolor_edges(num_vertices, num_edges))
    problem_fit = mlrose.DiscreteOpt(length=num_vertices,
                                     fitness_fn=fitness,
                                     maximize=True)

    print('MaxKColor')

    # Solve problem using the genetic algorithm
    best_state, best_fitness = mlrose.genetic_alg(problem_fit)

    print(f'GA: {best_fitness}')

    best_state, best_fitness = mlrose.hill_climb(problem_fit)

    print(f'Hill climb: {best_fitness}')

    best_state, best_fitness = mlrose.mimic(problem_fit)

    print(f'MIMIC: {best_fitness}')


def task1():
    """A task to compare some optimization problems
    """
    fourpeaks_task()
    # onemax_task()
    max_kcolor_task()

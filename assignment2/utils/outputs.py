import os

OUTPUT_DIRECTORY = 'outputs'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def optimization_grid_table(problem_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{problem_name}_{alg_name}_grid_table.json'


def optimization_grid_summary(problem_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{problem_name}_{alg_name}_grid_summary.json'


def nn_joblib(optimizer_name: str):
    """MLRose neural-network model"""
    return f'{OUTPUT_DIRECTORY}/nn_{optimizer_name}.joblib'


def nn_grid_table(algorithm_name: str):
    return f'{OUTPUT_DIRECTORY}/nn_{algorithm_name}_grid_table.json'


def nn_grid_summary(algorithm_name: str):
    return f'{OUTPUT_DIRECTORY}/nn_{algorithm_name}_grid_summary.json'

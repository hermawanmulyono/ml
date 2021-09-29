import os

OUTPUT_DIRECTORY = 'outputs'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def grid_results_json(problem_name: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{problem_name}_{alg_name}.json'

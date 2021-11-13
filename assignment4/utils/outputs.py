"""
This module contains the paths to output files
"""

import os
from typing import Optional

OUTPUT_DIRECTORY = 'outputs'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Some image outputs may be visualized if the operating system is Windows
# as it has GUI.
windows = (os.name == 'nt')


def table_joblib(problem: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{problem}_{alg_name}.joblib'


def table_score(problem: str, alg_name: str):
    return f'{OUTPUT_DIRECTORY}/{problem}_{alg_name}_grid_table.json'
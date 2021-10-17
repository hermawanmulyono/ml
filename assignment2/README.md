# Assignment 2 - Randomized Optimization

## Virtual Environment Setup

In Ubuntu 18.04, the following commands were tested to create
the virtual environment.

```shell
# Create a new environment "assignment2" with Python 3.7
conda create -n assignment2 python=3.7

# Install the rest of the packages
pip install -r requirements.txt
```

## Running The Code

Use the `main.py` script to run the code. This will automatically generate 
all plots and results.

```shell
# Must be run inside the virtual environment
python main.py
```

The code execution is as follows:
- If the `outputs` directory is empty, running `main.y` will run 
  all optimization algorithms and produce the output files.
- Otherwise, if some files can be found in the `outputs` 
directory, the code execution will skip generating files that can
be found. This implies that:
  - To re-run a particular problem with a particular optimization algorithm, 
    simply delete the corresponding `json` file.  
  - To regenerate a plot, remove the corresponding `.png` file.

## Directory Structure

Here is the directory structure.

```
.
├── outputs/    # Output directory to store the all outputs
├── tasks/      # Contains business logic of the project
├── utils/      # Utility packages 
├── main.py     # Script to run
└── README.md
```
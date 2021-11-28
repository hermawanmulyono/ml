# Assignment 4 - Markov Decision Processes

## Virtual Environment Setup

In Ubuntu 18.04, the following commands were tested to create
the virtual environment.

```shell
# Create a new environment "assignment4" with Python 3.7
conda create -n assignment4 python=3.7

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
be found. 

## Directory Structure

Here is the directory structure.

```
.
├── outputs/    # Output directory to store the all outputs
├── utils/      # Utility packages 
├── main.py     # Script to run
└── README.md
```
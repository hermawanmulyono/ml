# Assignment 1 - Supervised Learning

## Virtual Environment Setup
In Ubuntu 18.04, the following commands were tested to create
the virtual environment.

```shell
# Create a new environment "cs7641" with Python 3.7
conda create -n cs7641 python=3.7

# Pytorch needs to be installed separately
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Install the rest of the packages
pip install -r requirements.txt
```

## Running The Code

Use the `main.py` script to run the code.

```shell
# Must be run inside the virtual environment
python main.py

# Show help
python main.py --help
```

The code execution is as follows:
- If the `outputs` directory is empty, running `main.y` will train all 
the models. 
- Otherwise, if some files can be found in the `outputs` 
directory, the code execution will skip generating files that can
be found. This implies that:
  - To re-train a model, remove the files corresponding to the model from the 
    `outputs` 
directory.
  - To regenerate a plot, remove the corresponding `.png` file.


## Directory Structure

Here is the directory structure.

```
.
├── mnist/      
├── outputs/    # Output directory to store the all outputs
├── utils/      # Utility packages 
├── main.py     # Script to run
└── README.md
```

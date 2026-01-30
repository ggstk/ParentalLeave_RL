
# Analyzing strategic parental leave decisions using two-player multi-agent reinforcement learning 

This is the code for implementing the nash-q learning algorithm presented in the paper:
Analyzing strategic parental leave decisions using two-player multi-agent reinforcement learning 

## Installation


<!-- - To install, `cd` into the root directory and type `pip install -e .` -->

<!-- - Use `requirements.txt` to install dependencies. -->

- Create the conda environment using the provided configuration file
```
conda env create -f environment.yml
```
- Activate the environment
```
conda activate ParentLeave
```

### Core training parameters

- `--lr`: learning rate (default: `0.5`)
- `--gamma`: discount factor (default: `0.95`)

## Experimental Results & Logs
- `./Q_tables`:
- `./Q_learning_curve`:

## Example scripts
- `./run_ParentalLeave.py`:
- `./run_nash.py`:
- `./MARL.py`
- `./MARL_early_stopping.py`

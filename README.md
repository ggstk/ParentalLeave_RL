
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
- `--alpha`:The competitive career penalty, which is activated only if the employee has taken parental leave while the other employee in the same job position has not used it and competes for a promotion.
- `--delta`:The individual career penalty for taking parental leave, which means the decrement in promotion probability after taking parental leave.
- $U_1^+$:Employee1's perceived utility from taking annual parental leave
- $U_2^+$:Employee2's perceived utility from taking annual parental leave

## Experimental Results & Logs
- `./Q_tables`: Stores the Q-tables obtained from Nash Q-learning experiments, including  
  - `Q_0(s, a)`: State–action value table for Agent 0
  - `Q_1(s, a)`: State–action value table for Agent 1
- `./Q_learning_curve`:Stores learning curves for each experiment, recorded for predefined **states of interest**, and used to analyze convergence speed and training stability.
- `./saved_models`:
  - states of interest. `reset_states.json`
  - stores the Nash trajectories, including the evolution of strategies and visited state sequences during training.
  - `./results`:

## Example scripts
- `./run_ParentalLeave.py`:Script for single-agent experiments using Q-learning.
- `./run_nash.py`:Script for training two agents using Nash Q-learning.
-  `./multiprocess.py`:Runs multiple experiments in parallel using multiprocessing, enabling large-scale parameter sweeps.
- `./MARL.py`:Implements the Parental Leave environment and the Nash Q-learning algorithm.
- `./MARL_early_stopping.py`:Extends `MARL.py` by incorporating an early stopping mechanism to reduce unnecessary training and improve computational efficiency.
- `./calculate_probs.py`:
  - `calculate_both_agent_probability()`: Computes the probability that **both employees** end in the state where parental leave has already been taken (m = 0) under Nash equilibrium.
  - `calculate_one_agent_probability()`:Computes the probability that **only one employee** ends in the state where parental leave has already been taken (m = 0) under Nash equilibrium.
  - `calculate_each_agent_probability()`: Computes the individual probability for each employee of ending in the state where parental leave has already been taken (m = 0) under Nash equilibrium.
- `./collect_trajectory.py`: Collecting the Nash equilibrium trajectory.
- `./q_values_comparison.py`: Loads and merges the learned Q-tables of both agents and exports the Q-values to an Excel file.
- `./nash_equilibriums.py`: Computes the Nash equilibrium action(s) for each joint state using the learned Q-tables, and exports the resulting equilibrium policies to Excel files.
 
  


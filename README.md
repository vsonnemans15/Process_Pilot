# Process Pilot
## Summary of the paper
Prescriptive process monitoring (PresPM) studies techniques that leverage event logs to recommend actions at runtime that improve process outcomes. Recent PresPM approaches increasingly rely on reinforcement learning (RL). Because interacting with real-world processes is often infeasible, these approaches typically construct a Markov Decision Process (MDP) from historical event data to simulate the execution environment. This paper presents Process Pilot, an RL-based PresPM approach that discovers an MDP from an event log describing the stochastic environment in which the historical processes were executed and uses it to learn a policy for executing future processes in a way that optimizes their outcomes. The paper further proposes a collection of transition-based state abstraction functions for discovered MDPs and introduces measures of the quality of such models. An evaluation on synthetic and industrial event logs compares Process Pilot with prior work. The results show that different abstractions lead to substantially different policy effectiveness, with transition-based abstractions consistently outperforming alternatives. Moreover, higher-quality MDP representations yield recommendations that lead to improved process outcomes more consistently.

## Structure of the code
The structure of the code is as follows:

- **Process_Pilot/**
  - **data_logs/** – Preprocessed event logs used in the experiments
  - **dataset_manager/** – Utilities for loading and managing event logs
  - **Final_plots/** – Figures included in the paper
  - **SimBank/**
    - **SimBank/** – The SimBank simulator code provided by De Moor et al. (2025)
      - `activity_execution.py` – Execute an activity in the process
      - `confounding_level.py` – Create a dataset with a specified confounding level
      - `extra_flow_conditions.py` – The (extra) underlying mechanism of the control-flow
      - `petri_net_generator.py` – Setup initial control-flow
      - `requirements.txt` – Requirements of the SimBank simulator only
      - `SimBank_Generator_Guide.ipynb` – Guide on how to use SimBank
      - `simulation.py` – Main simulator code
  - **src/** – Main code for the experiments
    - **slurm_files/** – Slurm files to run the experiments
    - `bisimulation_distance.py` – Compute bisimulation distance and other quality measures
    - `MDP_functions.py` – Contains core functions for defining the MDP
    - `testing.py` – Evaluate the RL agent on testing cases
    - `training.py` – Training the RL agent on training cases
    - `utils.py` – Utilities

## Installation.
The ```requirements.txt``` file provides the necessary packages for all experiments. 
The code was written in Python ```3.10.16```.

## Experiments of the paper
```command.ls``` contains all the python commands to replicate the results in the paper.
```command_slurm.ls``` contains all SLURM scripts with the Python commands used to replicate the results reported in the paper (these are identical to the commands listed in ```command.ls```).

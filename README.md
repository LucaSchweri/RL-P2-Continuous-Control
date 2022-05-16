# Project 1: Navigation
by Luca Schweri, 2022

## Project Details

### Environment

In this environment, each agent controls a double-jointed arm to move it to a target location.

### Rewards

- **+0.1**: For each step that the agent's hand is in the target location

### Actions

Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### States

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

### Solved

For the single agent environment, the environment is solved as soon as the agent is able to get an average score higher than 30 over 100 consecutive episode.

For the 20 agent environment, the environment is solved as soon as the agent is able to get an average score higher than 30 over 100 consecutive episode over the average of all 20 agents.


## Getting Started

To set up your python environment use the following commands:

```
conda create --name udacity_rl python=3.6
conda activate udacity_rl

pip install -r requirements.txt
conda install -n udacity_rl pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

```

Download the Unity environment and change the path to the environment in the first code cell of the jupyter notebook [navigation.ipynb](navigation.ipynb):

Environment with **1** Agent:
- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Environment with **20** Agent:
- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Instructions

To train or test a new approach, first modify the [config.json](config.json) file and then open the jupyter notebook [navigation.ipynb](navigation.ipynb) by using the command
```
jupyter notebook
```

In the notebook you find three sections (which are better described in the jupyter notebook):
- **Trainig**: Can be used to train an agent.
- **Test**: Can be used to test an already trained agent.
- **Comparison**: Can be used to compare different approaches.

Note that you might need to change the location of the Unity environment in the first code cell.
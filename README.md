# Q-Learning Maze Solver
[helped by claude]
This project implements a Q-learning algorithm to solve a simple maze navigation problem. The agent learns to find the optimal path from start to goal while avoiding obstacles. 

## Features

- NxN grid maze environment with configurable size
- Customizable obstacles and walls
- Real-time visualization of the learning process
- Q-learning implementation with adjustable parameters
- Performance monitoring during training

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

```
pip install numpy matplotlib
```

## Usage

```
python maze_qlearning.py
```

2. The program will:
   - Create a maze environment
   - Train the Q-learning agent
   - Visualize the learning process in real-time
   - Display training progress every 100 episodes

## Parameters

You can modify the following parameters in the code:

- Maze size: Change `size` parameter in `MazeEnvironment`
- Learning rate: Modify `learning_rate` in `QLearningAgent`
- Discount factor: Adjust `gamma` in `QLearningAgent`
- Exploration rate: Change `epsilon` and related parameters
- Number of training episodes: Modify `episodes` parameter in `train()`

## Visualization

- Red dot: Agent's current position
- Green dot: Goal position
- Black cells: Walls/obstacles
- White cells: Valid paths

## License

MIT License
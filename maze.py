import numpy as np

class MazeEnvironment:
    def __init__(self, size=5):
        self.size = size
        # Create maze with walls (1) and paths (0)
        self.maze = np.zeros((size, size))
        # Add some random obstacles
        self.maze[1:4, 2] = 1  # Example wall
        # Start position (0,0) and goal position (size-1, size-1)
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.current_pos = self.start
        self.action_space = 4
        
    def reset(self):
        self.current_pos = self.start
        return self.current_pos
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = tuple(np.add(self.current_pos, moves[action]))
        
        # Check if move is valid
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            self.maze[new_pos] == 0):
            self.current_pos = new_pos
            
        # Calculate reward
        if self.current_pos == self.goal:
            reward = 100
            done = True
        elif self.current_pos == new_pos:  # Valid move
            reward = -1
            done = False
        else:  # Hit wall or boundary
            reward = -5
            done = False
            
        return self.current_pos, reward, done
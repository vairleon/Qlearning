import numpy as np
import matplotlib.pyplot as plt
from maze import MazeEnvironment

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.gamma = 0.95
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def visualize_maze(env, agent_pos):
    plt.clf()
    plt.imshow(env.maze, cmap='binary')
    plt.plot(agent_pos[1], agent_pos[0], 'ro', markersize=15)
    plt.plot(env.goal[1], env.goal[0], 'go', markersize=15)
    plt.grid(True)
    plt.pause(0.1)

def train(episodes=1000, visualize=False):
    env = MazeEnvironment()
    agent = QLearningAgent(env.size, env.action_space)
    
    if visualize:
        plt.ion()
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if visualize:
                visualize_maze(env, state)
                
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
    
    if visualize:
        plt.ioff()
    
    return agent

if __name__ == "__main__":
    trained_agent = train(episodes=500, visualize=False) 
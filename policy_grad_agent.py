import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from maze import MazeEnvironment

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # Ensure input is properly shaped
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        return self.network(x)

class PolicyGradientAgent:
    def __init__(self, action_size):
        self.state_size = 2  # x and y coordinates only
        self.action_size = action_size
        self.hidden_size = 64
        self.learning_rate = 0.01
        self.gamma = 0.99
        
        self.policy = PolicyNetwork(self.state_size, self.hidden_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def get_action(self, state):
        # Convert state to tensor and ensure proper shape
        state_tensor = torch.FloatTensor([state[0], state[1]])
        probs = self.policy(state_tensor)  
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update(self):
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Optimize the model
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.saved_log_probs = []
        self.rewards = []

def visualize_maze(env, agent_pos):
    plt.clf()
    plt.imshow(env.maze, cmap='binary')
    plt.plot(agent_pos[1], agent_pos[0], 'ro', markersize=15)
    plt.plot(env.goal[1], env.goal[0], 'go', markersize=15)
    plt.grid(True)
    plt.pause(0.1)

def train(episodes=1000, visualize=False):
    env = MazeEnvironment()
    agent = PolicyGradientAgent(env.action_space)
    
    if visualize:
        plt.ion()
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if visualize:
                visualize_maze(env, state)
            
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.rewards.append(reward)
            state = next_state
            episode_reward += reward
        
        # Update policy after each episode
        agent.update()
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {episode_reward}")
    
    if visualize:
        plt.ioff()
    
    return agent

if __name__ == "__main__":
    trained_agent = train(episodes=500, visualize=False)
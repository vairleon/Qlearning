import numpy as np
import matplotlib.pyplot as plt
from maze import MazeEnvironment

class QLearningAgent:
    def __init__(self, state_size, action_size, strategy='epsilon-greedy'):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.visit_counts = np.ones((state_size, state_size, action_size))  # Initialize with 1 to avoid division by zero
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # UCB parameters
        self.c = 2.0  # Exploration constant for UCB
        
        # Common parameters
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.strategy = strategy
        self.total_steps = 0
        
    def get_action(self, state):
        if self.strategy == 'epsilon-greedy':
            return self._get_action_epsilon_greedy(state)
        # Upper Confidence Bound
        elif self.strategy == 'ucb': 
            return self._get_action_ucb(state)
        else:
            raise ValueError("Unknown strategy. Use 'epsilon-greedy' or 'ucb'")
    
    def _get_action_epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])
    
    def _get_action_ucb(self, state):
        self.total_steps += 1
        ucb_values = self.q_table[state] + self.c * np.sqrt(
            np.log(self.total_steps) / self.visit_counts[state]
        )
        return np.argmax(ucb_values)
    
    def update(self, state, action, reward, next_state):
        # Update Q-value
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value
        
        # Update visit counts for UCB
        self.visit_counts[state][action] += 1
        
        # Decay epsilon if using epsilon-greedy
        if self.strategy == 'epsilon-greedy':
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

def visualize_maze(env, agent_pos):
    plt.clf()
    plt.imshow(env.maze, cmap='binary')
    plt.plot(agent_pos[1], agent_pos[0], 'ro', markersize=15)
    plt.plot(env.goal[1], env.goal[0], 'go', markersize=15)
    plt.grid(True)
    plt.pause(0.1)

def train(episodes=1000, visualize=False, strategy='epsilon-greedy'):
    env = MazeEnvironment()
    agent = QLearningAgent(env.size, env.action_space, strategy=strategy)
    
    if visualize:
        plt.ion()
    
    rewards_history = []
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
        
        rewards_history.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
    
    if visualize:
        plt.ioff()
    
    return agent, rewards_history

if __name__ == "__main__":
    # Train with epsilon-greedy
    trained_agent_eps, rewards_eps = train(episodes=500, visualize=False, strategy='epsilon-greedy')
    
    # Train with UCB
    trained_agent_ucb, rewards_ucb = train(episodes=500, visualize=False, strategy='ucb')
    
    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_eps, label='ε-greedy')
    plt.plot(rewards_ucb, label='UCB')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Exploration Strategies')
    plt.legend()
    plt.show() 
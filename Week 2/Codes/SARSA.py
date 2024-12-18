import gym
import numpy as np
import random
import time

# Initialize the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")  # Visualization enabled

# Discretize the continuous state space into a finite grid
num_bins = (10, 10, 10, 10)  # Number of bins for each state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-1, 1)   # Clip velocity values
state_bounds[3] = (-1, 1)   # Clip pole angular velocity values

# Hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.99       # Discount factor
epsilon = 1.0      # Initial exploration probability
epsilon_decay = 0.995  # Decay rate for epsilon
epsilon_min = 0.01     # Minimum epsilon value
num_episodes = 500     # Total episodes to train
max_steps = 200        # Max steps per episode

# Initialize Q-tables for Q-Learning and SARSA
q_table_qlearning = np.zeros(num_bins + (env.action_space.n,))
q_table_sarsa = np.zeros(num_bins + (env.action_space.n,))

# Function to discretize a continuous state
def discretize_state(state):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(4)]
    new_state = [int(np.digitize(ratio, np.linspace(0, 1, num_bins[i] - 1))) for i, ratio in enumerate(ratios)]
    return tuple(min(num_bins[i] - 1, max(0, new_state[i])) for i in range(4))

# Epsilon-greedy policy for action selection
def epsilon_greedy_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Train Q-Learning Agent
print("Training Q-Learning Agent...")
epsilon_q = epsilon
for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False

    for step in range(max_steps):
        env.render()
        action = epsilon_greedy_action(q_table_qlearning, state, epsilon_q)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Q-Learning update rule
        best_next_action = np.argmax(q_table_qlearning[next_state])
        q_table_qlearning[state + (action,)] += alpha * (
            reward + gamma * q_table_qlearning[next_state + (best_next_action,)] - q_table_qlearning[state + (action,)]
        )

        state = next_state
        if done:
            print(f"Q-Learning Episode {episode+1}: Completed in {step+1} steps")
            break

    epsilon_q = max(epsilon_q * epsilon_decay, epsilon_min)  # Decay epsilon

# Train SARSA Agent
print("\nTraining SARSA Agent...")
epsilon_sarsa = epsilon
for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    action = epsilon_greedy_action(q_table_sarsa, state, epsilon_sarsa)
    done = False

    for step in range(max_steps):
        env.render()
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        next_action = epsilon_greedy_action(q_table_sarsa, next_state, epsilon_sarsa)

        # SARSA update rule
        q_table_sarsa[state + (action,)] += alpha * (
            reward + gamma * q_table_sarsa[next_state + (next_action,)] - q_table_sarsa[state + (action,)]
        )

        state, action = next_state, next_action  # Move to next state and action
        if done:
            print(f"SARSA Episode {episode+1}: Completed in {step+1} steps")
            break

    epsilon_sarsa = max(epsilon_sarsa * epsilon_decay, epsilon_min)  # Decay epsilon

env.close()

# Compare Results
print("\nTraining Complete!")
print("Q-Learning Learned Q-Table:")
print(q_table_qlearning)
print("\nSARSA Learned Q-Table:")
print(q_table_sarsa)

import numpy as np
import gym
from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# Create the environment
env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)
env.reset()

# Parameters
n_states = env.observation_space.n
n_actions = env.action_space.n
alpha = 0.01  # Learning rate
gamma = 0.99  # Discount factor, adjusted for more future value consideration
epsilon = 0.5  # Exploration rate
n_episodes = 100
n_steps = 100  # Adjusted for practical run time
threshold = 1e-20

# Initialize Q-table
reward_table = [0,1,1,1,1,-10,1,-10,1,1,1,-10,-10,1,1,100]

# Function to choose the next action
def greedy_policy(state, epsilon, Q):
    if np.random.uniform(0, 1) < epsilon:
        a = env.action_space.sample()  # Explore action space
    else:
        a = np.argmax(Q[state, :])  # Exploit learned values
    return a


# Q-learning algorithm
def value_iteration_Q(env):
    Q = np.zeros((n_states, n_actions))
    for episode in range(n_episodes):
        updated_Q = np.copy(Q)
        s, prob = env.reset()
        for step in range(n_steps):
            a = greedy_policy(s, epsilon, Q)

            s_, reward, terminated, truncated, info = env.step(a)

            # Q-learning update rule
            Q[s, a] = Q[s, a] + alpha * (reward_table[s_] + gamma * np.max(Q[s_, :]) - Q[s, a])
            print(s, a, np.argmax(Q[s]))

            s = s_
            if terminated or truncated:
                break
        if (np.sum(np.fabs(updated_Q - Q)) <= threshold):
            print(Q)
            break
    return Q


# After training
Q_table = value_iteration_Q(env)
# Q_table = np.zeros((n_states, n_actions))
state, prob = env.reset()
for i in range(n_steps):
    env.render()
    action = np.argmax(Q_table[state])
    print(action)
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    if terminated or truncated:
        state, prob = env.reset()

env.close()


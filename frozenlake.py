import numpy as np
import pandas as pd
import gym

# Create the environment
env = gym.make('FrozenLake-v1', render_mode="human")
state, info = env.reset(seed=82)

# Parameters
n_states = env.observation_space.n
n_actions = env.action_space.n
alpha = 0.01  # Learning rate
gamma = 0.01  # Discount factor
n_episodes = 100
n_steps = 500

def updateValue(state, next_state, reward):
    V[state] += alpha * (reward_table[state] + gamma * V[next_state] - V[state])
    return V[state]

def td_policy(state, V):
    next_state_0, reward_0, _, _, _ = env.step(0)
    next_state_1, reward_1, _, _, _ = env.step(1)
    next_state_2, reward_2, _, _, _ = env.step(2)
    next_state_3, reward_3, _, _, _ = env.step(3)
    V_0 = updateValue(state, next_state_0, reward_0)
    V_1 = updateValue(state, next_state_1, reward_1)
    V_2 = updateValue(state, next_state_2, reward_2)
    V_3 = updateValue(state, next_state_3, reward_3)
    V_steps = [V_0, V_1, V_2, V_3]
    action = np.argmax(V_steps)
    return action

def monte_carlo_policy(path, V):
    next_state_0, reward_0, _, _, _ = env.step(0)
    next_state_1, reward_1, _, _, _ = env.step(1)
    next_state_2, reward_2, _, _, _ = env.step(2)
    next_state_3, reward_3, _, _, _ = env.step(3)
    V_0 = updateValue(state, next_state_0, reward_0)
    V_1 = updateValue(state, next_state_1, reward_1)
    V_2 = updateValue(state, next_state_2, reward_2)
    V_3 = updateValue(state, next_state_3, reward_3)
    V_steps = [V_0, V_1, V_2, V_3]
    action = 0
    for i, V in enumerate(V_steps):
        if (V < V_steps[i]):
            action = i

    return action

reward_table = [1,1,1,1,1,-10,1,-10,1,1,1,-10,-10,1,1,100]
V = [1,1,1,1,1,-10,1,-10,1,1,1,-10,-10,1,1,100]
path = []
G = 0

# TD policy function
for i in range(n_episodes):
    # for every step in the episode
    for t in range(n_steps):
        # select an action according to random policy
        # action = env.action_space.sample()
        action = td_policy(state, V)
        # action = monte_carlo_policy(state, V)

        # perform the selected action and store the next state information
        next_state, reward, terminated, truncated, info = env.step(action)
        # path.append(next_state)
        # G += gamma * reward
        # compute the value of the state
        V[state] += alpha * (reward_table[state] + gamma * V[next_state] - V[state])

        # update next state to the current state
        state = next_state

        print(V)

        # if the current state is the terminal state then break
        if terminated or truncated:
            # break
            state, info = env.reset()

    # V[state] = V[state] + alpha * G - V(state)

env.close()



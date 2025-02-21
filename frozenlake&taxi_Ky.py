import numpy as np
import pandas as pd
import gym
from warnings import filterwarnings
from collections import defaultdict
import random

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
# Create the environment
env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)
# env = gym.make('Taxi-v3', render_mode='human')
env.reset()

# Parameters
n_states = env.observation_space.n
n_actions = env.action_space.n
alpha = 0.01  # Learning rate
gamma = 0.01  # Discount factor
n_episodes = 10000
n_steps = 500
threshold = 1e-20

# designed_reward_table = [0,2,3,1,1,-10,4,-10,2,3,5,-10,-10,8,9,100]
# V = [1,1,1,1,1,-10,1,-10,1,1,1,-10,-10,1,1,100]
V = np.zeros(n_states)

'''TD algo contain these 2 defs'''
def value_iteration_TD(env):
    value_table = np.zeros(env.observation_space.n)
    # for every iteration
    for i in range(n_episodes):

        updated_value_table = np.copy(value_table)

        for s in range(env.observation_space.n):
            Q_values = [sum([prob*(r + gamma * updated_value_table[s_])
                            for prob, s_, r, _ in env.P[s][a]])
                        for a in range(env.action_space.n)]
            value_table[s] = max(Q_values)
            print(value_table)
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            break
        print(value_table)

    return value_table


def one_step_TD(state, V):
    A = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        for prob, nextState, reward, done in env.P[state][a]:
            A[a] += (reward + gamma * V[nextState])
    # print(np.argmax(A))
    return np.argmax(A)


'''MC algo contain these 3 defs'''
Q = defaultdict(float)
total_return = defaultdict(float) #storing the total return of the state-action pair
N = defaultdict(int) # dictionary for storing the count of the number of times a state-action pair is visited

def greedy_policy(state,Q):
    epsilon = 0.5
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])

def generate_episode(Q):

    #initialize a list for storing the episode
    episode = []

    #initialize the state using the reset function
    state,prob = env.reset()
    #then for each time step
    for t in range(n_steps):
        env.render()
        #select the action according to the epsilon-greedy policy
        action = greedy_policy(state,Q)
        print(action)
        #perform the selected action and store the next state information
        next_state, reward, done, info,_ = env.step(action)

        #store the state, action, reward in the episode list
        episode.append((state, action, reward))

        #if the next state is a final state then break the loop else update the next state to the current
        #state
        if done:
            break

        state = next_state

    return episode

def value_iteration_MC_one_visit(env):

    for i in range(n_episodes):

        #so, here we pass our initialized Q function to generate an episode
        episode = generate_episode(Q)

        #get all the state-action pairs in the episode
        all_state_action_pairs = [(s, a) for (s,a,r) in episode]

        #store all the rewards obtained in the episode in the rewards list
        rewards = [r for (s,a,r) in episode]

        #for each step in the episode
        for t, (state, action, reward) in enumerate(episode):

            #if the state-action pair is occurring for the first time in the episode
            if not (state, action) in all_state_action_pairs[0:t]:

                #compute the return R of the state-action pair as the sum of rewards
                R = sum(rewards[t:])

                #update total return of the state-action pair
                total_return[(state,action)] = total_return[(state,action)] + R

                #update the number of times the state-action pair is visited
                N[(state, action)] += 1

                #compute the Q value by just taking the average
                Q[(state,action)] = total_return[(state, action)] / N[(state, action)]
                print(Q)
    return Q


'''Run either part A or B, not at the same time'''

'''Part A: adjust n_episodes and n_steps for training to obtain value_table'''
# Run below for one_step_TD FrozenLake-v1
value_table = value_iteration_TD(env)
states, prob = env.reset()
for i in range(n_steps):
    env.render()
    # action = one_step_TD(states, value_table)
    action= one_step_TD(states, value_table)
    # print(action)
    next_state, reward, terminated, truncated, _ = env.step(action)
    states = next_state
    if terminated or truncated:
        # break
        states, prob = env.reset()

env.close()


'''Part B: adjust n_episodes and n_steps for training to obtain Q_table'''
#Run below for Monte Carlo Taxi-v3
# Q_table = value_iteration_MC_one_visit(env)
# value_table = value_iteration_TD(env)
# states, prob = env.reset()
# for i in range(n_steps):
#     env.render()
#     action= one_step_TD(states, value_table)
#     # action= one_step_TD(states, designed_reward_table)
#     action = greedy_policy(states, Q_table)
#     print(action)
#     next_state, reward, terminated, truncated, _ = env.step(action)
#     states = next_state
#     if terminated or truncated:
#         break
#         states, prob = env.reset()
#
# env.close()


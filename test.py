from Env import CabDriver
import numpy as np

c = CabDriver()
Time_matrix = np.load("TM.npy")

# get initiat state 
initial_state = c.state_init

# get action
index_actions = c.requests(initial_state)
print(f'index_actions: {index_actions}')

# step and reward
actions = index_actions[1]
random_action = actions[np.random.randint(len(actions))]
r = c.reward_func(initial_state, random_action, Time_matrix)
print(f'random action: {random_action}')
print(f'reward: {r}')

# next state
next_state = c.next_state_func(initial_state, random_action, Time_matrix)
print(f'next state: {next_state}')
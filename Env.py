# Import routines

from os import stat
import numpy as np
import math
import random
from gym import spaces
from numpy.lib.function_base import select

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
lamda_A=2    #lambda for poisson distribution
lamda_B=12
lamda_C=4
lamda_D=7
lamda_E=8

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.day = np.random.choice((0,1,2,3,4,5,6))
        self.time = np.random.choice(np.arange(0,t))
        self.location = np.random.choice(np.arange(0,m))
        self.action_space = spaces.Tuple((spaces.Discrete(5),spaces.Discrete(5)))
        self.state_space = (self.location, self.time, self.day)
        self.state_init = (self.location, self.time, self.day)
        self.time_matrix = np.load('TM.npy')
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input
    # def state_encod_arch1(self, state):
    #     """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
    #     return state_encod

    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        state_encod = np.array([1, 2, 3]) # No clue what this means should figure out
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(lamda_A)
        if location == 1:
            requests = np.random.poisson(lamda_B)
        if location == 2:
            requests = np.random.poisson(lamda_C)
        if location == 3:
            requests = np.random.poisson(lamda_D)
        if location == 4:
            requests = np.random.poisson(lamda_E)
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space.sample() for i in possible_actions_index] # why do we need possible_actions_index we can create possible actions by sampling from actions space
        actions.append([0,0])
        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        current_location = state[0]
        start_location = action[0]
        end_location = action[1]
        hour_of_day = state[1]
        day_of_week = state[2]
        revenue = R * Time_matrix[start_location][end_location][hour_of_day][day_of_week]
        cost_of_trip = C * (Time_matrix[start_location][end_location][hour_of_day][day_of_week] + Time_matrix[current_location][start_location][hour_of_day][day_of_week])
        cost_of_idle = C * (Time_matrix[0][0][hour_of_day][day_of_week])
        reward =  revenue -  cost_of_trip - cost_of_idle
        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        end_location = action[1]
        start_location = action[0]
        hour_of_day = state[1]
        day_of_week = state[2]
        next_state_location = end_location # this is the destination city
        # time in next state = current time + time it takes to reach next state
        next_state_hour_of_day = hour_of_day + Time_matrix[start_location][end_location][hour_of_day][day_of_week]
        # day in next state = next day ? if current time + time it takes to reach next state is > 24 or current day
        next_state_day_of_week = (day_of_week + 1) if next_state_hour_of_day > 24 else day_of_week
        return (next_state_location, next_state_hour_of_day, next_state_day_of_week)


    def reset(self):
        return self.action_space, self.state_space, self.state_init

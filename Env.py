# Import routines

from os import stat
import numpy as np
import random
from gym import spaces
from numpy.lib.function_base import select
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m
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
        self.day = np.random.choice(np.arange(0, d))
        self.time = np.random.choice(np.arange(0,t))
        self.location = np.random.choice(np.arange(0,m))
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
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
    def state_encod_arch2(self, state):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. 
        Hint: The vector is of size m + t + d"""
        state_encod = np.array([0 for _ in range(0, m + t + d)])
        state_encod[state[0]] = 1 # start location
        state_encod[m + state[1]] = 1 # hour of the day
        state_encod[m + t + state[2]] = 1 # day of the week
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

        possible_actions_index = [0] + random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        return possible_actions_index,actions   

    def get_updated_time(self, time, day):
        if(time > 23):
            time = time % 24
            day = day + 1
            if (day >=7):
                day = day % 7
        return time, day

    def step(self, state, action, Time_matrix, steps):
        """Takes in state, action, Time-matrix and steps and returns the state, reward, next_state, trip_hours
        
        Args:
        - state : current state
        - action: the action agent wants to take
        - time matrix: the function of time taken for travel
        - steps: steps is the hours completed so far. 
        """
        # print(f'step action..{action}')
        # when user is at A 10 AM On Monday, and receives a request for (B,C)
        current_location = state[0] # A
        start_location = action[0] # B
        end_location = action[1] # C
        current_hour_of_day = state[1] # 10
        current_day_of_week = state[2] # 1
        reward = 0

        # check if the action is to no_ride
        if action[0] == 0 and action[1] == 0:

            next_state_location = current_location
            next_state_hour_of_day = current_hour_of_day + 1
            next_state_day_of_week = current_day_of_week
            next_state_hour_of_day, next_state_day_of_week = self.get_updated_time(next_state_hour_of_day, next_state_day_of_week)
            
            hours_of_trip = 1 # 1 hour the driver is idle
            reward = -C
        
        elif action[0] == state[0]: # if the drive is at the pick up location

            next_state_location = end_location # this is the destination city
            start_location_to_end_location = int(Time_matrix[start_location][end_location][current_hour_of_day][current_day_of_week])
            next_state_hour_of_day = current_hour_of_day + start_location_to_end_location
            next_state_day_of_week = current_day_of_week
            
            # the time it takes to complete the trip from the start hour could be more than 24 hours, hence
            next_state_hour_of_day, next_state_day_of_week = self.get_updated_time(next_state_hour_of_day, next_state_day_of_week)
            
            hours_of_trip = start_location_to_end_location

        else:

            next_state_location = end_location # this is the destination city

            # the time it takes to reach the pick up location
            current_location_to_start_location = int(Time_matrix[current_location][start_location][current_hour_of_day][current_day_of_week])

            # If it takes 1 hour to each reach B from A, this is 11, 
            start_hour_of_day = current_hour_of_day + current_location_to_start_location
            start_day_of_week = current_day_of_week

            # this can be more than 23 in which case its next day
            start_hour_of_day, start_day_of_week = self.get_updated_time(start_hour_of_day, start_day_of_week)

            # this is the time it takes to reach B -> C
            start_location_to_end_location = int(Time_matrix[start_location][end_location][start_hour_of_day][start_day_of_week])
            
            # hour of day in next state = current time + time it takes to reach start location + time it takes to reach destination
            next_state_hour_of_day = current_hour_of_day + current_location_to_start_location + start_location_to_end_location
            next_state_day_of_week = start_day_of_week

            # the time it takes to complete the trip from the start hour could be more than 24 hours, hence
            next_state_hour_of_day, next_state_day_of_week = self.get_updated_time(next_state_hour_of_day, next_state_day_of_week)

            hours_of_trip = current_location_to_start_location + start_location_to_end_location

            # reward calculation
            # reward = revenue (B, C) - cost of (A-> B and B-> C)
            revenue = R * start_location_to_end_location
            cost_of_trip = C * (hours_of_trip)
            reward =  revenue - cost_of_trip
        
        # print(f'hours_of_trip...{hours_of_trip}')
        next_state = (next_state_location, next_state_hour_of_day, next_state_day_of_week)
        return state, reward, next_state, hours_of_trip



    def reset(self):
        return self.action_space, self.state_space, self.state_init

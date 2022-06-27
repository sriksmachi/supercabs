import gym
from gym.utils import seeding
import os
from os import stat
import numpy as np
import random
from gym import spaces
from numpy.lib.function_base import select
from itertools import permutations
from gym.utils import seeding
from gym import spaces

class ContosoCabs_v0 (gym.Env):
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
    MAX_STEPS = 10
    hoursoftrip = 0
    reward = 0
    done = False
    info = {}
    metadata = {
        "render.modes": ["human"]
    }

    def tm(self):
        """Time Matrix acts as the random matrix with random distance computed for (source,destination) pairs"""
        return np.random.randint(1, 24, (5,5,24,7))

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.day = np.random.choice(np.arange(0, self.d))
        self.time = np.random.choice(np.arange(0,self.t))
        self.location = np.random.choice(np.arange(0,self.m))
        self.action_space = spaces.Tuple((spaces.Discrete(self.m), spaces.Discrete(self.m)))
        self.action_space_values = [(p,q) for p in range(self.m) for q in range(self.m)]
        self.observation_space = spaces.Tuple((spaces.Discrete(self.m), spaces.Discrete(self.t), spaces.Discrete(self.d)))
        self.state_init = (self.location, self.time, self.day)
        self.state = self.state_init
        self.time_matrix = self.tm()
        self.episode_length = 24*30
        # Start the first round
        self.reset()

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(self.lamda_A)
        if location == 1:
            requests = np.random.poisson(self.lamda_B)
        if location == 2:
            requests = np.random.poisson(self.lamda_C)
        if location == 3:
            requests = np.random.poisson(self.lamda_D)
        if location == 4:
            requests = np.random.poisson(self.lamda_E)
        if requests >15:
            requests =15

        possible_actions_index = [0] + random.sample(range(1, (self.m-1)*self.m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        return possible_actions_index,actions   

    def get_updated_time(self, time, day):
        if(time > 23):
            time = time % 24
            day = day + 1
            if (day >=7):
                day = day % 7
        return time, day

    def step(self, action):
        """Takes in state, action and returns the state, reward, next_state, trip_hours
        
        Args:
        - action: the action agent wants to take
        """
        # print(f'step action..{action}')
        # when user is at A 10 AM On Monday, and receives a request for (B,C)
        current_location = self.state[0] # A
        start_location = action[0] # B
        end_location = action[1] # C
        current_hour_of_day = self.state[1] # 10
        current_day_of_week = self.state[2] # 1
        reward = 0

        # check if the action is invalid
        if action[0] == action[1]:

            next_state_location = current_location
            next_state_hour_of_day = current_hour_of_day + 1
            next_state_day_of_week = current_day_of_week
            next_state_hour_of_day, next_state_day_of_week = self.get_updated_time(next_state_hour_of_day, next_state_day_of_week)
            
            hours_of_trip = 1 # 1 hour the driver is idle
            reward = -self.C
        
        elif action[0] == current_location: # if the drive is at the pick up location

            # this is the destination city
            next_state_location = end_location 
            start_location_to_end_location = int(self.time_matrix[start_location][end_location][current_hour_of_day][current_day_of_week])
            next_state_hour_of_day = current_hour_of_day + start_location_to_end_location
            next_state_day_of_week = current_day_of_week
            
            # the time it takes to complete the trip from the start hour could be more than 24 hours, hence
            next_state_hour_of_day, next_state_day_of_week = self.get_updated_time(next_state_hour_of_day, next_state_day_of_week)
            hours_of_trip = start_location_to_end_location
            revenue = self.R * start_location_to_end_location
            cost_of_trip = self.C * (hours_of_trip)
            reward =  revenue - cost_of_trip

        else:
            # this is the destination city
            next_state_location = end_location 

            # the time it takes to reach the pick up location
            current_location_to_start_location = int(self.time_matrix[current_location][start_location][current_hour_of_day][current_day_of_week])

            # If it takes 1 hour to each reach B from A, this is 11, 
            start_hour_of_day = current_hour_of_day + current_location_to_start_location
            start_day_of_week = current_day_of_week

            # this can be more than 23 in which case its next day
            start_hour_of_day, start_day_of_week = self.get_updated_time(start_hour_of_day, start_day_of_week)

            # this is the time it takes to reach B -> C
            start_location_to_end_location = int(self.time_matrix[start_location][end_location][start_hour_of_day][start_day_of_week])
            
            # hour of day in next state = current time + time it takes to reach start location + time it takes to reach destination
            next_state_hour_of_day = current_hour_of_day + current_location_to_start_location + start_location_to_end_location
            next_state_day_of_week = start_day_of_week

            # the time it takes to complete the trip from the start hour could be more than 24 hours, hence
            next_state_hour_of_day, next_state_day_of_week = self.get_updated_time(next_state_hour_of_day, next_state_day_of_week)

            hours_of_trip = current_location_to_start_location + start_location_to_end_location

            # reward calculation
            # reward = revenue (B, C) - cost of (A-> B and B-> C)
            revenue = self.R * start_location_to_end_location
            cost_of_trip = self.C * (hours_of_trip)
            reward =  revenue - cost_of_trip
        
        self.reward = reward
        self.state = (next_state_location, next_state_hour_of_day, next_state_day_of_week)
        self.info["hours_of_trip"] += hours_of_trip # steps is hours completed so far.
        # if the driver travelled 30 hours terminate episode.
        self.done = self.info["hours_of_trip"] >= self.episode_length
        return [self.state, self.reward, self.done, self.info]

    def reset(self):
        self.state = (self.location, self.time, self.day)
        self.hoursoftrip = 0
        self.reward = 0
        self.done = False
        self.info = {}
        self.info["hours_of_trip"] = 0
        return self.state

    def render(self, mode="human"):
        s = "state: {}  reward: {:2d}  info: {}"
        print(s.format(self.state, self.reward, self.info))
    
    def seed (self, seed=None):
        """ Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
            number generators. The first value in the list should be the
            "main" seed, or the value which a reproducer should pass to
            'seed'. Often, the main seed equals the provided 'seed', but
            this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close (self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

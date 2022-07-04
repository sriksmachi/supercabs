from os import stat, walk
import numpy as np
import random
from itertools import permutations
from azureml.core import Run

import time
import shutil
import subprocess
import sys

import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray.tune as tune
from ray.rllib import train
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import ApexTrainer, APEX_DEFAULT_CONFIG
from contosocabs_env import ContosoCabs_v0

def on_train_result(info):
    '''Callback on train result to record metrics returned by trainer.
    '''
    run = Run.get_context()
    run.log(
        name='episode_reward_mean',
        value=info["result"]["episode_reward_mean"])
    run.log(
        name='episodes_total',
        value=info["result"]["episodes_total"])

def merge_dict(config, args):
    for key, value in config.items():
        if key in args:
            config[key] = args[key]
    return config

def initiate_train():
    select_env = "contosocabs-v0"
    register_env(select_env, lambda _: ContosoCabs_v0())
    ray.init(ignore_reinit_error=True)
    training_algorithm = "APEX"
    config = APEX_DEFAULT_CONFIG
    args = {
        'num_gpus' : 0,
        'num_workers' : 0,
        'num_cpus_per_worker' : 0,
        'env' : select_env
    }
    config = merge_dict(config, args)
    tune.run(
            run_or_experiment=training_algorithm,
            config=config,
            local_dir='./logs')

def main():
    initiate_train()

if __name__ == "__main__":
    main()


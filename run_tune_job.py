from os import stat, walk
import numpy as np
import random
from itertools import permutations
from azureml.core import Run

import time
import logging
from pyrsistent import m
from typing import Dict, Tuple

import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray.tune as tune
from ray.rllib import train
from ray.tune.registry import register_env
from ray.rllib.agents.dqn.apex import ApexTrainer, APEX_DEFAULT_CONFIG
from ray_on_aml.core import Ray_On_AML
from contosocabs_env import ContosoCabs_v0
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class Callbacks(DefaultCallbacks):
    def on_episode_end( self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs):
        '''Callback on train result to record metrics returned by trainer.
        '''
        run = Run.get_context()
        if episode is None:
            pass
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        print(f'result: {episode}')
        run.log(
            name='episode_reward_mean',
            value=sum(episode.agent_rewards.values())/len(episode.agent_rewards.keys()))
        run.log(
            name='episode_reward_max',
            value=max(episode.agent_rewards.values()))
        run.log(
            name='episode_length',
            value=episode.length)

def merge_dict(config, args):
    for key, _ in config.items():
        if key in args:
            config[key] = args[key]
    return config

def initiate_train():
    
    args = train.create_parser().parse_args()

    # Mapping configuration
    config = {}
    if args.run == "APEX":
        config = APEX_DEFAULT_CONFIG
    config["env"] = ContosoCabs_v0
    config["log_level"] = "INFO"
    config["callbacks"] = Callbacks
    config = merge_dict(config, args.config)
    print(f'config: {config}')

    tune.run(
            run_or_experiment=args.run,
            config=config,
            stop=args.stop,
            local_dir='./logs')

def main():
    ray_on_aml = Ray_On_AML()
    # If running on a GPU cluster use ray_on_aml.getRay(gpu_support=True)
    ray = ray_on_aml.getRay()
    if ray: #in the headnode
        logger.info("head node detected")
        time.sleep(15)
        print(ray.cluster_resources())
        initiate_train()
    else:
        logger.info("in worker node")

if __name__ == "__main__":
    main()


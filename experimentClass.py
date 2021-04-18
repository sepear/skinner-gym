import gym
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.evaluation import evaluate_policy
from ensemble import Ensemble
from loader import loader
from collections import defaultdict

from random_agent import RandomAgent
from matplotlib import pyplot as plt
import numpy as np

default_envs = ['MsPacmanNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
default_algos = ['DQN', 'PPO2', 'A2C']


class Experiment:
    def __init__(self,name,policy,timesteps, env_list=default_envs, algos_list=default_algos):

        self.name=name
        self.policy=policy
        self.timesteps=timesteps
        self.env_list = env_list
        self.algos_list = algos_list

        #self.n_algos = len(self.algos_list)
        self.models=dict()#(algo,env) as key, o quizá anidad



    def train(self):

        for algo in self.algos_list: 

            for env_name in self.env_list:
                #model = build_model(algo, env_name, self.policy, self.timesteps)
                env = gym.make(env_name)
                my_string=algo+"(policy=self.policy,env=env)"
                print(my_string)
                model=eval(algo+"(policy=self.policy,env=env)")





    def eval(self, n_eval=5, verbose=True):  # TODO add parameter to build ensemble
        pass
    
    def save(self):
        pass

    def show(self, metric='mean'):
        pass


if __name__ == '__main__':
    pass


import gym
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.evaluation import evaluate_policy
from ensemble import Ensemble
from experiment import Experiment
from loader import loader
from collections import defaultdict

from random_agent import RandomAgent
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    experimento = Experiment(
        env_list=['MsPacmanNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4'],
        algos_list=['dqn', 'ppo2', 'a2c'])

    experimento.run(n_eval=2, verbose=True)
    experimento.show()

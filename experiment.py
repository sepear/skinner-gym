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
default_algos = ['dqn', 'ppo2', 'a2c']


class Experiment:
    def __init__(self, env_list=default_envs, algos_list=default_algos):
        self.env_list = env_list
        self.algos_list = algos_list
        self.n_algos = len(self.algos_list)
        self.envs = dict()
        self.rewards = defaultdict(dict)
        self.models = defaultdict(dict)  # HAY QUE GUARDAR LOS MODELOS PARA ENSEMBLE

        for env_name in self.env_list:
            new_env = make_atari_env(env_name, num_env=1, seed=0)
            new_env = VecFrameStack(new_env, n_stack=4)
            self.envs[env_name] = new_env

        for algo in self.algos_list:
            for env_name, env in self.envs.items():
                self.models[env_name][algo] = loader(algo, env_name)

    def train(self):
        pass

    def run(self, n_eval=5, verbose=True):  # TODO add parameter to build ensemble
        for algo in self.algos_list:
            for env_name, env in self.envs.items():
                mean_reward, std_reward = evaluate_policy(self.models[env_name][algo], env=env,
                                                          n_eval_episodes=n_eval)
                # rewards[env_name][algo] = (mean_reward, std_reward)
                self.rewards[env_name][algo] = (mean_reward, std_reward)
                if verbose:
                    print("============ Evaluación finalizada de " + algo + " en " + env_name + " ============")
                    print(f"mean_reward={mean_reward}\n\n")

    def show(self, metric='mean'):
        if metric == 'standard':
            index = 1
            word = "Mean"
        else:
            index = 0
            word = "Standard"
        plt.style.use("fivethirtyeight")

        # https://python-graph-gallery.com/11-grouped-barplot/
        width = 0.25

        r = list()
        r.append(np.arange(self.n_algos))
        for i in range(1, self.n_algos):
            r.append([x + width for x in r[-1]])

        for counter, env_name in enumerate(self.env_list):

            plt.bar(r[counter], [value[index] for value in self.rewards[env_name].values()], width=width,
                    label=self.env_list[counter])

        # Add xticks on the middle of the group bars
        plt.title(word + " Reward Comparison")
        # plt.xlabel("Algorithm")
        plt.xticks([r_ + width for r_ in range(self.n_algos)], self.algos_list)

        # box = ax.get_position()

        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='upper right', fontsize='x-small')
        plt.show()


if __name__ == '__main__':
    pass


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

if __name__ == '__main__':
    n_eval_episodes = 2
    env_names = ['MsPacmanNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
    algos = ['dqn', 'ppo2', 'a2c']
    envs = dict()
    for env_name in env_names:
        new_env = make_atari_env(env_name, num_env=1, seed=0)
        new_env = VecFrameStack(new_env, n_stack=4)
        envs[env_name] = new_env

    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])
    models = defaultdict(dict)  # HAY QUE GUARDAR LOS MODELOS PARA ENSEMBLE
    rewards = defaultdict(dict)
    for algo in algos:
        for env_name, env in envs.items():
            if algo == 'Ensemble':
                pass  # TODO: TANTO SIMPLE COMO PONDERADO,DEBE SER UN PARAMETRO APARTE QUE PREGUNTA SI QUIERES HACER TAMBIEN EL ENSEMBLE, NO VA EN LA LISTA

            else:
                models[env_name][algo] = loader(algo, env_name)

                mean_reward, std_reward = evaluate_policy(models[env_name][algo], env=env,
                                                          n_eval_episodes=n_eval_episodes)
                # rewards[env_name][algo] = (mean_reward, std_reward)
                rewards[env_name][algo] = mean_reward
                print("============ Evaluación finalizada de " + algo + " en " + env_name + " ============")
                print(f"mean_reward={mean_reward}")

    print(rewards.keys())

    plt.style.use("fivethirtyeight")
    # https://python-graph-gallery.com/11-grouped-barplot/
    width = 0.25
    r = list()
    r.append(np.arange(len(algos)))
    for i in range(1, len(algos)):
        r.append([x + width for x in r[-1]])

    for counter, env_name in enumerate(env_names):
        plt.bar(r[counter], list(rewards[env_name].values()), width=width, label=env_names[counter])

    # Add xticks on the middle of the group bars
    plt.title("Mean Reward Comparison")
    #plt.xlabel("Algorithm")
    plt.xticks([r_ + width for r_ in range(len(algos))], algos)

    # Create legend & Show graphic
    plt.legend()
    plt.show()

    # mean_reward_DQN, std_reward_DQN = evaluate_policy(model_DQN, env=env_pacman, n_eval_episodes=100)
    # mean_reward_PPO2, std_reward_PPO2 = evaluate_policy(model_PPO2, env=env_pacman, n_eval_episodes=100)
    # mean_reward_A2C, std_reward_A2C = evaluate_policy(model_A2C, env=env_pacman, n_eval_episodes=100)
    # mean_reward_Ensemble, std_reward_Ensemble = evaluate_policy(model_Ensemble, env=env_pacman, n_eval_episodes=100)
    # mean_reward_weighted_Ensemble, std_reward_weighted_Ensemble = evaluate_policy(model_weighted_Ensemble,
    #                                                                              env=env_pacman,
    #
    #                                                                              n_eval_episodes=100)
    # x = ["DQN", "PPO2", "A2C", "Simple Ensemble", "Weighted Ensemble"]
    # y = [mean_reward_DQN, mean_reward_PPO2, mean_reward_A2C, mean_reward_Ensemble, mean_reward_weighted_Ensemble]
    # print(f"{x[0]}: \n\tmean reward:{mean_reward_DQN}\n\tstandard reward:{std_reward_DQN}")

    # print(f"{x[1]}: \n\tmean reward:{mean_reward_PPO2}\n\tstandard reward:{std_reward_PPO2}")

    # print(f"{x[2]}: \n\tmean reward:{mean_reward_A2C}\n\tstandard reward:{std_reward_A2C}")

    # print(f"{x[3]}: \n\tmean reward:{mean_reward_Ensemble}\n\tstandard reward:{std_reward_Ensemble}")

    # print(
    #    f"{x[4]}: \n\tmean reward:{mean_reward_weighted_Ensemble}\n\tstandard reward:{std_reward_weighted_Ensemble}")

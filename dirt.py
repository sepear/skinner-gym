import gym
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.evaluation import evaluate_policy
from ensemble import Ensemble
from random_agent import RandomAgent
if __name__ == '__main__':
    env_pacman = make_atari_env('MsPacmanNoFrameskip-v4', num_env=1, seed=0)
    env_pacman = VecFrameStack(env_pacman, n_stack=4)
    print(f"n:envs:{env_pacman.num_envs}")
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    model_DQN = DQN.load("trained_agents/dqn/MsPacmanNoFrameskip-v4.pkl")
    model_PPO2 = PPO2.load("trained_agents/ppo2/MsPacmanNoFrameskip-v4.pkl")
    model_A2C = A2C.load("trained_agents/a2c/MsPacmanNoFrameskip-v4.pkl")
    model_Ensemble = Ensemble(models={model_DQN, model_PPO2, model_A2C}, env=env_pacman)
    model_Ensemble_weighted = Ensemble(models={model_DQN, model_PPO2, model_A2C}, method="weighted", env=env_pacman)
    model_Random = RandomAgent(env_pacman)

    # obs = env_pacman.reset()
    # while True:
    #    action, _states = model_DQN.predict(obs)
    #    obs, rewards, dones, info = env_pacman.step(action)
    #    env_pacman.render()

    mean_reward_DQN, std_reward_DQN = evaluate_policy(model_DQN, env=env_pacman, n_eval_episodes=10)
    mean_reward_PPO2, std_reward_PPO2 = evaluate_policy(model_PPO2, env=env_pacman, n_eval_episodes=10)
    mean_reward_A2C, std_reward_A2C = evaluate_policy(model_A2C, env=env_pacman, n_eval_episodes=10)
    mean_reward_Ensemble, std_reward_Ensemble = evaluate_policy(model_Ensemble, env=env_pacman, n_eval_episodes=10)
    mean_reward_Ensemble_weighted, std_reward_Ensemble_weighted = evaluate_policy(model_Ensemble_weighted, env=env_pacman,
                                                                                  n_eval_episodes=10)
    #mean_reward_Random, std_reward_Random = evaluate_policy(model_Random, env=env_pacman, n_eval_episodes=10)


    print(f"DQN: \n\tmean reward:{mean_reward_DQN}\n\tstandard reward:{std_reward_DQN}")

    print(f"PPO2: \n\tmean reward:{mean_reward_PPO2}\n\tstandard reward:{std_reward_PPO2}")

    print(f"A2C: \n\tmean reward:{mean_reward_A2C}\n\tstandard reward:{std_reward_A2C}")

    print(f"Ensemble: \n\tmean reward:{mean_reward_Ensemble}\n\tstandard reward:{std_reward_Ensemble}")

    print(f"Ensemble Weighted: \n\tmean reward:{mean_reward_Ensemble_weighted}\n\tstandard reward:{std_reward_Ensemble_weighted}")

    #print(f"Random: \n\t mean reward:{mean_reward_Random}\n\tstandard reward:{std_reward_Random}")
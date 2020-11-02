# Unify the process of model loading
from stable_baselines import PPO2, DQN, A2C, ACER, TRPO


def loader(algo, env_name):
    if algo == 'dqn':
        return DQN.load("trained_agents/" + algo + "/" + env_name + ".pkl")
    elif algo == 'ppo2':
        return PPO2.load("trained_agents/" + algo + "/" + env_name + ".pkl")
    elif algo == 'a2c':
        return A2C.load("trained_agents/" + algo + "/" + env_name + ".pkl")
    elif algo == 'acer':
        return ACER.load("trained_agents/" + algo + "/" + env_name + ".pkl")
    elif algo == 'trpo':
        return TRPO.load("trained_agents/" + algo + "/" + env_name + ".pkl")
    # TODO: add more in the future?

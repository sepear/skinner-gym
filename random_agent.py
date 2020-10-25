import gym


class RandomAgent:

    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, obs, state=None, mask=None, deterministic=None):
        return self.action_space.sample(), None
        # random action from the action space

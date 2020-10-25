# Only tested with Stable-Baselines models
# Recurrent policies not tested
from collections import defaultdict
import numpy as np
from stable_baselines.common.evaluation import evaluate_policy


class Ensemble:
    def __init__(self, models, env, method="majority"):
        self.models = models
        self.method = method
        self.env = env
        if self.method == "weighted":
            self.weights = defaultdict(float)
            self.weights = {model: evaluate_policy(model, env=env, n_eval_episodes=10)[0] for model in models}

            self.total_weights = sum(self.weights.values())
            self.weights = {model: self.weights[model] / self.total_weights for model in models}

    def predict(self, obs, state=None, mask=None, deterministic=None):
        predictions = defaultdict(int)
        states = dict()
        for model in self.models:
            action, _state = model.predict(obs)
            action_tuple = tuple(action.tolist())  # arrays cant be keys
            states[action_tuple] = _state
            if self.method == "weighted":
                predictions[action_tuple] += 1*self.weights[model]
            else:
                predictions[action_tuple] += 1

        max_voted = max(predictions, key=predictions.get)
        state_voted = states[max_voted]
        return np.asarray(max_voted), state_voted

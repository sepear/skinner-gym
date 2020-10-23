# Only tested with Stable-Baselines models
#Recurrent policies not tested
from collections import defaultdict

class Ensemble:
    def __init__(self, models, method="majority"):
        self.models = models
        self.method = method


    def predict(self, obs):
        predictions = defaultdict(int)
        for model in self.models:
            action, _states = model.predict(obs)
            predictions[action]+=1
        return max(predictions, key=predictions.get)

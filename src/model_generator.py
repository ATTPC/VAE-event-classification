

class ModelGenerator:
    def __init__(self, model):
        self.hyperparam_vals = []
        self.loss_vals = []
        self.performance_vals = []
        self.model = model

    def generate_config(self,):
        model_inst, hyperparams = self._make_model()
        self.hyperparam_vals.append(hyperparams)
        return model_inst


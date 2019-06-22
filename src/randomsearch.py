from model_generator import ModelGenerator
import numpy as np
import os 
import sys
sys.path.append("../scripts")

class RandomSearch:
    def __init__(
            self,
            X,
            x_t,
            y_t,
            model_gen: ModelGenerator,
            clustering=True,
            architecture="vgg"
            ):
        """
        Parameters:
        -----------
        X: data to fit the model to
        x_t: data to measure performance on
        y_t: targets or labels to measure performance on
        model_gen: subclass of ModelGenerator
        """
        self.x_t = x_t
        self.y_t = y_t
        n_classes = len(np.unique(self.y_t))
        self.arch = architecture
        self.model_creator = model_gen(
                                X,
                                n_classes,
                                [self.x_t, self.y_t],
                                clustering,
                                architecture,
                                )

    def search(self, n, batch_size, save_dir):
        to_save = [
                self.model_creator.hyperparam_vals,
                self.model_creator.loss_vals,
                self.model_creator.performance_vals
                ]
        names = [
                "hyperparam_vals_"+self.arch+".npy",
                "loss_vals_"+self.arch+".npy",
                "performance_"+self.arch+".npy",
                ]
        for i in range(n):
            model_inst = self.model_creator.generate_config()
            lx, ly = self.model_creator.fit_model(model_inst, batch_size)
            performance  = self.model_creator.compute_performance(
                                    model_inst,
                                    self.x_t,
                                    self.y_t
                                    )
            self.savefiles(to_save, names, save_dir)

    def savefiles(self, to_save, names, save_dir):
        for o, n in zip(to_save, names):
            o = np.array(o)
            fn = os.path.normpath(save_dir+n)
            np.save(fn, o)

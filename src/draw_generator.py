import tensorflow as tf 
import numpy as np

from sklearn.metrics import  adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder

from model_generator import ModelGenerator
from draw import DRAW
from convae_generator import ConVaeGenerator

import sys
sys.path.append("../scripts")
import run


class DRAWGenerator(ModelGenerator):
    def __init__(
            self,
            X,
            n_classes=3,
            labelled_data=None,
            clustering=True,
            architecture="vgg",
            ):
        super().__init__(DRAW)
        self.architecture = architecture
        self.train_clustering = clustering
        self.labelled_data = labelled_data
        self.n_classes = n_classes
        self.read_write = ["attention", "conv"]
        self.max_layers = 7
        self.latent_types = ["include_KL", "include_MMD", None]
        self.activations = ["relu", "lrelu", "tanh"]
        self.etas = np.logspace(-5, -1, 5)
        if self.train_clustering:
            self.betas = np.linspace(0, 1, 10)
        else:
            self.betas = np.logspace(-1, 3, 5)
        self.ld = [3, 10, 20, 50, 100]
        self.recurrent_dim = [64, 128, 256, 512]
        self.deltas = np.linspace(0.5, 1.3, 50)
        self.N = [5, 12, 20, 40, 60]
        self.T = [3, 6, 10, 20, 40]
        #self.sd = [10, 50, 150]
        self.X = X

    def _make_model(self,):
        rw_type = self.read_write[np.random.randint(0,2)]
        config = self.sample_hyperparameters()
        T = self.T[np.random.randint(0, len(self.T))]
        dec_size = self.recurrent_dim[np.random.randint(0, len(self.recurrent_dim))]
        enc_size = self.recurrent_dim[np.random.randint(0, len(self.recurrent_dim))]

        if rw_type == "attention":
            attn_config = self._generate_attn_config()
            attn = True
            conv = None
            conv_architecture = {}
            config[0] = attn_config
        else:
            attn = None
            conv = True
            cc = config[0]
            conv_architecture = {
                    "filters": cc[0],
                    "kernel_size": cc[1],
                    "strides": cc[2],
                    "pool": cc[3],
                    "activation": [1,]*cc[4],
                    "n_layers": cc[4],
                    }
            act_func = self.activations[np.random.randint(0, len(self.activations))]
            conv_architecture["activation_func"] = act_func
            attn_config = {}

        parameters_config = config[1]
        beta = parameters_config[0]
        eta = parameters_config[1]
        beta1 = parameters_config[2]
        beta2 = parameters_config[3]
        latent_dim = parameters_config[4]

        mode_config = config[2]
        clustering_config = config[3]

        loss = ["mse", None][np.random.randint(0,2)]
        loss_kwds = {"reconst_loss":loss}
        if loss == "mse":
            beta /= 1e3

        model = self.model(
                T,
                dec_size,
                enc_size,
                latent_dim,
                self.X,
                beta=beta,
                train_classifier=False,
                use_conv=conv,
                conv_architecture=conv_architecture,
                use_attention=attn,
                attn_config=attn_config,
                mode_config=mode_config,
                clustering_config=clustering_config,
                labelled_data=self.labelled_data
                )

        opt = tf.train.AdamOptimizer
        opt_args =[eta,]
        opt_kwds = {
                "beta1":beta1,
                "beta2":beta2
                }

        if not conv:
            activation = self.activations[np.random.randint(0, len(self.activations))]
        else:
            activation = conv_architecture["activation_func"]

        config.append(loss)
        config.append(activation)
        config.append(T)

        graph_kwds = {
                "activation":activation,
                "n_encoder_cells": 1,
                "n_decoder_cells": 1,
                }

        model.compile_model(graph_kwds=graph_kwds, loss_kwds=loss_kwds)
        model.compute_gradients(opt, opt_args, opt_kwds)
        return model, config

    def _generate_attn_config(self):
        delta_r = self.deltas[np.random.randint(0, len(self.deltas))]
        delta_w = self.deltas[np.random.randint(0, len(self.deltas))]

        read_N = self.N[np.random.randint(0, len(self.N))]
        write_N = self.N[np.random.randint(0, len(self.N))]

        attn_config = {
                "read_N": read_N,
                "write_N": write_N,
                "write_N_sq": write_N**2,
                "delta_w": delta_w,
                "delta_r": delta_r,
                }
        return attn_config 


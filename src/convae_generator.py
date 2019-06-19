import tensorflow as tf 
import numpy as np
from model_generator import ModelGenerator
from convolutional_VAE import ConVae

class ConVaeGenerator(ModelGenerator):
    def __init__(self, X):
        super().__init__(ConVae)
        self.max_layers = 4 
        self.latent_types = ["include_KL", "include_MMD", None]
        self.etas = np.logspace(-5, -1, 5)
        self.betas = np.logspace(0, 5, 6)
        self.ld = [3, 10, 20, 50, 100, 250]
        self.X = X

    def sample_hyperparameters(self,):
        config = []
        n_layers = np.random.randint(
                        3,
                        self.max_layers
                        )
        valid_conv_out = 1
        input_dim = self.X.shape[1]
        while valid_conv_out == 1:
            conv_config = self._generate_conv_config(n_layers)
            valid_conv_out = self.conv_out(conv_config, input_dim)
        parameters_config = self._generate_param_config()
        n_latent_types = len(self.latent_types)
        latent = self.latent_types[np.random.randint(0, n_latent_types)]
        mode_config = {
                "simulated_mode": False,
                "restore_mode": False,
                "include_KL": False,
                "include_MMD": False,
                "include_KM:": False,
                }

        if not latent == None:
            mode_config[latent] = True
        clustering_config = {}

        config.append(conv_config)
        config.append(parameters_config)
        config.append(mode_config)
        config.append(clustering_config)
        return config

    def _generate_param_config(self,):
        beta = self.betas[np.random.randint(0, len(self.betas)-1)]
        eta = self.etas[np.random.randint(0, len(self.etas)-1)]
        beta1 = np.random.uniform(0.2, 0.96)
        beta2 = 0.99
        latent_dim = self.ld[np.random.randint(0, len(self.ld)-1)]
        parameters_config = [
                beta,
                eta,
                beta1,
                beta2,
                latent_dim
                ]
        return parameters_config

    def _generate_conv_config(self, n_layers):
        filter_sizes = np.array([3, 5, 7 ,9, 11])
        filter_architecture = 2**np.random.randint(1, 8, size=n_layers)
        which_kernels = np.random.randint(0, len(filter_sizes), size=n_layers)
        kernel_architecture = filter_sizes[which_kernels]
        strides_arcitecture = [1]*n_layers#np.random.randint(1, 3, size=n_layers)
        conv_config = [
                filter_architecture,
                kernel_architecture,
                strides_arcitecture,
                n_layers,
                ]

        print("CONV CONFIG", conv_config)
        return conv_config
    
    @classmethod
    def conv_out(self, conv_config, w):
        def o(w, k, s): return np.floor((w - k + 2*0)/s + 1)

        filter_a = conv_config[0]
        kernel_a = conv_config[1]
        strides_a = conv_config[2]
        n = conv_config[3]

        for l in range(n):
            k = kernel_a[l]
            s = strides_a[l]
            print("computed", w)
            w = o(w, k, s)
        
        print("final", w)
        if w < 2:
            return 1
        else:
            return 0

    def _make_model(self, ):
        config = self.sample_hyperparameters()
        conv_config = config[0]
        filter_architecture = conv_config[0]
        kernel_architecture = conv_config[1]
        strides_architecture = conv_config[2]
        n_layers = conv_config[3]

        parameters_config = config[1]
        beta = parameters_config[0]
        eta = parameters_config[1]
        beta1 = parameters_config[2]
        beta2 = parameters_config[3]
        latent_dim = parameters_config[4]

        mode_config = config[2]
        clustering_config = config[3]

        model = self.model(
                    n_layers,
                    filter_architecture,
                    kernel_architecture,
                    strides_architecture,
                    latent_dim,
                    self.X,
                    beta=beta,
                    mode_config=mode_config,
                    clustering_config=clustering_config
                )

        opt = tf.train.AdamOptimizer
        opt_args =[eta,]
        opt_kwds = {
                "beta1":beta1,
                "beta2":beta2
                }

        model.compile_model()
        model.compute_gradients(opt, opt_args, opt_kwds)
        return model, config

    def fit_model(self, model, batch_size,):
        sess = tf.InteractiveSession()
        lx, lz = model.train(
                    sess,
                    150,
                    "../drawing",
                    "../models", 
                    batch_size,
                    earlystopping=True
                )
        self.loss_vals.append((lx, lz))
        return lx, lz

    def compute_performance(self, model, x_t, y_t):
        perf = model.performance(x_t, y_t)
        self.performance_vals.append(perf)
        return perf


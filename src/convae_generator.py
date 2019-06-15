import tensorflow as tf 
from model_generator import ModelGenerator
from convolutional_VAE import ConVae

class ConVaeGenerator(ModelGenerator):
    def __init__(self, X):
        super().__init__(ConVae)
        self.make_hyperparam_grid()
        self.X = X

    def _make_model(self, ):
        config = self.sample_hyperparams()
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


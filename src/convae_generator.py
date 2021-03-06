import tensorflow as tf
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder

from model_generator import ModelGenerator
from convolutional_VAE import ConVae
from latent_classifier import test_model


class ConVaeGenerator(ModelGenerator):
    def __init__(
        self,
        X,
        n_classes=3,
        labelled_data=None,
        clustering=True,
        architecture="vgg",
        use_vgg_repr=False,
        target_images=None,
        use_dd=False,
        dd_targets=None,
    ):
        super().__init__(ConVae, use_vgg_repr, target_images, use_dd, dd_targets)
        self.architecture = architecture
        self.train_clustering = clustering
        self.labelled_data = labelled_data
        self.n_classes = n_classes
        self.max_layers = 7
        self.latent_types = ["include_KL", "include_MMD", None]
        self.activations = ["relu", "lrelu"]
        self.etas = np.logspace(-5, -1, 5)
        self.lambdas = np.linspace(0.0, 200, 50)
        if self.train_clustering:
            self.betas = np.linspace(0, 1, 10)
        else:
            self.betas = np.logspace(0, 4, 5)
        if self.use_vgg_repr:
            self.dense_layers = [1, 2, 3]
        self.ld = [3, 10, 20, 50, 100, 150, 200]
        self.cl_ld = [3, 5, 9, 11, 15, 20, 50, 100, 200]
        self.reg_strengths = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        # self.sd = [10, 50, 150]
        self.X = X

    def _make_model(self,):
        config = self.sample_hyperparameters()
        conv_config = config[0]
        filter_architecture = conv_config[0]
        kernel_architecture = conv_config[1]
        strides_architecture = conv_config[2]
        pooling_config = conv_config[3]
        n_layers = conv_config[4]

        parameters_config = config[1]
        beta = parameters_config[0]
        eta = parameters_config[1]
        beta1 = parameters_config[2]
        beta2 = parameters_config[3]
        latent_dim = parameters_config[4]
        reg_strength = parameters_config[5]
        # sampling_dim = parameters_config[5]

        mode_config = config[2]
        clustering_config = config[3]
        if self.architecture == "static":
            loss = "mse"
        else:
            loss = ["mse", None][np.random.randint(0, 2)]
        if loss == "mse":
            beta /= 1e3
            config[1][0] = beta
        model = self.model(
            n_layers,
            filter_architecture,
            kernel_architecture,
            strides_architecture,
            pooling_config,
            latent_dim,
            self.X.shape,
            beta=beta,
            # sampling_dim=sampling_dim,
            mode_config=mode_config,
            clustering_config=clustering_config,
            labelled_data=self.labelled_data,
            target_imgs=self.target_images,
        )
        if self.use_vgg_repr:
            n_dense = self.dense_layers[np.random.randint(0, len(self.dense_layers))]
            model.dense_layers = n_dense
            config[1].append(n_dense)
        if self.use_dd:
            lmbd = self.lambdas[np.random.randint(0, len(self.lambdas))]
            dense = np.random.randint(1, 3)
            model.lmbd = lmbd
            model.dd_targets = self.dd_targets
            model.dd_dense = dense
            config[1].append(lmbd)
            config[1].append(dense)

        opt = tf.train.AdamOptimizer
        opt_args = [eta]
        opt_kwds = {"beta1": beta1, "beta2": beta2}

        activation = self.activations[np.random.randint(0, len(self.activations))]
        if loss is None:
            out_act = "sigmoid"
        else:
            out_act = None

        graph_kwds = {"activation": activation, "output_activation": out_act}
        loss_kwds = {"reconst_loss": loss}
        config.append(loss)
        config.append(activation)
        tf.reset_default_graph()
        model.compile_model(graph_kwds=graph_kwds, loss_kwds=loss_kwds)
        model.compute_gradients(opt, opt_args, opt_kwds)
        return model, config

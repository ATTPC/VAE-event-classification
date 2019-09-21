import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from latent_classifier import test_model

import sys

sys.path.append("../scripts")
import run


def accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class ModelGenerator:
    def __init__(
        self,
        model,
        use_vgg_repr=False,
        target_images=None,
        use_dd=False,
        dd_targets=None,
    ):
        self.use_vgg_repr = use_vgg_repr
        if self.use_vgg_repr and target_images is None:
            raise ValueError("When using vgg reprsentation, must supply target images")
        self.use_dd = use_dd
        if self.use_dd and dd_targets is None:
            raise ValueError("When using dual decoders, must supply alt. repr")
        self.dd_targets = dd_targets
        self.target_images = target_images
        self.hyperparam_vals = []
        self.loss_vals = []
        self.performance_vals = []
        self.model = model
        self.init_run = run.run

    def generate_config(self,):
        model_inst, hyperparams = self._make_model()
        self.hyperparam_vals.append(hyperparams)
        return model_inst

    def sample_hyperparameters(self,):
        config = []
        n_layers = np.random.randint(3, self.max_layers)
        # n_layers = 7
        valid_conv_out = 1
        input_dim = self.X.shape[1]
        conv_config = self._generate_conv_config(n_layers)
        valid_conv_out = self.conv_out(conv_config, input_dim)
        parameters_config = self._generate_param_config()
        n_latent_types = len(self.latent_types)
        mode_config = {
            "simulated_mode": False,
            "restore_mode": False,
            "include_KL": False,
            "include_MMD": False,
            "include_KM": False,
            "batchnorm": False,
            "use_vgg": self.use_vgg_repr,
            "use_dd": self.use_dd,
        }

        if self.train_clustering:
            mode_config["include_KM"] = True
            clustering_config = self._generate_clustering_config()
        else:
            latent = self.latent_types[np.random.randint(0, n_latent_types)]
            if not latent == None:
                mode_config[latent] = True
            clustering_config = {}

        use_bm = np.random.randint(0, 2)
        if use_bm:
            mode_config["batchnorm"] = True

        config.append(conv_config)
        config.append(parameters_config)
        config.append(mode_config)
        config.append(clustering_config)
        return config

    def _generate_param_config(self,):
        beta = self.betas[np.random.randint(0, len(self.betas))]
        eta = self.etas[np.random.randint(0, len(self.etas))]
        beta1 = np.random.uniform(0.2, 0.96)
        beta2 = 0.99
        latent_dim = self.ld[np.random.randint(0, len(self.ld))]
        if self.train_clustering:
            latent_dim = self.cl_ld[np.random.randint(0, len(self.cl_ld))]
        # sampling_dim = self.sd[np.random.randint(0, len(self.sd))]
        reg_strength = self.reg_strengths[np.random.randint(0, len(self.reg_strengths))]
        parameters_config = [
            beta,
            eta,
            beta1,
            beta2,
            latent_dim,
            # sampling_dim,
            reg_strength,
        ]
        return parameters_config

    def _generate_conv_config(self, n_layers):
        if self.architecture == "vgg":
            strides_arcitecture = [1] * n_layers
            kernel_architecture = [3] * n_layers
            filter_architecture = self._make_vgg_filters(kernel_architecture)
            pooling_config = self._make_vgg_pooling_config(n_layers)
        elif self.architecture == "static":
            n_layers = 4
            strides_arcitecture = [2] * n_layers
            kernel_architecture = [5, 5, 3, 3]
            filter_architecture = self._make_filter_config(kernel_architecture)
            pooling_config = [0] * n_layers
        elif self.architecture == "ours":
            strides_arcitecture = [
                2
            ] * n_layers  # np.random.randint(1, 3, size=n_layers)
            kernel_architecture = self._make_kernel_config(n_layers)
            filter_architecture = self._make_filter_config(kernel_architecture)
            pooling_config = self._make_pooling_config(n_layers)

        conv_config = [
            filter_architecture,
            kernel_architecture,
            strides_arcitecture,
            pooling_config,
            n_layers,
        ]
        return conv_config

    def _make_vgg_filters(self, kernel_architecture):
        n_layers = len(kernel_architecture)
        base_filters = 2 ** np.random.randint(1, 4)
        filter_architecture = np.array([base_filters] * n_layers)
        where_double = [2, 4, 7, 10, 13]

        for i in range(n_layers):
            if i in where_double:
                filter_architecture[i:] *= 2

        return filter_architecture

    def _make_pooling_config(self, n_layers):
        # pooling_conf = np.random.randint(0, 2, n_layers)
        pooling_conf = [0] * n_layers
        return pooling_conf

    def _make_vgg_pooling_config(self, n_layers):
        pooling_config = [0] * n_layers
        where_pool = np.array([2, 5, 7, 10, 13]) - 1

        for i in range(n_layers):
            if i in where_pool:
                pooling_config[i] = 1

        return pooling_config

    def _generate_clustering_config(self,):
        clustering_config = {
            "n_clusters": self.n_classes,
            "alpha": 1,
            "delta": 0.01,
            "self_supervise": False,
        }
        pre_epochs = [10, 50, 100, 200, 300]
        pretrain_epochs = pre_epochs[np.random.randint(0, len(pre_epochs))]
        update_freq = [1, 50, 150, 200]
        update_interval = update_freq[np.random.randint(0, len(update_freq))]
        # pretrain_sim = np.random.randint(0, 2)
        pretrain_sim = 0
        if self.n_classes == 2:
            pretrain_sim = 0

        if pretrain_sim:
            X_sim = np.load("../data/simulated/pr_test_simulated.npy")[0:1000]
            y_sim = np.load("../data/simulated/test_targets.npy")[0:1000]
            oh = OneHotEncoder(sparse=False)
            y_sim = oh.fit_transform(y_sim.reshape(-1, 1))
            if self.n_classes > len(np.unique(y_sim)):
                tmp = np.zeros(np.array(y_sim.shape) + [0, 1])
                tmp[:, :-1] = y_sim
                y_sim = tmp
            clustering_config["X_c"] = X_sim
            clustering_config["Y_c"] = y_sim

        clustering_config["pretrain_simulated"] = pretrain_sim
        clustering_config["pretrain_epochs"] = pretrain_epochs
        clustering_config["update_interval"] = update_interval
        return clustering_config

    def _make_filter_config(self, kernel_architecture):
        n_layers = len(kernel_architecture)
        filter_architecture = []
        max_exp = 6
        filter_exp = np.random.randint(3, max_exp)
        n_filters = 2 ** filter_exp
        k = kernel_architecture[0]

        for i in range(n_layers):
            if kernel_architecture[i] != k:
                k = kernel_architecture[i]
                n_filters = 2 ** np.random.randint(filter_exp, max_exp)
                filter_exp += 1
                filter_architecture.append(n_filters)
            else:
                filter_architecture.append(n_filters)

        return filter_architecture

    def _make_kernel_config(self, n_layers):
        kernel_sizes = np.array([17, 15, 13, 11, 9, 7, 5, 3])
        kernel_sizes = kernel_sizes[np.random.randint(0, len(kernel_sizes) - 1) :]
        available_layers = n_layers
        n_of_each_kernel = []

        for k in kernel_sizes:
            if k == kernel_sizes[-1]:
                n_of_k = available_layers
            else:
                n_of_k = np.random.randint(0, available_layers)
            available_layers -= n_of_k
            n_of_each_kernel.append(n_of_k)

        kernel_architecture = []
        for i, n in enumerate(n_of_each_kernel):
            kernel_architecture += [kernel_sizes[i]] * n

        return kernel_architecture

    def fit_model(self, model, batch_size):
        self.sess = tf.InteractiveSession()
        with open("../scripts/run.py", "w") as fo:
            fo.write("run={}".format(run.run + 1))
        lx, lz = model.train(
            self.X,
            self.sess,
            100,
            batch_size,
            earlystopping=True,
            run=self.init_run,
            save_checkpoints=False,
            verbose=1,
        )
        self.loss_vals.append((lx, lz))
        self.init_run += 1
        return lx, lz

    def compute_performance(self, model, x_t, y_t):
        if self.train_clustering:
            y_pred, targets = model.predict_cluster(self.sess)
            ars = adjusted_rand_score(targets, y_pred)
            acc = accuracy(targets, y_pred)
            perf = (ars, acc)
        else:
            perf = test_model(x_t, y_t, model, self.sess)
        self.performance_vals.append(perf)
        return perf

    @classmethod
    def conv_out(self, conv_config, w):
        def o(w, k, s):
            return np.floor((w - k + (k - 1) / 2) / s + 1)

        def to(w, k, s):
            return np.ceil((w - 1) * s + 1)

        filter_a = conv_config[0]
        kernel_a = conv_config[1]
        strides_a = conv_config[2]
        pool_a = conv_config[3]
        to_compare = int(w)
        n = conv_config[4]

        for l in range(n):
            k = kernel_a[l]
            s = strides_a[l]
            w = o(w, k, s)
            if pool_a[l]:
                w /= 2

        if w < 2:
            return 1

        for l in reversed(range(n)):
            if pool_a[l]:
                w *= 2
            k = kernel_a[l]
            s = strides_a[l]
            w = to(w, k, s)

        if w == to_compare:
            return 0
        else:
            return 1

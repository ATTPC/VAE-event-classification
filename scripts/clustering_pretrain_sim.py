import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append("../src")

from convolutional_VAE import ConVae


X_sim = np.load("../data/simulated/pr_test_simulated.npy")[0:100]
y_sim = np.load("../data/simulated/test_targets.npy")[0:100]

run_130 = np.load("../data/clean/images/run_0130_label_False.npy")[:200]
run_150 = np.load("../data/clean/images/run_0130_label_False.npy")[:200]
X = np.concatenate([run_130, run_150])

x_train = np.load("../data/clean/images/test.npy")
y_train = np.load("../data/clean/targets/test_targets.npy")

n_layers = 4
filter_architecture = [20, 80, 30, 5]
kernel_arcitecture = [3, 3, 3, 3]
strides_architecture = [1, 1, 1, 1]
epochs = 25

latent_dim = 10
batch_size = 100

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": False,
        "include_KM": True
        }

clustering_config = {
        "n_clusters":3,
        "alpha":1,
        "X_c": X_sim,
        "Y_c": y_sim,
        "pretrain_simulated":True,
        }

cvae = ConVae(
        n_layers,
        filter_architecture,
        kernel_arcitecture,
        strides_architecture,
        latent_dim,
        X,
        beta=10000,
        mode_config=mode_config,
        clustering_config=clustering_config,
        labelled_data=[x_train, y_train]
        )

graph_kwds = {
        "activation":"tanh"
        }
cvae.compile_model(graph_kwds=graph_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-4, ]
opt_kwds = {
    "beta1": 0.8,
}

cvae.compute_gradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()
lx, lz = cvae.train(
        sess,
        epochs,
        "../drawing",
        "../models",
        batch_size,
        earlystopping=True,
        )


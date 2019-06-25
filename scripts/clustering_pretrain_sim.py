import tensorflow as tf
import numpy as np
import run
import os

from sklearn.preprocessing import OneHotEncoder

import sys
import h5py
sys.path.append("../src")

from convolutional_VAE import ConVae

print("PID:", os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

X_sim = np.load("../data/simulated/pr_test_simulated.npy")[0:1000]
y_sim = np.load("../data/simulated/test_targets.npy")[0:1000]

oh = OneHotEncoder(sparse=False)
y_sim = oh.fit_transform(y_sim.reshape(-1, 1))
tmp = np.zeros(np.array(y_sim.shape) + [0, 1]) 
tmp[:, :-1] = y_sim
y_sim = tmp

data = "noisy"

if data=="noisy":
    noisy_130 = np.load("../data/processed/all_0130.npy")[:2000]
    X = noisy_130
    x_train = np.load("../data/processed/train.npy")
    with h5py.File("../data/images.h5", "r") as fo:
        y_train = np.array(fo["train_targets"])
else:
    run_130 = np.load("../data/clean/images/run_0130_label_False.npy")[:1000]
    run_150 = np.load("../data/clean/images/run_0150_label_False.npy")[:1000]
    X = np.concatenate([run_130, run_150])
    x_train = np.load("../data/clean/images/train.npy")
    y_train = np.load("../data/clean/targets/train_targets.npy")

n_layers = 3
filter_architecture = [16]*3 + [32]*3 + [64]*2
kernel_arcitecture = [3, 3, 3, 3, 3, 3, 3, 3,]  
strides_architecture = [1,] * 3 + [1,] + [1,]*4
epochs = 5000

latent_dim = 75
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
        "delta":0.01,
        "pretrain_epochs": 200,
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
        beta=10,
        mode_config=mode_config,
        clustering_config=clustering_config,
        labelled_data=[x_train, y_train],
        )

graph_kwds = {
        "activation":"relu"
        }

cvae.compile_model(graph_kwds=graph_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-4, ]
opt_kwds = {
    "beta1": 0.8,
}

cvae.compute_gradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()

with open("run.py", "w") as fo:
    fo.write("run={}".format(run.run + 1))

lx, lz = cvae.train(
        sess,
        epochs,
        "../drawing",
        "../models",
        batch_size,
        earlystopping=True,
        run=run.run
        )


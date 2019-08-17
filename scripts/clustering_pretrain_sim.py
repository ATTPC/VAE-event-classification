import tensorflow as tf
import numpy as np
import run
import os

from sklearn.preprocessing import OneHotEncoder

import sys
import h5py
import scipy

sys.path.append("../src")

from convolutional_VAE import ConVae
from data_loader import *

print("PID:", os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_vgg = False
use_dd = True
data = "simulated"
if "vgg" in data:
    use_vgg = True
if data == "simulated":
    x_train, x_test, y_test = load_simulated("128")
    if use_dd:
        dd_train, dd_test = load_simulated_hist("128")
elif data == "cleanevent":
    X_sim = np.load("../data/simulated/images/pr_train_simulated.npy")
    y_sim = np.load("../data/simulated/targets/train_targets.npy")
    x_train, x_test, y_test = load_clean_event("128")
    where_junk = y_test == 2
    n_junk_to_steal = 200
    which_to_steal = np.random.randint(0, where_junk.shape[0], size=n_junk_to_steal)
    x_train = np.concatenate([x_train, X_sim], axis=0)
    X_sim = np.concatenate([X_sim, x_test[which_to_steal]], axis=0)
    y_sim = np.concatenate([y_sim, y_test[which_to_steal].argmax(1)], axis=0)
    oh = OneHotEncoder(sparse=False)
    y_sim = oh.fit_transform(y_sim.reshape(-1, 1))
elif data == "vgg_simulated":
    x_train, target_x_train, x_test, y_test = load_vgg_simulated("128")

n_layers = 4
filter_architecture = [32] * 2 + [64] * 2 + [128] * 1
kernel_arcitecture = [5, 5, 3, 3]
strides_architecture = [1, 2, 1, 2]
pooling_architecture = [0, 0, 0, 0]
epochs = 5000

latent_dim = 3
batch_size = 200

mode_config = {
    "simulated_mode": False,
    "restore_mode": False,
    "use_vgg": use_vgg,
    "use_dd": use_dd,
    "include_KL": False,
    "include_MMD": False,
    "include_KM": True,
}

clustering_config = {
    "n_clusters": 2,
    "alpha": 1,
    "delta": 0.01,
    "update_interval": 140,
    "pretrain_epochs": 2,
    "pretrain_simulated": False,
    "self_supervise": False,
}

cvae = ConVae(
    n_layers,
    filter_architecture,
    kernel_arcitecture,
    strides_architecture,
    pooling_architecture,
    latent_dim,
    x_train,
    beta=0.8,
    mode_config=mode_config,
    clustering_config=clustering_config,
    labelled_data=[x_test, y_test],
)
if use_vgg:
    cvae.target_images = target_x_train
if use_dd:
    cvae.dd_targets = dd_train
    cvae.lmbd = 10000
    cvae.dd_dense = 2

graph_kwds = {"activation": "lrelu"}

cvae.compile_model(graph_kwds=graph_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-4]
opt_kwds = {"beta1": 0.4}

cvae.compute_gradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()

with open("run.py", "w") as fo:
    fo.write("run={}".format(run.run + 1))

lx, lz = cvae.train(
    sess, epochs, batch_size, "../drawing", "../models", earlystopping=True, run=run.run
)

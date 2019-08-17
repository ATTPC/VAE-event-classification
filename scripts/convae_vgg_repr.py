import sys

sys.path.append("../src")
from convolutional_VAE import ConVae
from data_loader import load_simulated, load_clean, load_real
from data_loader import load_clean_vgg, load_real_vgg
from latent_classifier import test_model

import h5py

import numpy as np
import keras as ker
import os
import tensorflow as tf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


print("PID: ", os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
data = "real"
size = "128"

if data == "simulated":
    x_train, x_test, y_test = load_simulated(128)
elif data == "clean":
    x_train, x_test, y_test = load_clean(128)
elif data == "real":
    x_train, x_test, y_test = load_real(128)
elif data == "vgg_clean":
    vgg_x_train, x_train, x_test, y_test = load_clean_vgg(size)
elif data == "vgg_real":
    vgg_x_train, x_train, x_test, y_test = load_real_vgg(size)
    x_train = x_train.reshape((x_train.shape[0], -1))
# n_samp = 10000
# x_train = x_train[0:n_samp]
# vgg_x_train = vgg_x_train[0:n_samp]

n_layers = 2
filter_architecture = [32, 32, 32, 32, 32]
kernel_architecture = [5, 3, 5, 3, 3]
strides_architecture = [2, 2, 2, 2, 2]
pool_architecture = [0, 0, 0, 0, 0]

mode_config = {
    "simulated_mode": False,
    "restore_mode": False,
    "include_KL": False,
    "include_MMD": False,
    "include_KM": False,
    "batchnorm": True,
    "use_vgg": False,
}

experiments = 1
lxs = []
lzs = []
train_perf = []
test_perf = []

for i in range(experiments):
    epochs = 2000
    latent_dim = 1000
    batch_size = 150
    print("experiment: ", i)

    cvae = ConVae(
        n_layers,
        filter_architecture,
        kernel_architecture,
        strides_architecture,
        pool_architecture,
        latent_dim,
        x_train,
        beta=0,
        mode_config=mode_config,
        labelled_data=[x_test, y_test],
    )
    # cvae.target_imgs = x_train

    graph_kwds = {"activation": "lrelu", "output_activation": None}
    loss_kwds = {"reconst_loss": "mse"}
    cvae.compile_model(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-2]
    opt_kwds = {"beta1": 0.65}
    cvae.compute_gradients(opt)
    sess = tf.InteractiveSession()
    lx, lz = cvae.train(
        sess, epochs, batch_size, earlystopping=True, save_checkpoints=1, verbose=1
    )
    p = test_model(x_test, y_test, cvae, sess)
    sess.close()

    lxs.append(lx)
    lzs.append(lz)

    train_perf.append(p[0])
    test_perf.append(p[1])
    print()
    print("ITER NUMBER ", i)
    [print(i) for i in p[0]]
    print("-----------")
    [print(i) for i in p[1]]

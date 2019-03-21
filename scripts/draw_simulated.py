import sys
sys.path.append("../src")

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from draw import DRAW
from counter import *


T = 15
enc_size = 400
dec_size = 400
latent_dim = 10
epochs = 150

batch_size = 50
train_data = np.load("../data/simulated/pr_train_simulated.npy")
test_data = np.load("../data/simulated/pr_test_simulated.npy")

delta = 0.8
N = 55

delta_write = delta
delta_read = delta

read_N = N
write_N = N

array_delta_w = np.zeros((batch_size, 1))
array_delta_w.fill(delta_write)
array_delta_w = array_delta_w.astype(np.float32)

array_delta_r = np.zeros((batch_size, 1))
array_delta_r.fill(delta_read)
array_delta_r = array_delta_r.astype(np.float32)

attn_config = {
            "read_N": read_N,
            "write_N": write_N,
            "write_N_sq": write_N**2,
            "delta_w": array_delta_w,
            "delta_r": array_delta_r,
        }


draw_model = DRAW(
        T,
        dec_size,
        enc_size,
        latent_dim,
        batch_size,
        train_data,
        attn_config=attn_config
        )

graph_kwds = {
        "initializer": tf.initializers.glorot_normal
        }

loss_kwds = {
        "reconst_loss": None
        }

draw_model.CompileModel(graph_kwds, loss_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-2,]
opt_kwds = {
        "beta1": 0.5,
        }

draw_model.computeGradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()

data_dir = "../drawing"
model_dir = "../models"

lx, lz, = draw_model.train(sess, epochs, data_dir, model_dir, )

draw_model.generateLatent(sess, "../drawing", (train_data, test_data))

draw_model.generateSamples("../drawing", "../drawing")

sess.close()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 20), sharex=True)
plt.suptitle("Loss function components")

axs[0].plot(range(epochs), lx, label=r"$\mathcal{L}_x$")
axs[1].plot(range(epochs), lz, label=r"$\mathcal{L}_z$")

[a.legend() for a in axs]
[a.set_ylim((1000, 200)) for a in axs]

fig.savefig(
    "../plots/simulated_loss_functions.png")

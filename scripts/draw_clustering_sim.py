import sys
import os
import run
sys.path.append("../src")

import numpy as np
import tensorflow as tf
from keras.datasets import mnist

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from draw import DRAW
from counter import *

print("PID:", os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

T = 10
enc_size = 200
dec_size = 200
latent_dim = 5
epochs = 2000

batch_size = 100
"""
train_data = np.load("../data/simulated/pr_train_simulated.npy")
test_data = np.load("../data/simulated/pr_test_simulated.npy")
test_targets = np.load("../data/simulated/test_targets.npy")
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_data = (np.expand_dims(x_train, -1)/255)[0:5000]
test_data = np.expand_dims(x_test, -1)/255
test_targets = y_test

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

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": False,
        "include_KM": True
        }

clustering_config = {
        "n_clusters":10,
        "alpha":1,
        "delta":0.01,
        "pretrain_epochs": 200,
        "update_interval": 140,
        "pretrain_simulated":False,
        }

draw_model = DRAW(
        T,
        dec_size,
        enc_size,
        latent_dim,
        train_data,
        beta=10,
        use_attention=True,
        attn_config=attn_config,
        mode_config=mode_config,
        clustering_config=clustering_config,
        labelled_data=[test_data, test_targets] 
        )

graph_kwds = {
        "initializer": tf.initializers.glorot_normal
        }

loss_kwds = {
        "reconst_loss": None,
        }

draw_model.compile_model(graph_kwds, loss_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-3,]
opt_kwds = {
        "beta1": 0.6,
        }

draw_model.compute_gradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()

data_dir = "../drawing"
model_dir = "../models"

with open("run.py", "w") as fo:
    fo.write("run={}".format(run.run + 1))

lx, lz, = draw_model.train(
        sess,
        epochs,
        data_dir,
        model_dir,
        batch_size,
        earlystopping=True,
        run=run.run
        )



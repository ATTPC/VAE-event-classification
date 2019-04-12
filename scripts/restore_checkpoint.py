import sys
sys.path.append("../src")

import tensorflow as tf
import numpy as np
from draw import DRAW

restore_fn = "../models/draw_attn_epoch89.ckpt"

T = 7
enc_size = 900
dec_size = 900
latent_dim = 3
epochs = 100


treshold_value = 0.4
treshold_data = True

batch_size = 50

train_data = np.load("../data/processed/train.npy")
test_data = np.load("../data/processed/test.npy")

train_test = np.concatenate((train_data, test_data))

if treshold_data:
	train_test[train_test < treshold_value] = 0
delta = 1.04
N = 65

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
        "restore_mode": True
        }

with tf.device("/gpu:2"):
    draw_model = DRAW(
        T,
        dec_size,
        enc_size,
        latent_dim,
        batch_size,
        train_test,
        attn_config=attn_config,
        mode_config=mode_config,
    )

    graph_kwds = {
        "initializer": tf.initializers.glorot_normal
    }

    loss_kwds = {
        "reconst_loss": None,
        "include_KL": False
    }

    draw_model.CompileModel(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-2, ]
    opt_kwds = {
        "beta1": 0.5,
    }

    draw_model.computeGradients(opt, opt_args, opt_kwds)

    sess = tf.InteractiveSession()

    data_dir = "../drawing"
    model_dir = "../models"

    lx, lz, = draw_model.train(
    sess, epochs, data_dir, model_dir, earlystopping=False, checkpoint_fn=restore_fn)

    draw_model.generateLatent(sess, "../drawing", (train_data, test_data))
    draw_model.generateSamples("../drawing", "../drawing")

    sess.close()


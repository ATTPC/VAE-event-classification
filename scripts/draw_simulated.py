import sys
sys.path.append("../src")

import numpy as np
import tensorflow as tf

from draw import DRAW


T = 15
enc_size = 100
dec_size = 100
latent_dim = 10

batch_size = 50
train_data = np.load("../data/simulated/train_data.npy")
test_data = np.load("../data/simulated/test_data.npy")
all_data = np.concatenate((train_data, test_data), axis=0)

delta_write = 10
delta_read = 10 

read_N = 10
write_N = 10

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
        all_data,
        attn_config
        )

graph_kwds = {
        "initializer": tf.initializers.glorot_normal
        }

loss_kwds = {
        "reconst_loss": None
        }

draw_model.CompileModel(graph_kwds, loss_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-1,]
opt_kwds = {
        "beta1": 0.5,
        }

draw_model.computeGradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()

epochs = 5
data_dir = "../data"
model_dir = "../models"

draw_model.train(sess, epochs, data_dir, model_dir, )

draw_model.generateLatent(sess, "../drawing", (train_data, test_data))

draw_model.generateSamples("../drawing", "../drawing")

sess.close()

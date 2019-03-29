import sys
sys.path.append("../src")

import numpy as np
import tensorflow as tf

from draw import DRAW
from counter import *


T = 15
enc_size = 800
dec_size = 500
latent_dim = 50

batch_size = 50
train_data = np.load("../data/simulated/pr_train_simulated.npy")
test_data = np.load("../data/simulated/pr_test_simulated.npy")

#all_data = np.concatenate((train_data, test_data), axis=0)

delta_range = np.linspace(0.8, 1.1, 10)
N_range = np.arange(10, 50, 5)
repeats = 2
epochs = 40

loss_record_shape = (
        len(delta_range),
        len(N_range),
        repeats,
        2,
        epochs,
        )

hyperparam_vals = np.array((
        delta_range,
        N_range,
        repeats,
        [0, 1],
        epochs
        ))

np.save("../loss_records/hyperparam_vals_{}.npy".format(RUN_COUNT), hyperparam_vals)

loss_record = np.zeros(loss_record_shape)

for i, delta in enumerate(delta_range):

    delta_write = delta
    delta_read = delta

    for j, N in enumerate(N_range):

        for k in range(repeats):

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
            loss_record[i, j, k, 0] = lx
            loss_record[i, j, k, 1] = lz

            #draw_model.generateLatent(sess, "../drawing", (train_data, test_data))

            #draw_model.generateSamples("../drawing", "../drawing")

            sess.close()

np.save("../loss_records/delta_opt_{}.npy".format(RUN_COUNT) ,loss_record)

with open("counter.py", "w") as fo:
    fo.write("RUN_COUNT = {}".format(RUN_COUNT + 1))
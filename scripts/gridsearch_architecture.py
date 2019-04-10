
import sys
sys.path.append("../src")

import numpy as np
import tensorflow as tf

from draw import DRAW
from counter import *

batch_size = 50
T = 6
latent_dim = 3
delta = 1.04
N = 65
batch_size = 50

repeats = 3
epochs = 30

read_N = N
write_N = N

delta_write = delta
delta_read = delta

array_delta_w = np.zeros((batch_size, 1))
array_delta_w.fill(delta_write)
array_delta_w = array_delta_w.astype(np.float32)

array_delta_r = np.zeros((batch_size, 1))
array_delta_r.fill(delta_read)
array_delta_r = array_delta_r.astype(np.float32)

treshold_value = 0.4
treshold_data = True

all_0130 = np.load("../data/processed/all_0130.npy")
#all_0210 = np.load("../data/processed/all_0210.npy")
#all_data = np.concatenate([all_0130, all_0210])

all_data = all_0130

if treshold_data:
	all_data[all_data < treshold_value] = 0

state_sizes = np.arange(500, 1000, 100)
state_sizes = state_sizes[::-1]

time_array = np.arange(3, 9)
time_array = time_array[::-1]

loss_record_shape = (
        len(state_sizes),
        len(time_array),
        repeats, 
        2,
        epochs
        )

loss_record = np.zeros(loss_record_shape)

hyperparam_vals = np.array((
        state_sizes,
        time_array,
        repeats,
        [0, 1],
        epochs,
        ))


for i, state_size in enumerate(state_sizes):

    enc_size = state_size
    dec_size = state_size

    for j, T in enumerate(time_array):

        for k in range(repeats):

            attn_config = {
                        "read_N": read_N,
                        "write_N": write_N,
                        "write_N_sq": write_N**2,
                        "delta_w": array_delta_w,
                        "delta_r": array_delta_r,
                    }

            mode_config = {
                    "simulated_mode": False
                    }


            draw_model = DRAW(
                    T,
                    dec_size,
                    enc_size,
                    latent_dim,
                    batch_size,
                    all_data,
                    attn_config=attn_config,
                    mode_config=mode_config
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

            lx, lz, = draw_model.train(sess, epochs, data_dir, model_dir, save_checkpoints=False)
            loss_record[i, j, k, 0] = lx
            loss_record[i, j, k, 1] = lz

            #draw_model.generateLatent(sess, "../drawing", (train_data, test_data))

            #draw_model.generateSamples("../drawing", "../drawing")

            sess.close()

            print()
            print("----------------- ITER ------------------")
            print(state_size, T) 
            print("-----------------------------------------")
            print()

np.save("../loss_records/delta_opt_{}.npy".format(RUN_COUNT) ,loss_record)
np.save("../loss_records/hyperparam_vals_{}.npy".format(RUN_COUNT),hyperparam_vals)

with open("counter.py", "w") as fo:
    fo.write("RUN_COUNT = {}".format(RUN_COUNT + 1))

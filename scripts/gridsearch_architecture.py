
import sys
sys.path.append("../src")

import numpy as np
import tensorflow as tf

from sklearn.model_selection import  cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import h5py

import os

from draw import DRAW
from counter import *

print("PID :", os.getpid())

def longform_latent(latent,):
    longform_samples = np.zeros((
                        latent.shape[1], 
                        latent.shape[0]*latent.shape[2]
                        ))

    latent_dim = latent.shape[2]

    for i, evts in enumerate(latent):
        longform_samples[:, i*latent_dim:(i+1)*latent_dim] = evts

    return longform_samples


def compute_accuracy(X, y):

    model = LogisticRegression(
            solver="liblinear",
            multi_class="ovr",
            class_weight="balanced"
        )

    mlp = MLPClassifier(
            max_iter=1000,
            batch_size=10,
            hidden_layer_sizes=(80, 30, 10),
            early_stopping=True,
            learning_rate_init=0.01
        )

    mlp.fit(X, y)
    model.fit(X, y)

    logreg_score = np.average(cross_val_score(model, X, y, cv=4))
    mlp_score = np.average(cross_val_score(mlp, X, y, cv=4))
    return logreg_score, mlp_score

T = 5
delta = 0.8
N = 85
batch_size = 86

repeats = 2
epochs = 20

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
treshold_data = False

all_0130 = np.load("../data/processed/all_0130.npy")
train_data = np.load("../data/processed/train.npy")

with h5py.File("../data/images.h5", "r") as fo:
    train_targets = np.array(fo["train_targets"])[:-1]

#all_0210 = np.load("../data/processed/all_0210.npy")
#all_data = np.concatenate([all_0130, all_0210])

all_data = all_0130

if treshold_data:
	all_data[all_data < treshold_value] = 0

state_sizes = np.arange(1000, 2000, 200)
state_sizes = state_sizes[::-1]

latent_sizes = np.arange(10, 20, 2)
latent_sizes = latent_sizes[::-1]

include_kl = [True, False]

loss_record_shape = (
        len(state_sizes),
        len(latent_sizes),
        len(include_kl),
        repeats, 
        2,
        epochs
        )

accuracy_record_shape = (
        len(state_sizes),
        len(latent_sizes),
        len(include_kl),
        repeats,
        2
        )

loss_record = np.zeros(loss_record_shape)
classification_recod = np.zeros(accuracy_record_shape)

hyperparam_vals = np.array((
        state_sizes,
        latent_sizes,
        include_kl,
        repeats,
        [0, 1],
        epochs,
        ))

for i, state_size in enumerate(state_sizes):

    enc_size = state_size
    dec_size = state_size

    for j, latent_dim in enumerate(latent_sizes):

        for k, use_kl in enumerate(include_kl):

            for l in range(repeats):

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
                        "reconst_loss": None,
                        "include_KL": use_kl
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

                if any(np.isnan(lx)) or any(np.isnan(lz)):
                        continue

                print()
                loss_record[i, j, k, l, 0] = lx
                loss_record[i, j, k, l, 1] = lz

                draw_model.X = train_data

                latent_values , _, _ = draw_model.generateLatent(sess, "../drawing", (train_data, ), save=False)
                latent_values = longform_latent(latent_values[0])

                accuracies = compute_accuracy(latent_values, train_targets)
                classification_recod[i, j, k, l] = accuracies

                print("--------------------")
                print("Cell size: ", enc_size, " Latent dim : ", latent_dim, " KL : ", use_kl)
                print("logreg_acc", accuracies[0], "mlp_acc", accuracies[1])
                print("---------------------")
                sess.close()

            np.save("../loss_records/delta_opt_{}.npy".format(RUN_COUNT) ,loss_record)
            np.save("../loss_records/hyperparam_vals_{}.npy".format(RUN_COUNT),hyperparam_vals)
            np.save("../loss_records/accuracy_vals{}.npy".format(RUN_COUNT), classification_recod)

with open("counter.py", "w") as fo:
    fo.write("RUN_COUNT = {}".format(RUN_COUNT + 1))

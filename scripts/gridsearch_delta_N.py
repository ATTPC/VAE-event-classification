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


enc_size = 800
dec_size = 800
latent_dim = 3

batch_size = 86

all_data = np.load("../data/processed/all_0130.npy")
train_data = np.load("../data/processed/train.npy")

with h5py.File("../data/images.h5", "r") as fo:
    train_targets = np.array(fo["train_targets"])[:-1]

#all_data = np.concatenate((train_data, test_data), axis=0)

treshold_value = 0.4
treshold_data = False

if treshold_data:
	all_data[all_data < treshold_value] = 0

delta_range = np.linspace(0.8, 1.1, 6) 

t_range = np.arange(3, 8)
#t_range = t_range[::-1]
N_range = np.arange(55, 95, 10)
N_range = N_range[::-1]

repeats = 2
epochs = 20

loss_record_shape = (
        len(delta_range),
        len(N_range),
        len(t_range),
        repeats,
        2,
        epochs,
        )

accuracy_record_shape = (
        len(delta_range),
        len(N_range),
        len(t_range),
        repeats,
        2
        )

hyperparam_vals = np.array((
        delta_range,
        N_range,
        t_range,
        repeats,
        [0, 1],
        epochs
        ))

loss_record = np.zeros(loss_record_shape)
classification_recod = np.zeros(accuracy_record_shape)

for i, delta in enumerate(delta_range):

    delta_write = delta
    delta_read = delta

    for j, N in enumerate(N_range):

        for k, T in enumerate(t_range):

            for l in range(repeats):

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

                mode_config = {"simulated_mode": False}

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

                lx, lz, = draw_model.train(
                        sess,
                        epochs,
                        data_dir, 
                        model_dir, 
                        earlystopping=False,
                        save_checkpoints=False,
                        )
                
                if any(np.isnan(lx)) or any(np.isnan(lz)):
                        sess.close()
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
                print("delta: ", delta, " N : ", N, " T : ", T)
                print("logreg_acc", accuracies[0], "mlp_acc", accuracies[1])
                print("---------------------")

                #draw_model.generateSamples("../drawing", "../drawing")

                sess.close()

np.save("../loss_records/delta_opt_{}.npy".format(RUN_COUNT) ,loss_record)
np.save("../loss_records/hyperparam_vals_{}.npy".format(RUN_COUNT), hyperparam_vals)
np.save("../loss_records/accuracy_vals{}.npy".format(RUN_COUNT), classification_recod)


with open("counter.py", "w") as fo:
    fo.write("RUN_COUNT = {}".format(RUN_COUNT + 1))

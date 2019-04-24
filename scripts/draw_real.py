
from counter import *
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
import os

from sklearn.model_selection import  cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from keras.utils import to_categorical

import sys
sys.path.append("../src")

from draw import DRAW

matplotlib.use("Agg")

print("PID: ", os.getpid())

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

   #model = LogisticRegression(
   #         solver="liblinear",
   #         multi_class="ovr",
   #         class_weight="balanced"
   #     )

    mlp = MLPClassifier(
            max_iter=1000,
            batch_size=10,
            hidden_layer_sizes=(80, 30, 10),
            early_stopping=True,
            learning_rate_init=0.01
        )

    mlp.fit(X, y)
    #model.fit(X, y)

    #logreg_score = np.average(cross_val_score(model, X, y, cv=4))
    mlp_score = np.average(cross_val_score(mlp, X, y, cv=4))

    return mlp_score

T = 7
enc_size = 1200
dec_size = 600
latent_dim = 10
epochs = 100

treshold_value = 0.4
treshold_data = False

batch_size = 86

with h5py.File("../data/images.h5", "r") as fo:
    train_targets = np.array(fo["train_targets"])

all_0130 = np.load("../data/processed/all_0130.npy")
#all_0210 = np.load("../data/processed/all_0210.npy")
#all_data = np.concatenate([all_0130, all_0210])

all_data = all_0130

if treshold_data:
	all_data[all_data < treshold_value] = 0

train_data = np.load("../data/processed/train.npy")
test_data = np.load("../data/processed/test.npy")

train_test = np.concatenate((train_data, test_data))

if treshold_data:
	train_test[train_test < treshold_value] = 0
	train_data[train_data < treshold_value] = 0 

delta = 0.98
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

mode_config = {"simulated_mode": False}

with tf.device("/gpu:2"):
    train_targets = to_categorical(train_targets)

    draw_model = DRAW(
        T,
        dec_size,
        enc_size,
        latent_dim,
        batch_size,
        all_data,
        X_classifier=train_data,
        Y_classifier=train_targets,
        attn_config=attn_config,
        mode_config=mode_config,
        train_classifier=True
    )

    graph_kwds = {
        "initializer": tf.initializers.glorot_normal
    }

    loss_kwds = {
        "reconst_loss": None,
        "include_KL": True
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
        sess, epochs, data_dir, model_dir, earlystopping=False)

    draw_model.X = train_data
    draw_model.generateLatent(sess, "../drawing", (train_data, test_data))

    latent_values , _, _ = draw_model.generateLatent(sess, "../drawing", (train_data, ), save=False)
    latent_values = longform_latent(latent_values[0])

    #accuracies = compute_accuracy(latent_values, train_targets[:-1])

    #print()

    #print("--------------------")
    #print(accuracies)
    #print("---------------------")

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

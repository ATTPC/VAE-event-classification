
from counter import *
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
import os

from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from keras.utils import to_categorical

import sys
sys.path.append("../src")

from draw import DRAW
from data_loader import load_clean

matplotlib.use("Agg")

print("PID: ", os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def longform_latent(latent,):
    longform_samples = np.zeros((
                        latent.shape[1], 
                        latent.shape[0]*latent.shape[2]
                        ))

    latent_dim = latent.shape[2]

    for i, evts in enumerate(latent):
        longform_samples[:, i*latent_dim:(i+1)*latent_dim] = evts

    return longform_samples


def fit_logreg(X, y,):

    model = LogisticRegression(
            penalty="l2",
            solver="newton-cg",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=10000,
        )

    model.fit(X, y)

    #train_score = f1_score(y, model.predict(X), average=None)
    #test_score = f1_score(ytest, model.predict(Xtest), average=None)
    return model 

x_train, x_test, y_test = load_clean(80)
x_train = x_train[:10000]

T = 3
enc_size = 300
dec_size = 150
latent_dim  = 8
epochs = 300
batch_size = 100

delta = 0.98
N = 65

delta_write = delta
delta_read = delta

read_N = N
write_N = N

attn_config = {
    "read_N": read_N,
    "write_N": write_N,
    "write_N_sq": write_N**2,
    "delta_w": delta,
    "delta_r": delta,
}

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_KM": False,
        "include_MMD":True,
        }

conv_architecture = {
    "n_layers": 4,
    "filters": [8, 16, 32, 64],
    "kernel_size": [5, 5, 3, 3],
    "strides": [2, 2, 2, 2],
    "pool": [0, 0, 0, 0],
    "activation": [1, 1, 1, 1],
    "activation_func": "relu"
    }

model_train_targets = to_categorical(y_test)

draw_model = DRAW(
    T,
    dec_size,
    enc_size,
    latent_dim,
    x_train,
    beta=1e-4,
    train_classifier=False,
    use_conv=True,
    #use_attention=True,
    #X_classifier=x_test,
    #Y_classifier=model_train_targets,
    mode_config=mode_config,
    attn_config=attn_config,
    conv_architecture=conv_architecture
)

graph_kwds = {
    "initializer": tf.initializers.glorot_normal,
    "n_encoder_cells": 1,
    "n_decoder_cells": 1 
}

loss_kwds = {
    "reconst_loss": "mse",
    "scale_kl": False,
}

draw_model.compile_model(graph_kwds, loss_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-3, ]
opt_kwds = {
    "beta1": 0.8,
}

draw_model.compute_gradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()

data_dir = "../drawing"
model_dir = "../models"

lx, lz, = draw_model.train(
                        sess,
                        epochs,
                        data_dir,
                        model_dir,
                        batch_size,
                        earlystopping=True,
                        )

latent_test = draw_model.run_large(
                                sess,
                                draw_model.z_seq,
                                x_test,
                                )
latent_train = draw_model.run_large(
                                sess,
                                draw_model.z_seq,
                                x_train,
                                )
np.save("../drawing/latent/test_latent.npy", latent_test)
np.save("../drawing/latent/train_latent.npy", latent_train)

latent_test = np.array(latent_test)
latent_test = longform_latent(latent_test)

lr_train, lr_test, lry_train, lry_test = train_test_split(latent_test, y_test)
lr_model = fit_logreg(
        lr_train, lry_train)

pred_train = lr_model.predict(lr_train)
pred_test = lr_model.predict(lr_test)

train_score = f1_score(lry_train, pred_train, average=None)
test_score = f1_score(lry_test, pred_test, average=None)
print()

print("--------------------")
print("train: ", train_score)
print("test : ", test_score)
print("---------------------")

#draw_model.generate_samples("../drawing", )

sess.close()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 20), sharex=True)
plt.suptitle("Loss function components")

axs[0].plot(range(epochs), lx, label=r"$\mathcal{L}_x$")
axs[1].plot(range(epochs), lz, label=r"$\mathcal{L}_z$")

[a.legend() for a in axs]
[a.set_ylim((1000, 200)) for a in axs]

fig.savefig(
    "../plots/simulated_loss_functions.png")

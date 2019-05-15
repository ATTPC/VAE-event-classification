
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


def compute_accuracy(X, y, Xtest, ytest):

    model = LogisticRegression(
            penalty="l2",
            solver="newton-cg",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=10000,
        )

    model.fit(X, y)

    train_score = f1_score(y, model.predict(X), average=None)
    test_score = f1_score(ytest, model.predict(Xtest), average=None)

    return train_score, test_score 

T = 8
enc_size = 200
dec_size = 900
latent_dim  = 3
epochs = 50

treshold_value = 0.3
treshold_data = False

batch_size = 50

all_0130 = np.load("../data/clean/images/run_0130_label_False.npy")
all_0170 = np.load("../data/clean/images/run_0170_label_False.npy")
#all_0210 = np.load("../data/processed/all_0210.npy")
#all_data = np.concatenate([all_0130, all_0210])

all_data = np.concatenate([all_0130, all_0170])
del(all_0130)
del(all_0170)

if treshold_data:
	all_data[all_data < treshold_value] = 0

targets_0130 = np.load("../data/clean/targets/run_0130_targets.npy")
targets_0210 = np.load("../data/clean/targets/run_0210_targets.npy")
all_targets = np.concatenate([targets_0130, targets_0210])

labeled_0130 = np.load("../data/clean/images/run_0130_label_True.npy")
labeled_0210 = np.load("../data/clean/images/run_0210_label_True.npy")
all_labeled = np.concatenate([labeled_0130, labeled_0210])

train_data, test_data, train_targets, test_targets = train_test_split(
                                                        all_labeled,
                                                        all_targets,
                                                        test_size=0.33
                                                        )

if treshold_data:
	train_test[train_test < treshold_value] = 0
	train_data[train_data < treshold_value] = 0 

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
        "include_MMD":False,
        }

model_train_targets = to_categorical(train_targets)

draw_model = DRAW(
    T,
    dec_size,
    enc_size,
    latent_dim,
    all_data,
    beta=100,
    train_classifier=False,
    use_conv=False,
    use_attention=True,
    #X_classifier=train_data,
    #Y_classifier=model_train_targets,
    mode_config=mode_config,
    attn_config=attn_config,
)

graph_kwds = {
    "initializer": tf.initializers.glorot_normal,
    "n_encoder_cells": 1,
    "n_decoder_cells": 1 
}

loss_kwds = {
    "reconst_loss": None,
    "scale_kl": False,
}

draw_model.CompileModel(graph_kwds, loss_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-3, ]
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
                        batch_size,
                        earlystopping=False)

draw_model.X = train_data
draw_model.generate_latent(sess, "../drawing", (train_data, test_data))

latent_values , _, _ = draw_model.generate_latent(
                                    sess,
                                    "../drawing",
                                    (train_data, test_data ),
                                    save=False
                                    )

latent_train = longform_latent(latent_values[0])
latent_test = longform_latent(latent_values[1])

train_score, test_score = compute_accuracy(
        latent_train, train_targets,
        latent_test, test_targets)

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

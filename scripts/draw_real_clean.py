
from counter import *
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
import os

latent_dim = 9
epochs = 300
batch_size = 50

delta = 0.9
#N = 30

delta_write = delta
delta_read = delta

read_N = 8
write_N = 8

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
        "include_MMD": True,
        }

conv_architecture = {
    "n_layers": 4,
    "filters": [8, 8, 16, 16],
    "kernel_size": [5, 5, 3, 3],
    "strides": [1, 1, 1, 1],
    "pool": [1, 0, 1, 0],
    "activation": [1, 0, 1, 0],
    "activation_func": "relu"
    }

model_train_targets = to_categorical(y_test)
draw_model = DRAW(
    T,
    dec_size,
    enc_size,
    latent_dim,
    x_train,
    beta=11,
    train_classifier=False,
    use_conv=True,
    #use_attention=True,
    #X_classifier=x_test,
    #Y_classifier=model_train_targets,
    mode_config=mode_config,
    #attn_config=attn_config,
    conv_architecture=conv_architecture
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

draw_model.compile_model(graph_kwds, loss_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-4, ]
opt_kwds = {
    "beta1": 0.5,
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
lr_model = fit_logreg(lr_train, lry_train)

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

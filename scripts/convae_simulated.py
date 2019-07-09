import sys
sys.path.append("../src")
from convolutional_VAE import ConVae
from data_loader import load_simulated
from latent_classifier import test_model

import h5py

import numpy as np
import keras as ker
import os
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


print("PID: ", os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
x_train, x_test, y_test = load_simulated(1)
x_train = x_train[0:200]

n_layers = 4
filter_architecture = [8, 16, 32, 64]
kernel_architecture = [5, 5, 3, 3]
strides_architecture = [2, 2, 2, 2,]
pool_architecture = [0, 0, 0, 00]

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": True,
        "include_KM": False,
        "batchnorm": True 
        }

experiments = 3
lxs = []
lzs = []
train_perf = []
test_perf = []

for i in range(experiments): 
    epochs = 3
    latent_dim = 3
    batch_size = 150

    cvae = ConVae(
            n_layers,
            filter_architecture,
            kernel_architecture,
            strides_architecture,
            pool_architecture,
            latent_dim,
            x_train,
            beta=1e-1,
            mode_config=mode_config,
            labelled_data=[x_test, y_test]
            )

    graph_kwds = {"activation":"relu", "output_activation":None}
    loss_kwds = {"reconst_loss": "mse"}
    cvae.compile_model(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-4, ]
    opt_kwds = {
        "beta1": 0.7,
    }

    cvae.compute_gradients(opt,)
    
    sess = tf.InteractiveSession() 
    lx, lz = cvae.train(
            sess,
            epochs,
            batch_size,
            earlystopping=True,
            )
    p = test_model(x_test, y_test, cvae, sess)
    sess.close()

    lxs.append(lx)
    lzs.append(lz)

    train_perf.append(p[0])
    test_perf.append(p[1])
    print()

sess.close()

"""
MAKING PERFORMANCE TABLE
"""
performance = np.array([train_perf, test_perf])
names = ["Train", "Test"]
score_names = ["f1", "recall", "precision"]
classes = ["proton", "carbon"]

scores = np.zeros((
    len(names),
    len(score_names),
    2,
    len(classes),
    ))
for i in range(len(names)):
    for j in range(len(scores)):
        metric = performance[i, :, j, :]
        print("shape of indexed elems",metric.shape)
        avg = metric.mean(axis=0)
        std = metric.std(axis=0)
        scores[i, j] = (avg, std)


with open("../metrics_clf/simulated/f1_scores.tex", "w") as fo:
    fo.write(r" & Proton & Carbon \\ \n")
    fo.write(r" \hline \n")

    for i in range(2):
        fo.write(" "+ names[i]+" & ")
        if i == 0:
            fo.write(" & ".join(score_names))
            fo.write(" & ")
            fo.write(" & ".join(score_names))
            fo.write("\n")
        for j in range(len(scores)):
            for k in range(len(classes)):
                avg = scores[i, j, 0, k]
                std = scores[i, j, 1, k]
                to_write = " ${:.2f} \pm {:.2f}$ ".format(avg, std)
                fo.write(to_write)
        fo.write(r" \\ \n ")

"""
PLOTTING LOSSES
"""
viridis = matplotlib.cm.get_cmap('viridis')
inferno = matplotlib.cm.get_cmap('inferno')
losses = [lxs, lzs]
colors = [viridis, inferno]
plot_names = ["reconst_loss", "latent_loss"]

for i in range(2):
    fig, ax = plt.subplots()
    loss = losses[i]
    end_vals = [l[-1] for l in loss]
    min_val = min(end_vals)
    min_diffs = []

    for l in loss: 
        if np.nan in l:
            continue
        min_diff = (min_val - np.min(l))**2
        min_diffs.append(min_diff)

    min_diffs = np.array(min_diffs)
    print(min_diffs)
    a = min(min_diffs)
    b = max(min_diffs)
    alpha = 1/b * (1 - a/b)**(-1)
    beta = -a/b*(1-a)**(-1)
    def f(x): return alpha*x + beta
    min_diffs = f(min_diffs)
    print(min_diffs)


    for l in loss: 
        if np.nan in l:
            continue
        c = colors[i](min_diffs[i])
        ax.plot(
                np.arange(len(l)),
                l,
                linewidth=2,
                alpha=0.8,
                color=c
                )

    plt.savefig("../plots/simulated_clf/"+plot_names[i]+".pdf")
    plt.savefig("../plots/simulated_clf/"+plot_names[i]+".png")
    plt.close(fig=fig)



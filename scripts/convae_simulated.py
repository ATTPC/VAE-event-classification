import sys

sys.path.append("../src")
from convolutional_VAE import ConVae
from data_loader import *
from latent_classifier import test_model

import scipy
import h5py

import numpy as np
import keras as ker
import os
import tensorflow as tf
from run import run

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

print("PID: ", os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
data = "event"
size = "128"

if data == "simulated":
    x_train, x_test, y_test = load_simulated(128)
elif data == "clean":
    x_train, x_test, y_test = load_clean(128)
elif data == "event":
    which = "0210"
    x_train, x_test, y_test = load_real_event(size, which)
elif data == "real":
    x_train, x_test, y_test = load_real(128)
# x_train = x_train[0:200]

n_layers = 4
filter_architecture = [8, 8, 32, 32, 64, 16]
kernel_architecture = [5, 5, 3, 3]
strides_architecture = [2, 2, 2, 2]
pool_architecture = [0] * n_layers

mode_config = {
    "simulated_mode": False,
    "restore_mode": False,
    "include_KL": False,
    "include_MMD": True,
    "include_KM": False,
    "batchnorm": True,
    "use_vgg": False,
    "use_dd": False,
}

if mode_config["use_dd"]:
    q_hist_x_train, q_hist_x_test = load_real_event_hist("128")

clustering_config = {
    "n_clusters": 3,
    "alpha": 1,
    "delta": 0.01,
    "pretrain_simulated": False,
    "pretrain_epochs": 200,
    "update_interval": 3,
}

experiments = 1
lxs = []
lzs = []
train_perf = []
test_perf = []

for i in range(experiments):
    epochs = 2000
    latent_dim = 100
    batch_size = 150
    print("experiment: ", i)
    with open("run.py", "w") as fo:
        fo.write("run = {}".format(run + 1))

    cvae = ConVae(
        n_layers,
        filter_architecture,
        kernel_architecture,
        strides_architecture,
        pool_architecture,
        latent_dim,
        x_train,
        beta=0.01,
        mode_config=mode_config,
        clustering_config=clustering_config,
        labelled_data=[x_test, y_test],
    )
    if mode_config["use_dd"]:
        cvae.dd_targets = q_hist_x_train
        cvae.lmbd = 50

    graph_kwds = {"activation": "lrelu", "output_activation": None}
    loss_kwds = {"reconst_loss": "mse"}
    cvae.compile_model(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-3]
    opt_kwds = {"beta1": 0.7829}

    cvae.compute_gradients(opt, opt_args, opt_kwds)
    sess = tf.InteractiveSession(config=config)
    lx, lz = cvae.train(
        sess,
        epochs,
        batch_size,
        earlystopping=True,
        save_checkpoints=1,
        verbose=1,
        run=run,
    )
    p = test_model(x_test, y_test, cvae, sess)
    sess.close()

    lxs.append(lx)
    lzs.append(lz)

    train_perf.append(p[0])
    test_perf.append(p[1])
    print()
    print("ITER NUMBER ", i)
    print(train_perf)
    print(test_perf)


"""
MAKING PERFORMANCE TABLE
"""
performance = np.array([train_perf, test_perf])
np.save(
    "../metrics_clf/convae_" + data + "/performance_" + str(epochs) + ".npy",
    performance,
)

scores = np.zeros((len(names), len(score_names), 2, len(classes)))
for i in range(len(names)):
    for j in range(len(scores)):
        metric = performance[i, :, j, :]
        print("shape of indexed elems", metric.shape)
        avg = metric.mean(axis=0)
        std = metric.std(axis=0)
        scores[i, j] = (avg, std)


with open("../metrics_clf/simulated/f1_scores.tex", "w") as fo:
    fo.write(r" & Proton & Carbon \\ \n")
    fo.write(r" \hline \n")

    for i in range(2):
        fo.write(" " + names[i] + " & ")
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
viridis = matplotlib.cm.get_cmap("viridis")
inferno = matplotlib.cm.get_cmap("inferno")
losses = [lxs, lzs]
colors = [viridis, inferno]
plot_names = ["reconst_loss", "latent_loss"]
yaxis_names = [r"$L_x$", r"$L_z$"]

for i in range(2):
    fig, ax = plt.subplots()
    loss = losses[i]
    end_vals = [l[-1] for l in loss]
    min_val = min(end_vals)
    min_diffs = []

    for l in loss:
        if np.nan in l:
            continue
        min_diff = (min_val - np.min(l)) ** 2
        min_diffs.append(min_diff)

    min_diffs = np.array(min_diffs)
    print(min_diffs)
    a = min(min_diffs)
    b = max(min_diffs)
    alpha = 1 / b * (1 - a / b) ** (-1)
    beta = -a / b * (1 - a) ** (-1)

    def f(x):
        return alpha * x + beta

    min_diffs = f(min_diffs)
    print(min_diffs)

    for l in loss:
        if np.nan in l:
            continue
        c = colors[i](min_diffs[i])
        ax.plot(np.arange(len(l)), l, linewidth=2, alpha=0.8, color=c)

    plt.savefig("../plots/simulated_clf/" + plot_names[i] + ".pdf")
    plt.savefig("../plots/simulated_clf/" + plot_names[i] + ".png")
    plt.close(fig=fig)

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
import matplotlib.ticker as ticker
import matplotlib.ticker as ticker


print("PID: ", os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
x_train, x_test, y_test = load_simulated(1)
#x_train = x_train[0:100]

n_layers = 3
filter_architecture = [16, 16, 8]
kernel_architecture = [11, 5, 3]
strides_architecture = [2, 2, 2,]
pool_architecture = [0, 0, 0,]

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": True,
        "include_KM": False,
        "batchnorm": True, 
        }

experiments = 10
lxs = []
lzs = []
train_perf = []
test_perf = []

for i in range(experiments): 
    epochs = 2000
    latent_dim = 3 
    batch_size = 150
    print("experiment: ", i)

    cvae = ConVae(
            n_layers,
            filter_architecture,
            kernel_architecture,
            strides_architecture,
            pool_architecture,
            latent_dim,
            x_train,
            beta=10,
            mode_config=mode_config,
            labelled_data=[x_test, y_test]
            )

    graph_kwds = {"activation":"relu", "output_activation":"sigmoid"}
    loss_kwds = {"reconst_loss": None}
    cvae.compile_model(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-1, ]
    opt_kwds = {
        "beta1": 0.5572,
    }

    cvae.compute_gradients(opt,)
    sess = tf.InteractiveSession() 
    lx, lz = cvae.train(
            sess,
            epochs,
            batch_size,
            earlystopping=True,
            save_checkpoints=0,
            verbose=0
            )
    p = test_model(x_test, y_test, cvae, sess)
    sess.close()

    lxs.append(lx[1:])
    lzs.append(lz[1:])

    train_perf.append(p[0])
    test_perf.append(p[1])
    print()


"""
MAKING PERFORMANCE TABLE
"""
performance = np.array([train_perf, test_perf])
np.save("../metrics_clf/simulated/performance_"+str(epochs)+".npy", performance)
names = ["Train", "Test"]
score_names = ["f1", "recall", "precision"]
classes = ["proton", "carbon"]

with open("../metrics_clf/simulated/f1_scores"+str(epochs)+".tex", "w") as fo:
    fo.write(r" & \multicolumn{3}{c}{Proton} & \multicolumn{3}{c}{Carbon} \\ "+" \n")
    fo.write(r" \hline "+"\n")
    fo.write(r" & ")
    fo.write(r" & ".join(score_names))
    fo.write(r" & ")
    fo.write(r" & ".join(score_names))
    fo.write(r"\\ " + " \n")
    for i in range(len(names)):
        fo.write(" "+ names[i]+" & ")
        for j in range(len(score_names)):
            metric = performance[i, :, j, :]
            avg = metric.mean(axis=0)
            std = metric.std(axis=0)
            for k in range(len(classes)):
                to_write = r"$ \underset{{ \num{{+- {:.2e} }} }} {{\num{{ {:.2f} }} }}  $".format(std[k], avg[k] )
                if not (k+1)*(j+1) == len(score_names)*len(classes):
                    to_write += " & "
                fo.write(to_write)
        if names[i] == "Test":
            fo.write("\n ")
        else:
            fo.write(r" \\ "+"\n ")

"""
PLOTTING LOSSES
"""
viridis = matplotlib.cm.get_cmap('viridis')
inferno = matplotlib.cm.get_cmap('inferno')
losses = [lxs, lzs]
colors = [viridis, inferno]
plot_names = ["reconst_loss", "latent_loss"]
yaxis_names = [r"$L_x$", r"$L_z$"]

for i in range(2):
    fig, ax = plt.subplots()
    loss = losses[i]
    end_vals = [l[-1] for l in loss]
    min_val = min(end_vals)
    for j, l in enumerate(loss): 
        if np.nan in l:
            continue
        c = colors[0](performance[1, j, 0, 0])
        if i == 0:
            f = ax.plot
        else:
            f = ax.semilogy
        f(
            np.arange(len(l)),
            l,
            linewidth=3,
            alpha=0.6,
            color=c
        )
        ax.set_xlabel("Epoch", fontsize=12) 
        ax.set_ylabel(yaxis_names[i], fontsize=12) 

    ax.xaxis.set_major_locator(ticker.MultipleLocator(int(epochs/10)))
    plt.tight_layout()
    plt.savefig("../plots/simulated_clf/"+plot_names[i]+""+str(epochs)+".pdf")
    plt.savefig("../plots/simulated_clf/"+plot_names[i]+""+str(epochs)+".png")
    plt.close(fig=fig)

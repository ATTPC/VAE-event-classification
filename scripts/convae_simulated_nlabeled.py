
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import tensorflow as tf
import os 

import sys
sys.path.append("../src")
from convolutional_VAE import ConVae
from data_loader import load_simulated
from latent_classifier import performance_by_samples

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
x_train, x_test, y_test = load_simulated(1)
#x_train = x_train[0:200]

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
        "batchnorm": True, 
        }
epochs = 200
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
        beta=10,
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
        save_checkpoints=0,
        verbose=0
        )

performance, percent = performance_by_samples(x_test, y_test, cvae, sess)
print(performance)
viridis = matplotlib.cm.get_cmap('viridis')
fig, ax = plt.subplots()
ax.errorbar(
        percent,
        performance[:,0],
        yerr=performance[:,1],
        color=viridis(0.6)
        )

ax.set_xlabel("Percent of train data")
ax.set_ylabel("Proton f1 score")
plt.savefig("../plots/convae_simulated_clf/n_labelled_perf.pdf")
plt.savefig("../plots/convae_simulated_clf/n_labelled_perf.png")

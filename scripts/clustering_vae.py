
import sys
sys.path.append("../src")
from convolutional_VAE import ConVae

import h5py

import numpy as np
import keras as ker
import os
import tensorflow as tf

from sklearn.model_selection import  cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from keras.utils import to_categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.datasets import mnist
from run import *


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
            C=10,
            solver="saga",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=1000
        )

    model.fit(X, y)

    train_score = f1_score(y, model.predict(X), average=None)
    test_score = f1_score(ytest, model.predict(Xtest), average=None)

    return train_score, test_score 


# Training dat
data = "clean_size"
print("Dataset: ", data)

if data=="clean":
    all_0130 = np.load("../data/clean/images/run_0130_label_False.npy")
    all_0170 = np.load("../data/clean/images/run_0190_label_False.npy")
    #all_0210 = np.load("../data/processed/all_0210.npy")
    #all_data = np.concatenate([all_0130, all_0210])
    all_data = np.concatenate([all_0130, all_0170])
    X = all_data

    train_data = np.load("../data/clean/images/train.npy")
    train_targets = np.load("../data/clean/targets/train_targets.npy")

    x_test = np.load("../data/clean/images/test.npy")
    y_test = np.load("../data/clean/targets/test_targets.npy")

elif data=="clean_size":
    size = 80
    all_0130 = np.load("../data/clean/images/run_0130_label_False_size_{}.npy".format(size))
    all_0170 = np.load("../data/clean/images/run_0190_label_False_size_{}.npy".format(size))
    all_data = np.concatenate([all_0130, all_0170])
    all_data = all_data[np.random.randint(0, all_data.shape[0], size=2000)]
    X = all_data

    train_data = np.load("../data/clean/images/train_size_{}.npy".format(size))
    train_targets = np.load("../data/clean/targets/train_targets_size_{}.npy".format(size))
    x_test = np.load("../data/clean/images/train_size_{}.npy".format(size))
    y_test = np.load("../data/clean/targets/train_targets_size_{}.npy".format(size))

elif data=="noisy":
    X = np.load("../data/processed/all_0130.npy")
    with h5py.File("../data/images.h5", "r") as fo:
        y_train = np.array(fo["train_targets"])
        y_test = np.array(fo["test_targets"])

    x_train = np.load("../data/processed/train.npy")
    x_test = np.load("../data/processed/test.npy")

    #train_test = np.concatenate((train_data, test_data))
    #train_test_targets = np.concatenate((train_targets, test_targets))
else:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1)/255
    X = x_train
    #x_train = x_train[0:1000]
    x_test= np.expand_dims(x_test, -1)/255

n_layers = 5
filter_architecture = [32, 64, 128, 256, 256*2]
kernel_arcitecture = [5, 4, 3, 2, 2]
strides_architecture = [1, 2, 2, 2, 1]

print("Filters: ", filter_architecture)
print("Kernel: ", kernel_arcitecture)
print("Strides: ", strides_architecture)

epochs = 10000
latent_dim = 30
batch_size = 120

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": False,
        "include_KM": True
        }

clustering_config = {
        "n_clusters":3,
        "alpha":1,
        "delta":0.01
        }

print("Config: ")
print(mode_config)

cvae = ConVae(
        n_layers,
        filter_architecture,
        kernel_arcitecture,
        strides_architecture,
        latent_dim,
        X,
        beta=10000,
        sampling_dim=150,
        mode_config=mode_config,
        clustering_config=clustering_config,
        labelled_data=(x_test, y_test) 
        )

loss_kwds = {"reconst_loss": None}
graph_kwds = {"activation": "tanh"}#tf.keras.layers.LeakyReLU(0.3)}
cvae.compile_model(loss_kwds=loss_kwds, graph_kwds=graph_kwds)

opt = tf.train.AdamOptimizer
opt_args = [0.0001, ]
opt_kwds = {
    "beta1": 0.8,
}

with open("run.py", "w") as fo:
    fo.write("run={}".format(run+1))

cvae.compute_gradients(opt, opt_args, opt_kwds)
sess = tf.InteractiveSession()
lx, lz = cvae.train(
        sess,
        epochs,
        "../drawing",
        "../models",
        batch_size,
        earlystopping=True,
        run=run
        )


sess.close()
sys.exit()
cvae.X = train_data
cvae.generate_latent(sess, "../drawing", (train_data, test_data))

latent_values , _, _ = cvae.generate_latent(
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

#cvae.generate_samples("../drawing", )


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 20), sharex=True)
plt.suptitle("Loss function components")

axs[0].plot(range(epochs), lx, label=r"$\mathcal{L}_x$")
axs[1].plot(range(epochs), lz, label=r"$\mathcal{L}_z$")

[a.legend() for a in axs]
[a.set_ylim((1000, 200)) for a in axs]

fig.savefig(
    "../plots/simulated_loss_functions.png")


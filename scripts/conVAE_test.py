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
#X = np.load("../data/processed/all_0130.npy")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test= np.expand_dims(x_test, -1)

"""
#Labelled data for testing
with h5py.File("../data/images.h5", "r") as fo:
    train_targets = np.array(fo["train_targets"])
    test_targets = np.array(fo["test_targets"])

train_data = np.load("../data/processed/train.npy")
test_data = np.load("../data/processed/test.npy")

train_test = np.concatenate((train_data, test_data))
"""

n_layers = 4
filter_architecture = [20, 80, 30, 5]
kernel_arcitecture = [7, 5, 3, 2]
strides_architecture = [1, 2, 1, 1]
epochs = 25

latent_dim = 600
batch_size = 100

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": False,
        "include_KM": True
        }

clustering_config = {
        "n_clusters":10,
        "alpha":1,
        }

cvae = ConVae(
        n_layers,
        filter_architecture,
        kernel_arcitecture,
        strides_architecture,
        latent_dim,
        x_train,
        beta=1000,
        mode_config=mode_config,
        clustering_config=clustering_config
        )

cvae.p_f = np.random.normal(size=(x_train.shape[0], 10, ))

with tf.variable_scope("clusters"):
    cvae.clusters = tf.Variable(np.random.normal(size=(10, latent_dim)).astype(np.float32))

cvae.compile_model()

opt = tf.train.AdamOptimizer
opt_args = [1e-4, ]
opt_kwds = {
    "beta1": 0.5,
}

cvae.compute_gradients(opt, opt_args, opt_kwds)

sess = tf.InteractiveSession()
lx, lz = cvae.train(
        sess,
        epochs,
        "../drawing",
        "../models",
        batch_size,
        earlystopping=False,
        )


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

sess.close()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 20), sharex=True)
plt.suptitle("Loss function components")

axs[0].plot(range(epochs), lx, label=r"$\mathcal{L}_x$")
axs[1].plot(range(epochs), lz, label=r"$\mathcal{L}_z$")

[a.legend() for a in axs]
[a.set_ylim((1000, 200)) for a in axs]

fig.savefig(
    "../plots/simulated_loss_functions.png")


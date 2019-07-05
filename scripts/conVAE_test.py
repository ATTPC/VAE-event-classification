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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

n_layers = 3
filter_architecture = [32, 64, 128,]
#kernel_architecture = [19, 19, 17]
strides_architecture = [2, 2, 2, ]
pool_architecture = [0, 0, 0, ]
n_clust = 3

mode_config = {
        "simulated_mode": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": True,
        "include_KM": False,
        "batchnorm":False
        }

clustering_config = {
        "n_clusters":n_clust,
        "alpha":1,
        "delta":0.01,
        "pretrain_simulated":False,
        "pretrain_epochs": 200,
        "update_interval": 140,
        }

data = "clean"

if data == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1)/255
    x_test= np.expand_dims(x_test, -1)/255
    x_tot = np.concatenate([x_train, x_test])
    kernel_architecture = [5, 5, 3]
    n_clust = 10
if data == "simulated":
    train_data = np.load("../data/simulated/pr_train_simulated.npy")
    test_data = np.load("../data/simulated/pr_test_simulated.npy")
    test_targets = np.load("../data/simulated/test_targets.npy")
    x_train = train_data
    x_test = test_data
    y_test = test_targets
    kernel_architecture = [5, 5, 3, 3]
    filter_architecture = [16, 32, 64, 64]
    clustering_config["pretrain_epochs"] = 20
    clustering_config["n_clusters"] = 2
    n_layers = 4
    #kernel_architecture = [19, 19, 17]
elif data=="clean":
    run_130 = np.load("../data/clean/images/run_0130_label_False_size_80.npy")[:, 1:, 1:, :]
    run_150 = np.load("../data/clean/images/run_0150_label_False_size_80.npy")[:, 1:, 1:, :]
    #run_170 = np.load("../data/clean/images/run_0170_label_False_size_50.npy")[:, 1:-1, 1:-1, :]
    run_190 = np.load("../data/clean/images/run_0190_label_False_size_80.npy")[:, 1:, 1:, :]
    run_210 = np.load("../data/clean/images/run_0210_label_False_size_80.npy")[:, 1:, 1:, :]
    x_train = np.concatenate([run_130, run_150,  run_190, run_210])
    x_test = np.load("../data/clean/images/train_size_80.npy")[:, 1:, 1:, :]
    y_test = np.load("../data/clean/targets/train_targets_size_80.npy")
    y_test = np.squeeze(y_test)
    kernel_architecture = [5, 5, 3, 3]
    filter_architecture = [32, 64, 128, 128]
    strides_architecture += [2,]
    pool_architecture += [0,]
    clustering_config["pretrain_epochs"] = 200
    clustering_config["n_clusters"] = 3
    n_layers = 4
if data == "real":
    #Labelled data for testing
    with h5py.File("../data/images.h5", "r") as fo:
        train_targets = np.array(fo["train_targets"])
        test_targets = np.array(fo["test_targets"])
    all_0130 = np.load("../data/processed/all_0130.npy")
    x_train = all_0130
    #train_data = np.load("../data/processed/train.npy")
    test_data = np.load("../data/processed/test.npy")
    train_data = np.load("../data/processed/train.npy")
    x_test = train_data
    y_test = train_targets 
    kernel_architecture = [7, 7, 5, 5]
    filter_architecture = [32, 64, 64, 128]
    clustering_config["pretrain_epochs"] = 90
    clustering_config["n_clusters"] = 3
    n_layers = 4

epochs = 2000

latent_dim = 9
batch_size = 256

cvae = ConVae(
        n_layers,
        filter_architecture,
        kernel_architecture,
        strides_architecture,
        pool_architecture,
        latent_dim,
        x_train,
        #all_0130,
        beta=11,
        #sampling_dim=100,
        clustering_config=clustering_config,
        mode_config=mode_config,
        #labelled_data=[test_data, test_targets],
        labelled_data=[x_test, y_test]
        )

graph_kwds = {"activation":"relu", "output_activation":None}
loss_kwds = {"reconst_loss": None}
cvae.compile_model(graph_kwds, loss_kwds)

opt = tf.train.AdamOptimizer
opt_args = [1e-3, ]
opt_kwds = {
    "beta1": 0.7,
}

cvae.compute_gradients(opt,)
sess = tf.InteractiveSession()

lx, lz = cvae.train(
        sess,
        epochs,
        "../drawing",
        "../models",
        batch_size,
        earlystopping=True,
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


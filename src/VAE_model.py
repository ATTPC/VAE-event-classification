#!/usr/bin/env python3

from data_loader import DataLoader

from keras import backend as K
import keras as ker

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Reshape
from keras.layers import Conv2DTranspose
from keras.layers import ZeroPadding2D, ZeroPadding3D
from keras.layers import Input, Lambda

from keras.losses import mse, binary_crossentropy

import keras.regularizers as reg
import keras.optimizers as opt

import matplotlib.pyplot as plt


file_location = "/home/solli-comphys/github/VAE-event-classification/data/real/packaged/x-y/proton-carbon-junk-noise.h5"
X_train, y_train, X_test, y_test = DataLoader(file_location)


def rgb2gray(rgb):

    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


X_train = rgb2gray(X_train) / 255
X_test = rgb2gray(X_test) / 255

X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


kernel_size = 4
filters = 20
latent_dim = 10
num_layers = 2

in_layer = Input(shape=(128, 128, 1))
h1 = in_layer
shape = K.int_shape(h1)

for i in range(1, num_layers + 1):
    filters *= 2
    h1 = Conv2D(
        filters,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        use_bias=True,
        kernel_regularizer=reg.l2(0.01),
        bias_regularizer=reg.l2(0.01),
    )(h1)


shape = K.int_shape(h1)
h1 = Flatten()(h1)

h1 = Dense(16, activation="relu")(h1)
mean = Dense(latent_dim)(h1)
var = Dense(latent_dim)(h1)

sample = Lambda(sampling, output_shape=(latent_dim,))([mean, var])

encoder = Model(in_layer, [mean, var, sample], name="encoder")
encoder.summary()

# %%
latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
de1 = Dense(shape[1] * shape[2] * shape[3], activation="relu")(latent_inputs)

de1 = Reshape((shape[1], shape[2], shape[3]))(de1)

for i in reversed(range(1, num_layers + 1)):
    de1 = Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        use_bias=True,
        kernel_regularizer=reg.l2(0.01),
        bias_regularizer=reg.l2(0.01),
    )(de1)
    filters //= 2

outputs = Conv2DTranspose(
    filters=1,
    kernel_size=kernel_size,
    activation="sigmoid",
    padding="same",
    use_bias=True,
    kernel_regularizer=reg.l2(0.01),
    bias_regularizer=reg.l2(0.01),
    name="decoder_output",
)(de1)
decoder = Model(input=latent_inputs, output=outputs)

outputs = decoder(encoder(in_layer)[2])
vae = Model(in_layer, outputs, name="vae")


def vae_loss(y_true, y_pred):
    xent_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)) * 784
    kl_loss = -0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss


vae.compile(optimizer="adam", loss=[vae_loss])

# %%

earlystop = ker.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=2,
    patience=0,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)

tensorboard = ker.callbacks.TensorBoard(
    log_dir="./Graph", write_graph=True, histogram_freq=0, write_images=True
)

vae.fit(
    X_train,
    X_train,
    validation_data=(X_test, X_test),
    epochs=20,
    batch_size=100,
    callbacks=[earlystop, tensorboard],
)


vae.save("/home/solli-comphys/github/VAE-event-classification/models/attpc_vae.h5")
encoder.save("/home/solli-comphys/github/VAE-event-classification/models/attpc_enc.h5")
decoder.save("/home/solli-comphys/github/VAE-event-classification/models/attpc_dec.h5")

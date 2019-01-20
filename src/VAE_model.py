#!/usr/bin/env python3

import sys

from keras import backend as K
import keras as ker
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

#tf.enable_eager_execution()

# %%

H, W = 128, 128 #image dimensions
n_pixels = H*W #number of pixels in image
kernel_size = [3, 3]
dec_size = 256
T = 2
batch_size = 100
input_size = (batch_size, H, W, 1)

train_iters = 2
eta = 1e-3
eps = 1e-8

read_size = 2*n_pixels
write_size = n_pixels
latent_dim = 20

DO_SHARE=None

# network variables


tf.flags.DEFINE_boolean("read_attn", False, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", False, "enable attention for writer")
FLAGS = tf.flags.FLAGS

e = tf.random_normal((batch_size, latent_dim), mean=0, stddev=1)
x = tf.placeholder(tf.float32, shape=(batch_size, H, W, 1))

encoder = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2,
            input_shape=[H, W, 1],
            output_channels=1,
            kernel_shape=kernel_size
            )

decoder = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)


# network operations

def linear_conv(x, output_dim):
    w = tf.get_variable("w", [128, 128, output_dim])
    b = tf.get_variable(
            "b",
            [output_dim],
            initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.tensordot(x, w, [[1, 2], [0, 1]]), (100, 20)) + b

def linear(x, output_dim):
    w = tf.get_variable("w", [x.get_shape()[1], output_dim])
    b = tf.get_variable(
            "b",
            [output_dim],
            initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def read(x, x_hat, h_dec_prev):
    return tf.concat([x, x_hat], 3)


def encode(state, input):
    with tf.variable_scope("encoder", reuse=DO_SHARE):
        return encoder(input, state)


def sample(h_enc):
    """
    samples z_t from NormalDistribution(mu, sigma)
    """
    with tf.variable_scope("mu", reuse=DO_SHARE):
        mu = linear_conv(h_enc, latent_dim)
    with tf.variable_scope("sigma", reuse=DO_SHARE):
        logsigma = linear_conv(h_enc, latent_dim)
        sigma = tf.exp(logsigma)

    return (mu + sigma*e, mu, logsigma, sigma)

def decode(state, input):
    with tf.variable_scope("decoder", reuse=DO_SHARE):
        return decoder(input, state)


def write(h_dec):
    with tf.variable_scope("write", reuse=DO_SHARE):
        return linear(h_dec, n_pixels)


canvas_seq = [0]*T
mus, logsigmas, sigmas = [0]*T, [0]*T, [0]*T

#initial states
h_dec_prev = tf.zeros(input_size)
enc_state = encoder.zero_state(batch_size, tf.float32)
dec_state = decoder.zero_state(batch_size, tf.float32)

for t in range(T):
    c_prev = tf.zeros(input_size) if t == 0 else canvas_seq[t-1]
    x_hat = x - tf.sigmoid(c_prev)
    r = read(x, x_hat, h_dec_prev)
    h_enc, enc_state = encode(enc_state, tf.concat([r, h_dec_prev], 3))
    z, mus[t], logsigmas[t], sigmas[t] = sample(h_enc)
    h_dec, dec_state = decode(dec_state, z)
    canvas_seq[t] = c_prev+tf.reshape(write(h_dec), (100, 128, 128, 1))
    DO_SHARE = True


sys.exit()

# %%


t_fn = "/home/solli-comphys/github/VAE-event-classification/data/processed/train.npy"
te_fn = "/home/solli-comphys/github/VAE-event-classification/data/processed/test.npy"

X_train = np.load(t_fn)
X_test = np.load(te_fn)

earlystop = ker.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=2,
                            patience=0,
                            verbose=0,
                            mode='auto',
                            restore_best_weights=True
                              )


vae.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        epochs=20,
        batch_size=50,
        callbacks=[earlystop, ]
    )


vae.save("/home/solli-comphys/github/VAE-event-classification/models/attpc_vae.h5")
encoder.save("/home/solli-comphys/github/VAE-event-classification/models/attpc_enc.h5")
decoder.save("/home/solli-comphys/github/VAE-event-classification/models/attpc_dec.h5")

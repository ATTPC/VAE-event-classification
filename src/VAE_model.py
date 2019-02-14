#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import tensorflow as tf

import matplotlib
matplotlib.use("Agg")


# tf.enable_eager_execution()
print(os.getpid())

# %%
test_mode = False

H, W = 128, 128  # image dimensions
n_pixels = H*W  # number of pixels in image
N = 3 if test_mode else 60 # number of filters
use_attention = True

dec_size = 10 if test_mode else 1200
enc_size = 10 if test_mode else 1200
T = 5 if test_mode else 10
batch_size = 50

epochs = 5 if test_mode else 150
eta = 1e-3
eps = 1e-8

read_size = 2*n_pixels
write_size = n_pixels
latent_dim = 10 if test_mode else 8

DO_SHARE = None

# network variables

FLAGS = tf.flags.FLAGS

x = tf.placeholder(tf.float32, shape=(batch_size, n_pixels))
regularizer = None

encoder = tf.nn.rnn_cell.LSTMCell(
    enc_size,
    state_is_tuple=True,
    activity_regularizer=tf.contrib.layers.l2_regularizer(0.01),
)

decoder = tf.nn.rnn_cell.LSTMCell(
    dec_size,
    state_is_tuple=True,
    activity_regularizer=tf.contrib.layers.l2_regularizer(0.01),
)

print(encoder.output_size, decoder.output_size)

# reshape operation


def longform(x):
    return tf.reshape(x, (batch_size, n_pixels))


# network operations

def linear_conv(x, output_dim,):
    w = tf.get_variable("w", [128, 128, output_dim])
    b = tf.get_variable(
        "b",
        [output_dim],
        initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.tensordot(x, w, [[1, 2], [0, 1]]), (100, 20)) + b


def linear(x, output_dim, regularizer=tf.contrib.layers.l2_regularizer, lmbd=0.1):
    w = tf.get_variable("w", [x.get_shape()[1], output_dim],
                        regularizer=regularizer(lmbd),
                        )
    b = tf.get_variable(
        "b",
        [output_dim],
        initializer=tf.constant_initializer(0.0),
        regularizer=regularizer(lmbd))

    return tf.matmul(x, w) + b


def read_no_attn(x, x_hat, h_dec_prev):
    return tf.concat([x, x_hat], 1)


def encode(state, input):
    with tf.variable_scope("encoder", reuse=DO_SHARE):
        return encoder(input, state)


def sample(h_enc):
    """
    samples z_t from NormalDistribution(mu, sigma)
    """
    e = tf.random_normal((batch_size, latent_dim), mean=0, stddev=1)

    with tf.variable_scope("mu", reuse=DO_SHARE):
        mu = linear(h_enc, latent_dim, lmbd=0.1)
    with tf.variable_scope("sigma", reuse=DO_SHARE):
        sigma = linear(h_enc, latent_dim,
                       lmbd=0.1,
                       regularizer=tf.contrib.layers.l1_regularizer)
        sigma = tf.clip_by_value(sigma, 1, 1e4)
        logsigma = tf.log(sigma)

    return (mu + sigma*e, mu, logsigma, sigma)


def decode(state, input):
    with tf.variable_scope("decoder", reuse=DO_SHARE):
        return decoder(input, state)


def write_no_attn(h_dec):
    with tf.variable_scope("write", reuse=DO_SHARE):
        return linear(h_dec, n_pixels)


def attn_params(scope, h_dec):
    with tf.variable_scope(scope, reuse=DO_SHARE):
        tmp = linear(h_dec, 5)
        gx, gy, logsigma_sq, logdelta, loggamma = tf.split(
            tmp, 5, 1)

    sigma_sq = tf.exp(logsigma_sq)
    delta = tf.exp(logdelta)
    gamma = tf.exp(loggamma)

    gx = (H + 1)/2 * (gx + 1)
    gy = (W + 1)/2 * (gy + 1)
    delta = (H - 1)/(N - 1) * delta

    return gx, gy, sigma_sq, delta, gamma


def filters(gx, gy, sigma_sq, delta, gamma, N):
    i = tf.convert_to_tensor(np.arange(N, dtype=np.float32))

    mu_x = gx + (i - N/2 - 0.5) * delta  # batch_size, N
    mu_y = gy + (i - N/2 - 0.5) * delta
    # print(mu_x.get_shape(), gx.get_shape(), i.get_shape())
    a = tf.convert_to_tensor(np.arange(H, dtype=np.float32))
    b = tf.convert_to_tensor(np.arange(W, dtype=np.float32))

    A, MU_X = tf.meshgrid(a, mu_x)  # batch_size, N * H
    B, MU_Y = tf.meshgrid(b, mu_y)

    A = tf.reshape(A, [batch_size, N, H])
    B = tf.reshape(B, [batch_size, N, W])

    MU_X = tf.reshape(MU_X, [batch_size, N, H])
    MU_Y = tf.reshape(MU_Y, [batch_size, N, W])

    sigma_sq = tf.reshape(sigma_sq, [batch_size, 1, 1])

    Fx = tf.exp(- tf.square(A - MU_X)/(2*sigma_sq))
    Fy = tf.exp(- tf.square(B - MU_Y)/(2*sigma_sq))

    Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 1, keepdims=True), eps)
    Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 1, keepdims=True), eps)

    return Fx, Fy


def read_attn(x, xhat, h_dec_prev, Fx, Fy, gamma):
    Fx_t = tf.transpose(Fx, perm=[0, 2, 1])

    x = tf.reshape(x, [batch_size, 128, 128])
    xhat = tf.reshape(xhat, [batch_size, 128, 128])

    FyxFx_t = tf.reshape(tf.matmul(Fy, tf.matmul(x, Fx_t)), [-1, N*N])
    FyxhatFx_t = tf.reshape(tf.matmul(Fy, tf.matmul(x, Fx_t)), [-1, N*N])

    return gamma * tf.concat([FyxFx_t, FyxhatFx_t], 1)


def write_attn(h_dec, Fx, Fy, gamma):
    with tf.variable_scope("writeW", reuse=DO_SHARE):
        w = linear(h_dec, write_size, tf.contrib.layers.l1_regularizer, lmbd=1e-2)

    w = tf.reshape(w, [-1, H, W])
    Fy_t = tf.transpose(Fy, perm=[0, 2, 1])
    tmp = tf.matmul(w, Fx)
    tmp = tf.reshape(tf.matmul(Fy_t, tmp), [-1, H*H])

    return tmp/tf.maximum(gamma, eps)


def read_a(x, xhat, h_dec_prev):
    params = attn_params("read", h_dec_prev)
    Fx, Fy = filters(*params, N)
    return read_attn(x, xhat, h_dec_prev, Fx, Fy, params[-1])


def write_a(h_dec):
    params_m = attn_params("write", h_dec)
    Fx_m, Fy_m = filters(*params_m, H)
    return write_attn(h_dec, Fx_m, Fy_m, params_m[-1])


read = read_a if use_attention else read_no_attn
write = write_a if use_attention else write_no_attn

canvas_seq = [0]*T
mus, logsigmas, sigmas = [0]*T, [0]*T, [0]*T

# initial states
h_dec_prev = tf.zeros((batch_size, dec_size))
c_prev = tf.zeros((batch_size, n_pixels))

enc_state = encoder.zero_state(batch_size, tf.float32)
dec_state = decoder.zero_state(batch_size, tf.float32)

# Unrolling the computational graph for the LSTM
for t in range(T):
    # computing the error image
    x_hat = x - tf.sigmoid(c_prev)
    r = read(x, x_hat, h_dec_prev)

    h_enc, enc_state = encode(enc_state, tf.concat([r, h_dec_prev], 1))
    z, mus[t], logsigmas[t], sigmas[t] = sample(h_enc)
    h_dec, dec_state = decode(dec_state, z)
    canvas_seq[t] = c_prev+write(h_dec)

    h_dec_prev = h_dec
    c_prev = canvas_seq[t]
    DO_SHARE = True


# Loss functions Lx Lz

def binary_crossentropy(t, o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))


x_recons = tf.sigmoid(canvas_seq[-1])
#x_recons = canvas_seq[-1]

Lx = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x, x_recons), 1))
#Lx = tf.losses.mean_squared_error(x, x_recons)  # tf.reduce_mean(Lx)
#Lx = tf.losses.mean_pairwise_squared_error(x, x_recons)

KL_loss = [0]*T

for t in range(T):
    mu_sq = tf.square(mus[t])
    sigma_sq = tf.square(sigmas[t])
    logsigma_sq = tf.square(logsigmas[t])

    KL_loss[t] = tf.reduce_sum(mu_sq + sigma_sq - 2*logsigma_sq, 1)

KL = 0.5 * tf.add_n(KL_loss) - T/2
Lz = tf.reduce_mean(KL)

cost = Lx + Lz
#reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#reg_var = tf.sum(reg_var)
cost += tf.losses.get_regularization_loss()

# OPTIMIZATION

optimizer = tf.train.AdamOptimizer(eta, beta1=0.5)
grads = optimizer.compute_gradients(cost)

for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)

train_op = optimizer.apply_gradients(grads)

t_fn = "../data/processed/train.npy"
te_fn = "../data/processed/test.npy"

X_train = np.load(t_fn)
n_train = X_train.shape[0]

X_test = np.load(te_fn)
n_test = X_test.shape[0]

print("N Train: ", n_train, "| N test :", n_test )

train_iters = n_train // batch_size

fetches = []
fetches.extend([Lx, Lz, z, train_op])

sess = tf.InteractiveSession()

saver = tf.train.Saver()
tf.global_variables_initializer().run()

for v in tf.global_variables():
    print("%s : %s " % (v.name, v.get_shape()))


def store_result():
    print("saving model and drawings")
    canvasses = sess.run(canvas_seq, feed_dict)
    canvasses = np.array(canvasses)
    references = np.array(feed_dict[x])
    
    filename = "../drawing/canvasses.npy"
    np.save(filename, canvasses)

    model_fn = "../models/draw.ckpt"
    saver.save(sess, model_fn)

    ref_fn = "../drawing/references.npy"
    np.save(ref_fn, references)

class BatchManager:

    def __init__(self, nSamples):
        self.available = np.arange(0, nSamples)
        self.nSamples = nSamples

    def fetch_minibatch(self, x):
        tmp_ind = np.random.randint(0, self.nSamples, size=(batch_size, ))
        ind = self.available[tmp_ind]

        self.available = np.delete(self.available, tmp_ind)
        self.nSamples = len(self.available)
        return X_train[ind].reshape(batch_size, n_pixels)


n_mvavg = 5
moving_average = [0] * (epochs // n_mvavg)
best_average = 1e5
to_average = [0]*n_mvavg
ta_which = 0
all_lx = [0]*epochs
all_lz = [0]*epochs

for i in range(epochs):
    Lxs = [0]*train_iters
    Lzs = [0]*train_iters
    bm_inst = BatchManager(len(X_train))

    for j in range(train_iters):
        x_train = bm_inst.fetch_minibatch(X_train)
        feed_dict = {x: x_train}
        results = sess.run(fetches, feed_dict)
        Lxs[j], Lzs[j], _, _ = results

#        with tf.variable_scope("sigma", reuse=DO_SHARE):
#            w = tf.get_variable("w",)
#            print(sess.run(w))
#
    all_lz[i] = tf.reduce_mean(Lzs).eval()
    all_lx[i] = tf.reduce_mean(Lxs).eval()

    if all_lz[i] < 0:
        print("broken training")
        print("Lx = ", all_lx[i])
        print("Lz = ", all_lz[i])
        sess.close()
        break

    to_average[ta_which] = all_lx[i] + all_lz[i]
    ta_which += 1

    if (1 + i) % n_mvavg == 0 and i > 0:
        ta_which = 0
        moving_average[i // n_mvavg] = tf.reduce_mean(to_average).eval()
        to_average = [0] * n_mvavg

        if moving_average[i // n_mvavg] < best_average and i > 1:
            store_result()
            best_average = moving_average[i // n_mvavg]
        print("epoch=%d : Lx: %f Lz: %f mvAVG: %f" % (
            i,
            all_lx[i],
            all_lz[i],
            moving_average[i//n_mvavg]))

# TRAINING DONE


# Generate samples 
"""
n_samples = 10
generated_samples = np.zeros((n_samples, T))

dec_state = decoder.zero_state(batch_size, tf.float32)
h_dec = tf.zeros((batch_size, dec_size))
canvas_seq = [0]*T

for t in range(T):
    z = tf.convert_to_tensor(np.random.rand(batch_size, latent_dim).astype(np.float32))
    h_dec, dec_state = decode(dec_state, z)
    canvas_seq[t] = c_prev+write(h_dec)

    h_dec_prev = h_dec
    c_prev = canvas_seq[t]
"""

# Generate latent expressions
X_tup = (X_train, X_test)
lat_vals = []


for i in range(2):
    X = X_tup[i]
    n_latent = (X.shape[0]//batch_size)*batch_size
    latent_values = np.zeros((n_latent, latent_dim))
    for i in range(X.shape[0]//batch_size):
        start = i * batch_size
        end = (i+1) * batch_size
        to_feed = X[start:end].reshape((batch_size, H*W))
        feed_dict = {x: to_feed}

        _, _, z, _ = sess.run(fetches, feed_dict)
        latent_values[start:end] = z
    
    lat_vals.append(latent_values)

for i in range(2):
    fn = "train_latent.npy" if i == 0 else "test_latent.npy"
    l = lat_vals[i]
    np.save("../drawing/latent/" + fn, l)

sess.close()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 20), sharex=True)
plt.suptitle("Loss function components")

axs[0].plot(range(epochs), all_lx, label=r"$\mathcal{L}_x$")
axs[1].plot(range(epochs), all_lz, label=r"$\mathcal{L}_z$")

[a.legend() for a in axs]

fig.savefig(
    "../plots/loss_functions.png")

print("print DONE!")





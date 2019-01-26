#!/usr/bin/env python3

import sys
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

# tf.enable_eager_execution()

# %%
test_mode = True

H, W = 128, 128  # image dimensions
n_pixels = H*W  # number of pixels in image
kernel_size = [2, 2]
dec_size = 10 if test_mode else 100  # 00
enc_size = 10 if test_mode else 100  # 00
T = 5 if test_mode else 20
batch_size = 13
input_size = (batch_size, H, W, 1)

epochs = 20 if test_mode else 100
eta = 1e-3
eps = 1e-8

read_size = 2*n_pixels
write_size = n_pixels
latent_dim = 13 if test_mode else 50

DO_SHARE = None

# network variables

FLAGS = tf.flags.FLAGS

e = tf.random_normal((batch_size, latent_dim), mean=0, stddev=1)
x = tf.placeholder(tf.float32, shape=(batch_size, n_pixels))

# encoder = tf.contrib.rnn.ConvLSTMCell(
#             conv_ndims=2,
#             input_shape=[H, W, 1],
#             output_channels=1,
#             kernel_shape=kernel_size,
#             )

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
initializer = tf.initializers.glorot_uniform()

decoder = tf.contrib.rnn.ConvLSTMCell(
    conv_ndims=2,
    input_shape=[H, W, 1],
    kernel_shape=kernel_size,
    output_channels=1
)

encoder = tf.nn.rnn_cell.LSTMCell(
    enc_size,
    state_is_tuple=True,
    activity_regularizer=tf.contrib.layers.l2_regularizer(0.01),
)

print(encoder.output_size, decoder.output_size)

# reshape operation


def longform(x):
    return tf.reshape(x, (batch_size, n_pixels))


def wideform(x):
    return tf.reshape(x, (batch_size, H, W, 1))


# network operations

def linear_conv(x, output_dim):
    w = tf.get_variable("w", [128, 128, output_dim],
                        regularizer=regularizer
                        )
    b = tf.get_variable(
        "b",
        [output_dim],
        initializer=tf.constant_initializer(0.0),
        regularizer=regularizer
    )
    return tf.reshape(tf.tensordot(x, w, [[1, 2], [0, 1]]), (batch_size, latent_dim)) + b


def linear(x, output_dim):
    w = tf.get_variable("w", [x.get_shape()[1], output_dim],
                        regularizer=regularizer,
                        )
    b = tf.get_variable(
        "b",
        [output_dim],
        initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, w)  # + b


def read(x, x_hat, h_dec_prev):
    return tf.concat([x, x_hat], 1)


def encode(state, input):
    with tf.variable_scope("encoder", reuse=DO_SHARE):
        return encoder(input, state)


def sample(h_enc):
    """
    samples z_t from NormalDistribution(mu, sigma)
    """
    with tf.variable_scope("mu", reuse=DO_SHARE):
        mu = linear(h_enc, latent_dim)
    with tf.variable_scope("sigma", reuse=DO_SHARE):
        logsigma = linear(h_enc, latent_dim)
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

# initial states
h_dec_prev = tf.zeros((batch_size, n_pixels))
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

    with tf.variable_scope("z", reuse=DO_SHARE):
        z_wide = wideform(linear(z, n_pixels))

    h_dec, dec_state = decode(dec_state, z_wide)

    canvas_seq[t] = c_prev + longform(h_dec)

    h_dec_prev = longform(h_dec)
    c_prev = canvas_seq[t]
    DO_SHARE = True


# Loss functions Lx Lz

def binary_crossentropy(t, o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))


x_recons = tf.tanh(canvas_seq[-1])
# Lx = tf.reduce_sum(binary_crossentropy(longform(x), x_recons), 1)
# Lx = tf.losses.mean_squared_error(x, x_recons)  # tf.reduce_mean(Lx)
Lx = tf.losses.mean_pairwise_squared_error(x, x_recons)

KL_loss = [0]*T

for t in range(T):
    mu_sq = tf.square(mus[t])
    sigma_sq = tf.square(sigmas[t])
    logsigma_sq = tf.square(logsigmas[t])

    KL_loss[t] = tf.reduce_sum(mu_sq + sigma_sq - 2*logsigma_sq, 1)

KL = 0.5 * tf.add_n(KL_loss) - T/2
Lz = 1/(T * latent_dim) * tf.reduce_mean(KL)

cost = Lx + Lz
reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_var)
cost += reg_term

# OPTIMIZATION

optimizer = tf.train.AdamOptimizer(eta, beta1=0.5)
grads = optimizer.compute_gradients(cost)

for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)

train_op = optimizer.apply_gradients(grads)

t_fn = "/Users/solli/Documents/github/VAE-event-classification/data/processed/train.npy"
te_fn = "/Users/solli/Documents/github/VAE-event-classification/data/processed/test.npy"

X_train = np.load(t_fn)
n_train = X_train.shape[0]

X_test = np.load(te_fn)
n_test = X_test.shape[0]

train_iters = n_train // batch_size

fetches = []
fetches.extend([Lx, Lz, train_op])

sess = tf.InteractiveSession()

saver = tf.train.Saver()
tf.global_variables_initializer().run()

for v in tf.global_variables():
    print("%s : %s " % (v.name, v.get_shape()))


def store_result():
    print("saving model and drawings")
    canvasses = sess.run(canvas_seq, feed_dict)
    canvasses = np.array(canvasses)

    filename = "/Users/solli/Documents/github/VAE-event-classification/drawing/canvasses.npy"
    np.save(filename, canvasses)

    model_fn = "/Users/solli/Documents/github/VAE-event-classification/models/draw.ckpt"
    saver.save(sess, model_fn)


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
        Lxs[j], Lzs[j], _ = results

#        with tf.variable_scope("sigma", reuse=DO_SHARE):
#            w = tf.get_variable("w",)
#            print(sess.run(w))
#
    all_lz[i] = tf.reduce_mean(Lzs).eval()
    all_lx[i] = tf.reduce_mean(Lxs).eval()

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


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 20), sharex=True)
plt.suptitle("Loss function components")

axs[0].plot(range(epochs), all_lx, label=r"$\mathcal{L}_x$")
axs[1].plot(range(epochs), all_lz, label=r"$\mathcal{L}_z$")

fig.savefig(
    "/Users/solli/Documents/github/VAE-event-classification/plots/loss_functions.png")

sess.close()

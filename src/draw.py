import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Input,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Lambda

from keras import backend as K

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import signal
import sys

from batchmanager import BatchManager
from model import LatentModel


class DRAW(LatentModel):
    """
    An implementation of the DRAW algorithm proposed by Gregor et. al 
    ArXiv id: 1502.04623v2

    Model requires initialization with hyperparameters and optional configuration 
    detailed in __init__ 

    Importantly the data is taken at initialization and accessed with Train. 

    Model is compiled with the CompileModel and trained with the Train method. 
    """

    def __init__(
        self,
        T,
        dec_size,
        enc_size,
        latent_dim,
        X,
        beta=1,
        train_classifier=False,
        use_attention=None,
        use_conv=None,
        X_classifier=None,
        Y_classifier=None,
        labelled_data=None,
        attn_config=None,
        mode_config=None,
        conv_architecture=None,
        clustering_config=None,
        test_split=0,
    ):

        # adding save on interrupt

        self.simulated_mode = False
        self.restore_mode = False
        self.include_KL = False
        self.include_MMD = False
        self.include_KM = False
        self.pretrain_simulated = False
        self.use_vgg = False
        self.conv_architecture = conv_architecture

        super().__init__(X, latent_dim, beta, mode_config)

        self.T = T
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.eps = 1e-8

        self.train_classifier = train_classifier
        self.labelled_data = labelled_data

        if self.train_classifier:
            test_split = test_split if test_split != 0 else 0.25
            self.X_c, self.X_c_test, self.Y_c, self.Y_c_test = train_test_split(
                X_classifier, Y_classifier, test_size=test_split
            )

        self.use_attention = use_attention
        self.use_conv = use_conv

        self.restore_mode = False
        self.simulated_mode = False

        if self.use_attention and attn_config is None:
            print(
                """If attention is used then parameters read_N, write_N and corresponding 
                    deltas must be supplied in dict attn_config"""
            )

        elif self.use_attention:
            for key, val in attn_config.items():
                if isinstance(val, (np.ndarray,)):
                    val = tf.convert_to_tensor(val)

                setattr(self, key, val)

        if self.include_KM and clustering_config == None:
            raise RuntimeError("when KM is true a config must be supplied")
        elif self.include_KM and clustering_config != None:
            for key, val in clustering_config.items():
                setattr(self, key, val)

        if self.use_conv:
            if self.conv_architecture is None:
                raise RuntimeError("When conv is true must supply conv architecture")

        self.DO_SHARE = None

    def _ModelGraph(
        self,
        initializer=tf.initializers.glorot_normal,
        kernel_reg_strength=1e-2,
        activation="tanh",
        output_activation="sigmoid",
        n_encoder_cells=2,
        n_decoder_cells=2,
    ):
        self.kernel_reg_strength = kernel_reg_strength
        activations = {
            "relu": tf.keras.layers.ReLU(),
            "lrelu": tf.keras.layers.LeakyReLU(),
            "tanh": Lambda(tf.keras.activations.tanh),
            "sigmoid": Lambda(tf.keras.activations.sigmoid),
        }
        if self.use_conv:
            self.activation = activations[self.conv_architecture["activation_func"]]
        else:
            self.activation = activations[activation]

        self.x = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.batch_size = tf.shape(self.x)[0]

        encoder_cells = []
        decoder_cells = []
        self.n_decoder_cells = n_decoder_cells
        self.n_encoder_cells = n_encoder_cells

        for i in range(n_encoder_cells):
            encoder_cells.append(
                tf.nn.rnn_cell.LSTMCell(
                    self.enc_size,
                    state_is_tuple=True,
                    activity_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_strength),
                    activation=self.activation,
                    initializer=initializer,
                )
            )

        for i in range(n_decoder_cells):
            decoder_cells.append(
                tf.nn.rnn_cell.LSTMCell(
                    self.dec_size,
                    state_is_tuple=True,
                    activity_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_strength),
                    activation=self.activation,
                    initializer=initializer,
                )
            )

        self.encoder = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)
        self.decoder = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)

        self.read = self.read_a if self.use_attention else self.read_no_attn
        self.write = self.write_a if self.use_attention else self.write_no_attn

        if self.use_conv:
            self.read = self.read_conv
            self.write = self.write_conv

        self.canvas_seq = [0] * self.T
        self.z_seq = [0] * self.T
        self.dec_state_seq = [0] * self.T

        self.mus, self.logsigmas, self.sigmas = [0] * self.T, [0] * self.T, [0] * self.T

        # initial states
        h_dec_prev = tf.zeros((self.batch_size, self.dec_size))
        c_prev = tf.zeros((self.batch_size, self.n_input))

        dec_state = self.decoder.zero_state(self.batch_size, tf.float32)
        enc_state = self.encoder.zero_state(self.batch_size, tf.float32)
        self.latent_cell = tf.nn.rnn_cell.GRUCell(
            self.latent_dim, activation=self.activation
        )
        latent_state = self.latent_cell.zero_state(self.batch_size, tf.float32)

        # Unrolling the computational graph for the LSTM
        for t in range(self.T):
            # computing the error image
            if t == 0:
                x_hat = c_prev
            else:
                x_hat = self.x - c_prev

            """ Encoder operations  """
            r = self.read(self.x, x_hat, h_dec_prev)
            if self.batchnorm:
                r = BatchNormalization(axis=-1, center=True, scale=True, epsilon=1e-4)(
                    r
                )

            # r = tf.tanh(r)
            h_enc, enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1))
            if self.batchnorm:
                h_enc = BatchNormalization(
                    axis=-1, center=True, scale=True, epsilon=1e-4
                )(h_enc)

            """ Compute latent sample """
            z  = self.compute_latent_sample(t, h_enc)

            """ Decoder operations """
            h_dec, dec_state = self.decode(dec_state, z)
            # dropout_h_dec = tf.keras.layers.Dropout(0.1)(h_dec, )
            if self.batchnorm:
                h_dec = BatchNormalization(
                    axis=-1, center=True, scale=True, epsilon=1e-4
                )(h_dec)
            if t < (self.T-1):
                self.canvas_seq[t] = c_prev + self.write(h_dec)
            else:
                if output_activation is None:
                    self.canvas_seq[t] = c_prev + self.write(h_dec)
                else:
                    out_act = activations[output_activation]
                    self.canvas_seq[t] = out_act(c_prev + self.write(h_dec))

            """Summarise variables """
            vars_ = [z, h_enc, h_dec]
            names = ["latent_sample", "encoder_out", "decoder_out"]
            for i in range(len(vars_)):
                with tf.variable_scope(names[i] + "_{}".format(t)):
                    v = vars_[i]
                    self.variable_summary(v)

            """ Storing and updating values """
            self.z_seq[t] = z
            self.dec_state_seq[t] = dec_state
            h_dec_prev = h_dec
            c_prev = self.canvas_seq[t]
            self.DO_SHARE = True

        if self.train_classifier:
            z_stacked = tf.stack(self.z_seq)
            z_stacked = tf.transpose(self.z_seq, perm=[1, 0, 2])
            Z = tf.reshape(z_stacked, (-1, self.T * self.latent_dim))

            with tf.variable_scope("logreg"):
                tmp = self.linear(Z, self.Y_c.shape[1])

            self.logits = tf.nn.softmax(tmp)


    def _ModelLoss(self, reconst_loss=None, scale_kl=False):
        """
        Parameters
        ----------

        reconst_loss : function to compute reconstruction loss. Must take two arguments target and output
        and return object of shape (batch_size, 1)

        Computes the losses in reconstruction the target image and the KL loss in the latent expression wrt.
        the target normal distribution.
        """

        if reconst_loss is None:
            x_recons = self.canvas_seq[-1]
            reconst_loss = self.binary_crossentropy
            self.Lx = tf.reduce_mean(tf.reduce_sum(reconst_loss(self.x, x_recons), 1))
        if reconst_loss == "mse":
            x_recons = self.canvas_seq[-1]
            self.Lx = tf.losses.mean_squared_error(self.x, x_recons)

        # x_recons = tf.clip_by_value(self.canvas_seq[-1], 0, 1)

        # Lx = tf.losses.mean_squared_error(x, x_recons)  # tf.reduce_mean(Lx)
        # Lx = tf.losses.mean_pairwise_squared_error(x, x_recons)
        self.scale_kl = scale_kl

        if self.include_KL:
            KL_loss = [0] * self.T

            for t in range(self.T):
                mu_sq = tf.square(self.mus[t])
                sigma_sq = tf.square(self.sigmas[t])
                logsigma_sq = tf.square(self.logsigmas[t])
                KL_loss[t] = tf.reduce_sum(mu_sq + sigma_sq - 2 * logsigma_sq, 1)

            KL = self.beta * (0.5 * tf.add_n(KL_loss) - self.T / 2)
            if scale_kl:
                self.kl_scale = tf.placeholder(dtype=tf.float32, shape=(1,))
                self.Lz = tf.reduce_mean(KL)
                self.Lz *= self.kl_scale
            else:
                self.Lz = tf.reduce_mean(KL)
        elif self.include_KM:
            self.p = tf.placeholder(tf.float32, (None, self.n_clusters))
            if self.include_MMD:
                mmd = self.compute_mmd(self.p, self.q)
                self.Lz = self.beta * mmd
            else:
                self.Lz = tf.keras.metrics.kullback_leibler_divergence(self.p, self.q)
                self.Lz = self.beta * tf.reduce_mean(self.Lz)

            if self.pretrain_simulated:
                self.y_batch = tf.placeholder(tf.float32)
                self.clf_acc = tf.metrics.accuracy(
                    tf.math.argmax(self.y_batch, axis=-1),
                    tf.math.argmax(self.q, axis=-1),
                )
                self.classifier_cost = tf.losses.softmax_cross_entropy(
                    self.y_batch, self.q
                )
        elif self.include_MMD:
            n = self.batch_size
            #norm1 = tf.distributions.Normal(0.0, 1.0)
            #norm2 = tf.distributions.Normal(6.0, 1.0)
            #binom = tf.distributions.Multinomial(1.0, probs=[0.5, 0.5])
            self.Lz = 0

            """
            size  = self.T * self.latent_dim
            all_z = tf.transpose(tf.stack(self.z_seq), (1, 0, 2))
            all_z = tf.reshape(all_z, (n, size))

            y1 = norm1.sample((n, size))
            y2 = norm2.sample((n, size))
            y3 = norm3.sample((n, size))

            w = binom.sample((size))
            ref = w[:,0]*y1 + w[:,1]*y2 + w[:,2]*y3
            #print("FUCKO", w.get_shape(), y1.get_shape())
            self.Lz = self.compute_mmd(ref, all_z)

            """
            for t in range(self.T):
                z = self.z_seq[t]
                #y1 = norm1.sample((n, self.latent_dim))
                #y2 = norm2.sample((n, self.latent_dim))
                # y3 = norm3.sample((n, self.latent_dim))
                #w = binom.sample((n, self.latent_dim))
                #ref = w[:, :, 0] * y1 + w[:, :, 1] * y2  # + w[:, :, 2]*y3
                ref = tf.random.normal(tf.stack([self.batch_size, self.latent_dim]))
                mmd = self.compute_mmd(ref, z)
                self.Lz += mmd
                #self.Lz += tf.losses.mean_squared_error(tf.reduce_mean(z), 3)
            self.Lz = self.beta * self.Lz  # - self.T/2

        else:
            self.Lz = tf.constant(0, dtype=tf.float32) * self.Lx

        cost = self.Lz + self.Lx
        cost += tf.losses.get_regularization_loss()
        self.cost = cost
        tf.summary.scalar("Lx", self.Lx)
        tf.summary.scalar("Lz", self.Lz)
        tf.summary.scalar("cost", self.cost)

        if self.train_classifier:
            self.y_batch = tf.placeholder(tf.float32, shape=(None, self.Y_c.shape[1]))
            self.classifier_cost = tf.reduce_mean(
                tf.reduce_sum(self.binary_crossentropy(self.y_batch, self.logits))
            )


    def compute_latent_sample(self, t, h_enc,):
        if self.include_KL:
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sample(h_enc)
        if self.include_KM:
            with tf.variable_scope("sample"):
                z, latent_state = self.latent_cell(h_enc, latent_state)
                if t == (self.T - 1):
                    self.clusters = tf.get_variable(
                        "clusters",
                        shape=(self.n_clusters, self.latent_dim),
                        initializer=tf.initializers.random_uniform(),
                    )
                    self.q = self.clustering_layer(z)
        else:
            with tf.variable_scope("sample", reuse=self.DO_SHARE):
                z = self.linear(
                        h_enc,
                        self.latent_dim,
                        lmbd=self.kernel_reg_strength
                        )
        return z

    def encode(self, state, input):
        with tf.variable_scope("encoder", reuse=self.DO_SHARE):
            return self.encoder(input, state)

    def decode(self, state, input):
        with tf.variable_scope("decoder", reuse=self.DO_SHARE):
            return self.decoder(input, state)

    def sample(self, h_enc):
        """
        Parameters
        ----------

        h_enc : output from the encoder LSTM 

        samples z_t from a parametrized NormalDistribution(mu, sigma)
        the parametrization is trained to approach a normal via a KL loss
        """

        e = tf.random_normal((self.batch_size, self.latent_dim), mean=0, stddev=1)
        with tf.variable_scope("mu", reuse=self.DO_SHARE):
            mu = self.linear(
                    h_enc,
                    self.latent_dim,
                    lmbd=self.kernel_reg_strength
                    )

        with tf.variable_scope("sigma", reuse=self.DO_SHARE):
            sigma = self.linear(
                h_enc,
                self.latent_dim,
                lmbd=self.kernel_reg_strength,
                regularizer=tf.contrib.layers.l2_regularizer,
            )
            sigma = tf.nn.relu(sigma)

            sigma = tf.clip_by_value(sigma, 1, 100)
            logsigma = tf.log(sigma)

        return (mu + sigma * e, mu, logsigma, sigma)

    def read_no_attn(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)

    def write_no_attn(self, h_dec):
        with tf.variable_scope("write", reuse=self.DO_SHARE):
            return self.linear(h_dec, self.n_input)

    def read_conv(self, x, x_hat, h_dec_prev):
        with tf.variable_scope("gamma", reuse=self.DO_SHARE):
            gamma = self.linear(h_dec_prev, 1, lmbd=self.kernel_reg_strength)
            self.variable_summary(gamma)
        x = gamma * x
        x = tf.reshape(x, (tf.shape(x)[0], self.H, self.W, 1))
        x_hat = tf.reshape(x_hat, (tf.shape(x)[0], self.H, self.W, 1))
        out = tf.concat((x, x_hat), axis=3)
        if self.use_vgg:
            with tf.variable_scope("read", reuse=self.DO_SHARE):
                out = tf.concat((out, tf.zeros_like(x)), axis=3)
                out = tf.keras.layers.Conv2D(3, 2, activation="relu")(out)

                vgg = VGG16(include_top=False, weights="imagenet", input_tensor=out)
                out = vgg.layers[-1].output

                vgg_shape = out.get_shape()
                flat_shape = vgg_shape[1] * vgg_shape[2] * vgg_shape[3]

                out = tf.reshape(out, (tf.shape(x)[0], flat_shape))
                with tf.variable_scope("out", reuse=self.DO_SHARE):
                    out = self.linear(out, 100, lmbd=self.kernel_reg_strength)
                return out

        else:
            with tf.variable_scope("read", reuse=self.DO_SHARE):
                pow_2 = self.H % 2 ** self.conv_architecture["n_layers"] != 0
                for i in range(self.conv_architecture["n_layers"]):

                    filters = self.conv_architecture["filters"][i]
                    kernel_size = self.conv_architecture["kernel_size"][i]
                    strides = self.conv_architecture["strides"][i]
                    pool = self.conv_architecture["pool"][i]
                    if i == 0 and pow_2:
                        padding = "valid"
                    else:
                        padding = "same"

                    out = tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=(kernel_size, kernel_size),
                        strides=(strides, strides),
                        padding=padding,
                        use_bias=True,
                        name="conv_{}".format(i),
                        # kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_strength),
                        # bias_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_strength),
                    )(out)

                    if pool:
                        out = tf.keras.layers.MaxPool2D(2)(out)

                    if not self.DO_SHARE:
                        print("conv shape: ", out.get_shape())

                    if self.conv_architecture["activation"][i]:
                        _a = self.conv_architecture["activation_func"]
                        if self.batchnorm:
                            if _a == "relu" or _a == "lrelu":
                                out = self.activation(out)
                                out = BatchNormalization(
                                    axis=-1, center=True, scale=True, epsilon=1e-4
                                )(out)
                            else:
                                out = BatchNormalization(
                                    axis=-1, center=True, scale=True, epsilon=1e-4
                                )(out)
                                out = self.activation(out)
                        else:
                            out = self.activation(out)

                self.out_shape = out.get_shape()
                self.flat_shape = (
                    self.out_shape[1] * self.out_shape[2] * self.out_shape[3]
                )

                if not self.DO_SHARE:
                    print("Final shape: ", self.flat_shape)
                with tf.variable_scope("conv_out"):
                    self.variable_summary(out)

                out = tf.reshape(out, (tf.shape(x)[0], self.flat_shape))
                return out

    def write_conv(self, h_dec):
        if h_dec.get_shape()[1] > (self.H * self.W):
            raise ValueError(
                "decoder size is not implemented to handle larger \
                    sizes than H*W"
            )

        out_size = 0
        with tf.variable_scope("write", reuse=self.DO_SHARE):
            out = self.linear(h_dec, self.flat_shape, lmbd=self.kernel_reg_strength)
            start_shape = (self.out_shape[1], self.out_shape[2], self.out_shape[3])
            out = tf.keras.layers.Reshape(start_shape)(out)
            pow_2 = self.H % 2 ** self.conv_architecture["n_layers"] != 0

            for i in reversed(range(self.conv_architecture["n_layers"])):
                """
                pad = self.conv_architecture["padding"][i]
                if pad != 0:
                    out = tf.keras.layers.ZeroPadding2D((pad, pad))(out)
                """
                kernel_size = self.conv_architecture["kernel_size"][i]
                strides = self.conv_architecture["strides"][i]
                pool = self.conv_architecture["pool"][i]

                if i == self.conv_architecture["n_layers"] - 1 and pow_2:
                    padding = "valid"
                else:
                    padding = "same"
                if i == 0:
                    filters = 1
                else:
                    filters = self.conv_architecture["filters"][i - 1]
                out = tf.keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(strides, strides),
                    padding=padding,
                    use_bias=True,
                    name="convT_{}".format(i),
                    # kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_strength),
                    # bias_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_strength),
                )(out)

                if pool:
                    out = tf.keras.layers.UpSampling2D(size=(2, 2))(out)
                if not self.DO_SHARE:
                    print("Deconv shape: ", out.get_shape())
                if self.conv_architecture["activation"][i]:
                    _a = self.conv_architecture["activation_func"]
                    if self.batchnorm:
                        if _a == "relu" or _a == "lrelu":
                            out = self.activation(out)
                            out = BatchNormalization(
                                axis=-1, center=True, scale=True, epsilon=1e-4
                            )(out)
                        else:
                            out = BatchNormalization(
                                axis=-1, center=True, scale=True, epsilon=1e-4
                            )(out)
                            out = self.activation(out)
                    else:
                        out = self.activation(out)

        out = tf.keras.layers.Reshape((self.n_input,))(out)
        return out

    def attn_params(self, scope, h_dec, N):
        with tf.variable_scope(scope, reuse=self.DO_SHARE):
            tmp = self.linear(
                    h_dec,
                    4,
                    regularizer=tf.contrib.layers.l2_regularizer,
                    lmbd=self.kernel_reg_strength
                    )
            gx, gy, logsigma_sq, loggamma = tf.split(tmp, 4, 1)

        sigma_sq = tf.exp(logsigma_sq)
        gamma = tf.exp(loggamma)

        if scope == "write":
            delta = self.delta_w  # tf.exp(logdelta)
        else:
            delta = self.delta_r


        gx = (self.H + 1) / 2 * (gx + 1)
        gy = (self.W + 1) / 2 * (gy + 1)
        # delta = (H - 1)/(N - 1) * delta

        return gx, gy, sigma_sq, delta, gamma

    def filters(self, gx, gy, sigma_sq, delta, gamma, N):
        i = tf.convert_to_tensor(np.arange(N, dtype=np.float32))

        mu_x = gx + (i - N / 2 - 0.5) * delta  # batch_size, N
        mu_y = gy + (i - N / 2 - 0.5) * delta
        # print(mu_x.get_shape(), gx.get_shape(), i.get_shape())
        a = tf.convert_to_tensor(np.arange(self.H, dtype=np.float32))
        b = tf.convert_to_tensor(np.arange(self.W, dtype=np.float32))

        A, MU_X = tf.meshgrid(a, mu_x)  # batch_size, N * H
        B, MU_Y = tf.meshgrid(b, mu_y)

        A = tf.reshape(A, [-1, N, self.H])
        B = tf.reshape(B, [-1, N, self.W])

        MU_X = tf.reshape(MU_X, [-1, N, self.H])
        MU_Y = tf.reshape(MU_Y, [-1, N, self.W])

        sigma_sq = tf.reshape(sigma_sq, [-1, 1, 1])

        Fx = tf.exp(-tf.square(A - MU_X) / (2 * sigma_sq))
        Fy = tf.exp(-tf.square(B - MU_Y) / (2 * sigma_sq))

        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keepdims=True), self.eps)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keepdims=True), self.eps)

        return Fx, Fy

    def read_attn(self, x, xhat, h_dec_prev, Fx, Fy, gamma, N):
        Fx_t = tf.transpose(Fx, perm=[0, 2, 1])

        x = tf.reshape(x, [-1, self.H, self.W])
        xhat = tf.reshape(xhat, [-1, self.H, self.W])

        FyxFx_t = tf.reshape(tf.matmul(Fy, tf.matmul(x, Fx_t)), [-1, N * N])
        FyxhatFx_t = tf.reshape(tf.matmul(Fy, tf.matmul(x, Fx_t)), [-1, N * N])

        return gamma * tf.concat([FyxFx_t, FyxhatFx_t], 1)

    def write_attn(self, h_dec, Fx, Fy, gamma):

        with tf.variable_scope("writeW", reuse=self.DO_SHARE):
            w = self.linear(
                h_dec,
                self.write_N_sq,
                tf.contrib.layers.l2_regularizer,
                lmbd=self.kernel_reg_strength,
            )

        w = tf.reshape(w, [-1, self.write_N, self.write_N])
        Fy_t = tf.transpose(Fy, perm=[0, 2, 1])

        tmp = tf.matmul(w, Fx)
        tmp = tf.reshape(tf.matmul(Fy_t, tmp), [-1, self.H * self.H])

        return tmp / tf.maximum(gamma, self.eps)

    def read_a(self, x, xhat, h_dec_prev):
        params = self.attn_params("read", h_dec_prev, self.read_N)
        Fx, Fy = self.filters(*params, self.read_N)
        return self.read_attn(x, xhat, h_dec_prev, Fx, Fy, params[-1], self.read_N)

    def write_a(self, h_dec):
        params_m = self.attn_params("write", h_dec, self.write_N)
        Fx_m, Fy_m = self.filters(*params_m, self.write_N)
        return self.write_attn(h_dec, Fx_m, Fy_m, params_m[-1])


if __name__ == "__main__":

    T = 9
    enc_size = 10
    dec_size = 16
    latent_dim = 8

    batch_size = 12
    train_data = np.zeros((5600, 128, 128, 1))
    test_data = np.zeros((1100, 128, 128, 1))

    delta_write = 10
    delta_read = 10

    mode_config = {
        "simulated_mdoe": False,
        "restore_mode": False,
        "include_KL": False,
        "include_MMD": True,
    }

    attn_config = {
        "read_N": 10,
        "write_N": 10,
        "write_N_sq": 10 ** 2,
        "delta_w": delta_write,
        "delta_r": delta_read,
    }

    draw_model = DRAW(
        T,
        dec_size,
        enc_size,
        latent_dim,
        train_data,
        use_conv=True,
        mode_config=mode_config,
    )

    graph_kwds = {"initializer": tf.initializers.glorot_normal}

    loss_kwds = {"reconst_loss": None}

    draw_model.compile_model(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-1]
    opt_kwds = {"beta1": 0.5}

    draw_model.compute_gradients(opt, opt_args, opt_kwds)

    sess = tf.InteractiveSession()

    epochs = 2
    data_dir = "../data"
    model_dir = "../models"

    draw_model.train(sess, epochs, data_dir, model_dir, 200)

    draw_model.generateLatent(sess, "../drawing", (train_data, test_data))

    draw_model.generateSamples("../drawing", "../drawing")

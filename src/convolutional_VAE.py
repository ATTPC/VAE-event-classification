import tensorflow as tf
from keras import backend as K
import keras as ker
import numpy as np

import keras.regularizers as reg
import keras.optimizers as opt

from model import LatentModel

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Lambda, ZeroPadding2D


class ConVae(LatentModel):
    """
    A class implementing a convolutional variational autoencoder. 
    """

    def __init__(
            self,
            n_layers,
            filter_arcitecture,
            kernel_architecture,
            strides_architecture,
            latent_dim,
            X,
            beta=1,
            mode_config=None,
            sampling_dim=100,

            train_classifier=False,
            ):

        tf.reset_default_graph()

        super().__init__(X, latent_dim, beta, mode_config)

        self.n_layers = n_layers

        self.filter_arcitecture = filter_arcitecture
        self.kernel_architecture = kernel_architecture
        self.strides_architecture = strides_architecture

        self.latent_dim = latent_dim
        self.sampling_dim = sampling_dim

        self.train_classifier = train_classifier

    def _ModelGraph(
            self,
            kernel_reg=reg.l2,
            kernel_reg_strength=0.01,
            bias_reg=reg.l2,
            bias_reg_strenght=0.01,
            activation="relu",
            ):
        """
        Parameters
        ----------

        Compiles the model graph with the parameters specified from the model
        initialization 
        """

        self.x = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.x_img = tf.reshape(self.x, (tf.shape(self.x)[0], self.H, self.W, self.ch))
        h1 = self.x_img
        shape = K.int_shape(h1)

        k_reg = kernel_reg(kernel_reg_strength)
        b_reg = bias_reg(bias_reg_strenght)

        for i in range(self.n_layers):
            filters = self.filter_arcitecture[i]
            kernel_size = self.kernel_architecture[i]
            strides = self.strides_architecture[i]

            h1 = Conv2D(
                    filters,
                    kernel_size,
                    activation=activation,
                    strides=strides,
                    padding="valid",
                    use_bias=True,
                    kernel_regularizer=k_reg,
                    bias_regularizer=b_reg
                    )(h1)

        shape = K.int_shape(h1)
        h1 = tf.reshape(h1, (tf.shape(self.x)[0], shape[1]*shape[2]*shape[3]))

        h1 = Dense(self.sampling_dim, activation='relu')(h1)

        if self.include_KL:
            self.mean = Dense(self.latent_dim)(h1)
            self.var = Dense(self.latent_dim, activation="relu")(h1)

            sample = Lambda(
                        self.sampling,
                        output_shape=(self.latent_dim,),
                        name="sampler")([self.mean, self.var]
                        )
        else:
            sample = Dense(self.latent_dim)(h1)

        self.z_seq = [sample,]
        self.dec_state_seq = []
        #self.encoder = Model(in_layer, [self.mean, self.var, sample], name="encoder")

        # %%
        de1 = Dense(
                shape[1] * shape[2] * shape[3],
                activation='relu')(sample)

        de1 = tf.reshape(de1, (tf.shape(self.x)[0], shape[1], shape[2], shape[3]))

        for i in reversed(range(self.n_layers)):
            filters = self.filter_arcitecture[i]
            kernel_size = self.kernel_architecture[i]
            strides = self.strides_architecture[i]

            de1 = Conv2DTranspose(
                                filters=filters,
                                kernel_size=kernel_size,
                                activation=activation,
                                strides=strides,
                                padding="valid",
                                use_bias=True,
                                kernel_regularizer=k_reg,
                                bias_regularizer=b_reg,
                                )(de1)


        decoder_out = Conv2DTranspose(
                                filters=1,
                                kernel_size=2,
                                activation='sigmoid',
                                padding='same',
                                use_bias=True,
                                kernel_regularizer=k_reg,
                                bias_regularizer=b_reg,
                                name='decoder_output')(de1)

        decoder_out = ZeroPadding2D(padding=((1, 0), (1, 0)))(decoder_out)
        print("LOOK HERE FUCKO", decoder_out.get_shape())

        decoder_out = tf.reshape(decoder_out, (tf.shape(self.x)[0], self.n_input), )

        self.output = decoder_out

        #self.decoder = Model(latent_inputs, decoder_out, name="decoder")

        #outputs = self.decoder(self.encoder(in_layer)[2])

        #self.vae = Model(in_layer, outputs, name='vae')

        #self.encoder.summary()
        #self.decoder.summary()

    def _ModelLoss(self, reconst_loss=None, scale_kl=False):
        x_recons = self.output
        print(x_recons.get_shape())

        if reconst_loss==None:
            reconst_loss = self.binary_crossentropy

        self.Lx = tf.reduce_mean(tf.reduce_sum(
                                    reconst_loss(self.x, x_recons), 1))

        if self.include_KL:
            mu_sq = tf.square(self.mean)
            sigma_sq = tf.square(self.var)
            logsigma_sq = tf.square(tf.log(self.var))
            KL_loss = tf.reduce_sum(mu_sq + sigma_sq - 2*logsigma_sq, 1)

            KL = self.beta * 0.5 * KL_loss
            self.Lz = KL

        self.cost = self.Lx + self.Lz
        self.scale_kl = scale_kl

        
    def kl_loss(self, args):
        mean, var = args
        kl_loss = - 0.5 * K.sum(1 + self.var - K.square(self.mean) - K.exp(self.var), axis=-1)
        return tf.reduce_mean(kl_loss, keepdims=True)

    def sampling(self, args):
        """
        Reparameterization trick by sampling fr an isotropic unit Gaussian.
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



if __name__ == "__main__":

    latent_dim = 8

    n_layers = 4 
    filter_architecture = [20, 40, 10, 5]
    kernel_architecture = [7, 5, 3, 2]
    strides_architecture = [1, 2, 1, 1]

    batch_size = 12
    train_data = np.zeros((5600, 128, 128, 1))
    test_data = np.zeros((1100, 128, 128, 1))

    mode_config = {
        "simulated_mdoe": False,
        "restore_mode": False,
        "include_KL": True,
        #"include_MMD": True,
    }

    vae_model = ConVae(
        n_layers, 
        filter_architecture,
        kernel_architecture,
        strides_architecture,
        latent_dim,
        train_data,
        beta=1,
        mode_config=mode_config,
    )

    graph_kwds = {
        #"initializer": tf.initializers.glorot_normal
    }

    loss_kwds = {
        "reconst_loss": None
    }

    vae_model.compile_model(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-1, ]
    opt_kwds = {
        "beta1": 0.5,
    }

    vae_model.compute_gradients(opt, opt_args, opt_kwds)

    sess = tf.InteractiveSession()

    epochs = 2
    data_dir = "../data"
    model_dir = "../models"

    vae_model.train(sess, epochs, data_dir, model_dir, 200)

    vae_model.generateLatent(sess, "../drawing", (train_data, test_data))

    vae_model.generateSamples("../drawing", "../drawing")


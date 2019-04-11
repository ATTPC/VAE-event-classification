import tensorflow as tf
from keras import backend as K
import keras as ker

import keras.regularizers as reg
import keras.optimizers as opt

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Reshape
from keras.layers import Conv2DTranspose
from keras.layers import ZeroPadding2D, ZeroPadding3D
from keras.layers import Input, Lambda


class ConVae:
    """
    A class implementing a convolutional variational autoencoder. 
    """

    def __init__(
            self,
            input_dimensions, 
            n_layers,
            filter_arcitecture,
            kernel_architecture,
            strides_architecture,
            latent_dim,
            batch_size,
            X,
            sampling_dim=100,
            ):

        tf.reset_default_graph()

        self.X = X
        self.input_dimensions = list(input_dimensions)
        self.n_layers = n_layers

        self.filter_arcitecture = filter_arcitecture
        self.kernel_architecture = kernel_architecture
        self.strides_architecture = strides_architecture

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.sampling_dim = sampling_dim

    def train(
            self,
            epochs=100,
            save_dir=None,
            callbacks=[]
            ):

        n_samples = self.X.shape[0]
        input_size = self.input_dimensions[0]*self.input_dimensions[1]

        flat_X = self.X.reshape((n_samples, input_size))

        self.vae.fit(
                self.X,
                flat_X,
                epochs=epochs,
                batch_size=self.batch_size,
                callbacks=callbacks
            )

        self.vae.save_weights("../models/conVAE/attpc_vae.h5")
        self.encoder.save_weights("../models/conVAE/attpc_enc.h5")
        self.decoder.save_weights("../models/conVAE/attpc_dec.h5")

    def CompileModel(
            self,
            compile_kwargs={},
            ):
        """
        """
        self._ModelGraph(**compile_kwargs)
        self.compiled = True

    def CompileLoss(self, optimizer="adam"):
        if self.compiled:

            self.vae.add_loss(self.kl_loss)
            self.vae.compile(
                    optimizer=optimizer, 
                    loss="binary_crossentropy",
                    )

            return 0
        else:
            return 1

    def _ModelGraph(
            self,
            kernel_reg=reg.l2,
            kernel_reg_strength=0.01,
            bias_reg = reg.l2,
            bias_reg_strenght=0.01,
            activation="relu",
            ):
        """
        Parameters
        ----------

        Compiles the model graph with the parameters specified from the model
        initialization 
        """

        in_layer = Input(self.input_dimensions)

        h1 = in_layer
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
        h1 = Flatten()(h1)

        h1 = Dense(self.sampling_dim, activation='relu')(h1)
        self.mean = Dense(self.latent_dim)(h1)
        self.var = Dense(self.latent_dim)(h1)


        sample = Lambda(self.sampling, output_shape=(self.latent_dim,), name="sampler")([self.mean, self.var])

        self.encoder = Model(in_layer, [self.mean, self.var, sample], name="encoder")

        # %%
        latent_inputs = Input((self.latent_dim, ), name='z_sampling')
        de1 = Dense(
                shape[1] * shape[2] * shape[3],
                activation='relu')(latent_inputs)

        de1 = Reshape((shape[1], shape[2], shape[3]))(de1)

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

        de1 = ZeroPadding2D(padding=((1, 0), (1, 0)))(de1)

        decoder_out = Conv2DTranspose(filters=1,
                                kernel_size=2,
                                activation='sigmoid',
                                padding='same',
                                use_bias=True,
                                kernel_regularizer=k_reg,
                                bias_regularizer=b_reg,
                                name='decoder_output')(de1)

        decoder_out = Flatten()(decoder_out)

        self.decoder = Model(latent_inputs, decoder_out, name="decoder")

        outputs = self.decoder(self.encoder(in_layer)[2])

        self.kl_loss = Lambda(self.klLoss, output_shape=(1,), name="kl-loss")([self.mean, self.var])
        self.vae = Model(in_layer, outputs, name='vae')

        self.encoder.summary()
        self.decoder.summary()



    def klLoss(self, args):
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

    def restore(self, path):
        
        self.encoder.load_weights(path+"attpc_enc.h5")
        print("loaded enc")
        self.decoder.load_weights(path+"attpc_dec.h5")
        print("loaded dec")
        self.vae.load_weights(path+"attpc_vae.h5")
        print("loaded vae")

        self.restored = True

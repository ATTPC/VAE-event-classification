import tensorflow as tf
from keras import backend as K
import keras as ker
import numpy as np

import keras.regularizers as reg
import keras.optimizers as opt

from model import LatentModel

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Lambda, ZeroPadding2D
from tensorflow.keras.layers import ReLU, LeakyReLU, BatchNormalization


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
            pooling_architecture,
            latent_dim,
            X,
            beta=1,
            mode_config=None,
            clustering_config=None,
            sampling_dim=100,
            labelled_data=None,
            train_classifier=False,
            ):

        tf.reset_default_graph()

        self.simulated_mode = False
        self.restore_mode = False
        self.include_KL = False
        self.include_MMD = False
        self.include_KM = False
        self.pretrain_simulated = False
        self.batchnorm = False

        super().__init__(X, latent_dim, beta, mode_config)
        self.use_attention = False
        self.T = 1

        self.labelled_data = labelled_data

        self.n_layers = n_layers

        self.filter_arcitecture = filter_arcitecture
        self.kernel_architecture = kernel_architecture
        self.strides_architecture = strides_architecture
        self.pooling_architecture = pooling_architecture

        if self.include_KM and clustering_config == None:
            raise RuntimeError("when KM is true a config must be supplied")
        elif self.include_KM and clustering_config != None:
            for key, val in clustering_config.items():
                setattr(self, key, val)

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
            output_activation="sigmoid",
            ):
        """
        Parameters
        ----------

        Compiles the model graph with the parameters specified from the model
        initialization 
        """
        activations = {
                "relu": ReLU,
                "lrelu": LeakyReLU,
                "tanh": tf.tanh,
                "sigmoid": tf.sigmoid,
                }

        keras_activations = {
                "relu": ker.activations.relu,
                "lrelu": ker.layers.advanced_activations.LeakyReLU,
                "tanh": ker.activations.tanh,
                "sigmoid": ker.activations.sigmoid,
                }

        self.x = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.batch_size = tf.shape(self.x)[0]
        self.x_img = tf.reshape(self.x, (self.batch_size, self.H, self.W, self.ch))
        h1 = self.x_img
        shape = K.int_shape(h1)

        k_reg = kernel_reg(kernel_reg_strength)
        b_reg = bias_reg(bias_reg_strenght)

        for i in range(self.n_layers):
            with tf.name_scope("conv_"+str(i)):
                filters = self.filter_arcitecture[i]
                kernel_size = self.kernel_architecture[i]
                strides = self.strides_architecture[i]
                h1 = Conv2D(
                        filters,
                        (kernel_size, kernel_size),
                        strides=(strides, strides),
                        padding="valid",
                        use_bias=True,
                        kernel_regularizer=k_reg,
                        bias_regularizer=b_reg
                        )(h1)

                if i == 0:
                    continue
                if activation == None:
                   pass 
                elif activation=="relu" or activation=="lrelu":
                    a = activations[activation]
                    if activation == "relu":
                        h1 = a()(h1)
                    else: 
                        h1 = a(0.1)(h1)
                    with tf.name_scope("batch_norm"):
                        if self.batchnorm:
                            h1 = BatchNormalization(
                                    axis=-1,
                                    center=True,
                                    scale=True,
                                    epsilon=1e-4,
                                    )(h1)
                            self.variable_summary(h1)
                else:
                    a = activations[activation]
                    with tf.name_scope("batch_norm"):
                        if self.batchnorm:
                            h1 = BatchNormalization(
                                    axis=-1,
                                    center=True,
                                    scale=True,
                                    epsilon=1e-4,
                                    )(h1)
                            self.variable_summary(h1)
                    h1 = a(h1)

                if self.pooling_architecture[i]:
                    h1 = tf.layers.max_pooling2d(h1, 2, 2)

        shape = K.int_shape(h1)
        h1 = tf.reshape(h1, (self.batch_size, shape[1]*shape[2]*shape[3]))
        h1 = Dense(self.sampling_dim, activation=activation)(h1)

        if self.include_KL:
            self.mean = Dense(self.latent_dim)(h1)
            self.var = Dense(self.latent_dim, activation=activation)(h1)

            sample = Lambda(
                        self.sampling,
                        output_shape=(self.latent_dim,),
                        name="sampler")([self.mean, self.var]
                        )
        else:
            sample = Dense(
                    self.latent_dim,
                    activation=activation,
                    )(h1)

        if self.include_KM:
            self.clusters = tf.get_variable(
                                    "clusters",
                                    shape=(self.n_clusters, self.latent_dim) ,
                                    initializer=tf.initializers.random_uniform()
                                    )
            self.q = self.clustering_layer(sample)

        self.z_seq = [sample,]
        self.dec_state_seq = []
        #self.encoder = Model(in_layer, [self.mean, self.var, sample], name="encoder")

        # %%
        de1 = Dense(
                shape[1] * shape[2] * shape[3],
                activation=activation)(sample) 
        with tf.name_scope("dense"):
            if self.batchnorm:
                de1 = BatchNormalization()(de1)
            self.variable_summary(de1)

        de1 = tf.reshape(de1, (self.batch_size, shape[1], shape[2], shape[3]))

        for i in reversed(range(self.n_layers)):
            with tf.name_scope("t_conv_"+str(i)):
                if i==0:
                    filters = 1
                else:
                    filters = self.filter_arcitecture[i-1]
                kernel_size = self.kernel_architecture[i]
                strides = self.strides_architecture[i]

                if self.pooling_architecture[i]:
                    de1 = ker.layers.UpSampling2D(size=(2,2))(de1) 
                if activation == None:
                    de1 = de1
                elif activation=="relu" or activation=="lrelu":
                    a = keras_activations[activation]
                    if activation == "relu":
                        de1 = a(de1)
                    else: 
                        de1 = a(0.1)(de1)
                    with tf.name_scope("batch_norm"):
                        if self.batchnorm:
                            de1 = ker.layers.BatchNormalization(
                                    axis=-1,
                                    center=True,
                                    scale=True,
                                    epsilon=1e-4,
                                    )(de1)
                            self.variable_summary(de1)
                else:
                    a = keras_activations[activation]
                    with tf.name_scope("batch_norm"):
                        if self.batchnorm:
                            de1 = ker.layers.BatchNormalization(
                                    axis=-1,
                                    center=True,
                                    scale=True,
                                    epsilon=1e-4,
                                    )(de1)
                            self.variable_summary(de1)
                    de1 = a(de1)

                layer = ker.layers.Conv2DTranspose(
                                    filters=filters,
                                    kernel_size=(kernel_size, kernel_size),
                                    strides=(strides, strides),
                                    output_padding=(strides-1, strides-1),
                                    padding="valid",
                                    use_bias=True,
                                    kernel_regularizer=k_reg,
                                    bias_regularizer=b_reg,
                                    )
                de1 = layer(de1)

        #decoder_out = ZeroPadding2D(padding=((1, 0), (1, 0)))(decoder_out)
        with tf.name_scope("final_deconv"):
            with tf.name_scope("batch_norm"):
                if self.batchnorm:
                    de1 = ker.layers.BatchNormalization(
                        axis=-1,
                        center=True,
                        scale=True,
                        epsilon=1e-4,
                        )(de1)
                    self.variable_summary(de1)
            print("output a", output_activation)
            decoder_out = keras_activations[output_activation](de1)
        print("FINAL O", decoder_out.get_shape())
        decoder_out = tf.reshape(decoder_out, (self.batch_size, self.n_input), )

        self.output = decoder_out
        self.canvas_seq = [decoder_out, ]

        #self.decoder = Model(latent_inputs, decoder_out, name="decoder")

        #outputs = self.decoder(self.encoder(in_layer)[2])

        #self.vae = Model(in_layer, outputs, name='vae')

        #self.encoder.summary()
        #self.decoder.summary()

    def _ModelLoss(self, reconst_loss=None, scale_kl=False):
        x_recons = self.output
        if reconst_loss==None:
            reconst_loss = self.binary_crossentropy
            self.Lx = tf.reduce_mean(tf.reduce_sum(
                                        reconst_loss(self.x, x_recons), 1))
        elif reconst_loss=="mse":
            self.Lx = tf.losses.mean_squared_error(self.x, x_recons)

        if self.pretrain_simulated:
            self.y_batch = tf.placeholder(tf.float32)
            self.clf_acc = tf.metrics.accuracy(
                                tf.math.argmax(self.y_batch, axis=-1), 
                                tf.math.argmax(self.q, axis=-1)
                                )
            self.classifier_cost = tf.losses.softmax_cross_entropy(self.y_batch, self.q)

        if self.include_KL:
            mu_sq = tf.square(self.mean)
            sigma_exp = tf.exp(self.var)
            #logsigma_sq = tf.square(tf.log1p(self.var))
            KL_loss = tf.reduce_sum(mu_sq + sigma_exp - self.var - 1, 1)
            KL = self.beta * 0.5 * tf.reduce_sum(KL_loss)
            self.Lz = KL

        if self.include_MMD:
            z = self.z_seq[0]
            n = self.batch_size

            norm1 = tf.distributions.Normal(0., 1.)
            norm2 = tf.distributions.Normal(6., 1.)
            binom = tf.distributions.Multinomial(1., probs=[0.5, 0.5])

            y1 = norm1.sample((n, self.latent_dim))
            y2 = norm2.sample((n, self.latent_dim))
            #y3 = norm3.sample((n, self.latent_dim))

            w = binom.sample((n, self.latent_dim))
            ref = w[:, :, 0]*y1 + w[:, :, 1]*y2 #+ w[:, :, 2]*y3

            #ref = tf.random.normal(tf.stack([self.batch_size, self.latent_dim]))
            mmd = self.compute_mmd(ref, z)
            self.Lz = self.beta*mmd

        if self.include_KM:
            self.p = tf.placeholder(tf.float32, (None, self.n_clusters))
            if self.include_MMD:
                mmd = self.compute_mmd(self.p, self.q)
                self.Lz = self.beta*mmd
            else:
                self.Lz = tf.keras.metrics.kullback_leibler_divergence(self.p, self.q)
                self.Lz = tf.reduce_mean(self.Lz)

        else:
            self.Lz = self.Lx*0

        self.cost = self.beta*self.Lx + (1-self.beta)*self.Lz
        self.scale_kl = scale_kl
        tf.summary.scalar("Lx", self.Lx)
        tf.summary.scalar("Lz", self.Lz)
        tf.summary.scalar("cost", self.cost)

    def kl_loss(self, args):
        """
        kl loss to isotropic zero mean unit variance gaussian
        """
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


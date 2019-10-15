import tensorflow as tf
import numpy as np
from sklearn.metrics import adjusted_rand_score as ars
from convolutional_VAE import ConVae
from batchmanager import BatchManager


class entropy_callback(tf.keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch, logs={}):
        c1, c2, c3 = self.get_losses(logs)
        self.alpha = self.alpha*0.99**epoch
        self.beta = self.beta*1.01**epoch

    def get_losses(self, logs):
        c1 = logs["reconstructions_loss"]
        c2 = logs["soft_prob_loss"]
        c3 = logs["batch_ent_loss"]
        return c1, c2, c3

class probabilities_log(tf.keras.callbacks.Callback):
    def __init__(self, model, nclasses, latent_dim, data):
        self.prob_log = []
        self.latent_log = []
        self.data = data
        self.nclasses = nclasses
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs={}):
        bm = BatchManager(self.data.shape[0], 100, shuffle=False)
        prob_all = np.zeros((self.data.shape[0], self.nclasses))
        latent_all = np.zeros((self.data.shape[0], self.self.latent_dim))
        for batch in bm:
            latent, probs, _ = self.model.predict_on_batch(self.data[batch])
            prob_all[batch] = probs[1]
            latent_all[batch] = latent[1]
        self.prob_log.append(prob_all)
        self.latent_log.append(latent_all)


class mixae_model:
    def __init__(self, ae_args, alpha=0.1, beta=10, rec=10, vae_beta=10):
        self.ae_args = ae_args
        self.input_tensor = None
        self.alpha = tf.keras.backend.variable(alpha, dtype=tf.float32)
        self.beta = tf.keras.backend.variable(beta, dtype=tf.float32)
        self.rec = tf.keras.backend.variable(rec, dtype=tf.float32)
        self.vae_beta = tf.keras.backend.variable(vae_beta, dtype=tf.float32)
        self.epsilon = tf.keras.backend.constant(1e-8, dtype=tf.float32)
        self.training = True

    def compile(self, n_ae,):
        latent_samples = []
        reconstructions = []
        autoencoders = []
        self.n_ae = n_ae
        for i in range(n_ae):
            ae = self.make_autoencoder(i)
            reconstructions.append(ae.output)
            autoencoders.append(ae)
            latent_samples.append(ae.z_seq[0])
        latent_samples = tf.keras.layers.concatenate(latent_samples, axis=-1)
        # make soft predictions
        kernel_reg = self.ae_args[3]["kernel_reg"]
        kernel_reg = kernel_reg(self.ae_args[3]["kernel_reg_strength"])
        dense_sizes = [300, 100]
        for i, d in enumerate(dense_sizes):
            if i == 0:
                p = tf.keras.layers.Dense(
                    d,
                    kernel_regularizer=kernel_reg,
                    activation="relu"
                )(latent_samples)
            else:
                p = tf.keras.layers.Dense(
                    d,
                    kernel_regularizer=kernel_reg,
                    activation="relu"
                )(p)
            if self.training:
                p = tf.keras.layers.Dropout(0.1)(p)
        p = tf.keras.layers.Dense(
            n_ae,
            kernel_regularizer=kernel_reg,
        )(p)
        p = tf.keras.layers.Softmax(name="soft_prob")(p)

        batch_identity = tf.keras.layers.Lambda(tf.identity, name="batch_ent")
        batch_ent = batch_identity(p)
        stack = tf.keras.layers.Lambda(tf.stack, name="reconstructions")
        reconstructions = stack(reconstructions,)

        train_out = [
            reconstructions,
            p,
            batch_ent,
        ]
        opt = tf.keras.optimizers.Adam
        opt_args = {"lr": 1e-3, "clipnorm": 10, "decay": 1e-3/20}
        # mixae trainable model
        to_train_model = tf.keras.models.Model(
            inputs=[self.input_tensor],
            outputs=train_out,
        )
        if self.use_KL:
            mean_stack = tf.keras.layers.Lambda(tf.stack, name="means")
            std_stack = tf.keras.layers.Lambda(tf.stack, name="std_sq")
            means = mean_stack([ae.mean for ae in autoencoders])
            std_sqs = std_stack([ae.var for ae in autoencoders])
            to_train_model.add_loss(
                    self.vae_beta*self.kl_div_loss(means, std_sqs)
                    )

        to_train_model.compile(
            opt(**opt_args),
            loss=[
                mixae_model.reconstruction_loss(p, self.pixel_weight),
                mixae_model.classification_entropy,
                mixae_model.batch_entropy,
                # mixae_model.psuedo_uniformity,
            ],
            loss_weights=[self.rec, self.alpha, self.beta],
        )
        # classifier model for pretrained probs
        pretrain_model = tf.keras.models.Model(
            inputs=[self.input_tensor],
            outputs=[
                reconstructions,
            ]
        )
        pretrain_model.compile(
            opt(**opt_args),
            loss=[
                mixae_model.reconstruction_loss(p, self.pixel_weight),
            ],
            metrics={"soft_prob": "accuracy"}
        )
        # cluster model predicting only probs
        cluster_model = tf.keras.models.Model(
            inputs=[self.input_tensor],
            outputs=[p, ]
        )
        cluster_model.compile(
            opt(**opt_args),
            loss=[tf.keras.losses.categorical_crossentropy],
            metrics=["accuracy"]
        )
        # model with all outputs
        full_model = tf.keras.models.Model(
            inputs=[self.input_tensor],
            outputs=[
                stack(latent_samples),
                p,
                reconstructions,
            ]
        )
        # print(full_model.summary())
        # print(p.output_shape)
        # print(reconstructions.output_shape)
        return to_train_model, pretrain_model, cluster_model, full_model

    def make_autoencoder(self, i):
        positional = self.ae_args[0]
        dictargs = self.ae_args[1]
        graph_positional = self.ae_args[2]
        graph_dicts = self.ae_args[3]
        ae = ConVae(*positional, **dictargs)
        if self.input_tensor is None:
            self.input_tensor = tf.keras.layers.Input(shape=(ae.n_input,))
            self.ninput = ae.n_input
            self.latent_dim = positional[-2]
            self.pixel_weight = tf.keras.backend.constant(
                1/self.ninput, dtype=tf.float32)
            self.use_KL = dictargs["mode_config"]["include_KL"]
        graph_dicts["input_tensor"] = self.input_tensor
        graph_dicts["mixae_n"] = str(i)
        ae._ModelGraph(*graph_positional, **graph_dicts)
        return ae

    def kl_div_loss(self, means, log_sigma_sqs):
        prior_mean_tensors = self.make_means()
        print("prior", prior_mean_tensors.get_shape())
        print("approx", means.get_shape(), log_sigma_sqs.get_shape())
        sigma_sqs = tf.exp(log_sigma_sqs)
        each_ae_loss = 0.5*tf.reduce_sum(
            sigma_sqs
            - 1
            - log_sigma_sqs
            + tf.math.square(means)
            + tf.math.square(prior_mean_tensors)
            - 2*prior_mean_tensors*means,
            axis=-1
        )
        all_ae_loss = tf.reduce_mean(each_ae_loss, axis=-1)
        return tf.reduce_mean(all_ae_loss)

    def make_means(self,):
        means = np.zeros((self.n_ae, self.latent_dim))
        for i in range(self.n_ae):
            if i == 0:
                means[i] = np.random.randint(-10, 10, size=self.latent_dim)
            else:
                dist = False
                while not dist:
                    new = np.random.randint(-10, 10, size=self.latent_dim)
                    close_all = []
                    for mean in means:
                        closeness = np.linalg.norm(new-mean)
                        close_all.append(0 if closeness < 2 else 1)
                    if np.all(close_all):
                        dist = True
                    else:
                        dist = False
        means = tf.keras.backend.constant(means, dtype=tf.float32)
        return tf.expand_dims(means, axis=1)

    @staticmethod
    def reconstruction_loss(layer, weight):
        def square_error(y_true, y_pred):
            print("target", y_true.get_shape())
            print("recons", y_true.get_shape())
            y_true = tf.transpose(y_true, perm=[1, 0, 2, ])
            mul = tf.math.squared_difference(y_true, y_pred)
            print("mse_pre_sum", mul.get_shape())
            return tf.math.reduce_mean(mul, axis=-1)

        def loss(y_true, y_pred):
            mse = square_error(y_true, y_pred)
            print("mse reduced", mse.get_shape())
            print("probs ", layer.get_shape())
            weighted = tf.keras.layers.multiply(
                [
                    layer,
                    tf.transpose(mse, perm=[1, 0])
                ]
            )
            print("weighted", weighted.get_shape())
            loss_val = tf.reduce_sum(weighted, axis=-1)
            return tf.reduce_mean(loss_val)  # *weight
        return loss

    @staticmethod
    def batch_entropy(y_true, y_pred):
        print("batch_ent")
        batch_pred_sum = tf.reduce_mean(y_pred, axis=0)
        log_pred_sum = tf.clip_by_value(batch_pred_sum, 1e-10, 1.0)
        log_pred_sum = tf.math.log(log_pred_sum)
        entropy_contents = tf.keras.layers.multiply(
            [
                batch_pred_sum,
                log_pred_sum
            ]
        )
        entropy_contents = tf.reduce_sum(entropy_contents)
        entropy_contents = entropy_contents - 0.9  # tf.keras.backend.epsilon()
        batch_ent = - tf.math.divide(1, entropy_contents)
        return batch_ent

    @staticmethod
    def psuedo_uniformity(y_true, y_pred):
        batch_mean = tf.reduce_mean(y_pred, axis=0)
        sm_batch_mean = tf.nn.softmax(batch_mean)
        return tf.reduce_sum(tf.math.multiply(batch_mean, sm_batch_mean))

    @staticmethod
    def classification_entropy(y_true, y_pred):
        print("clf ent")
        clipped_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
        log_pred = tf.math.log(clipped_pred)
        entropy = tf.keras.layers.multiply(
            [
                log_pred,
                y_pred,
            ]
        )
        entropy = - tf.reduce_sum(entropy, axis=0)
        return tf.reduce_mean(entropy)

from convolutional_VAE import ConVae
import tensorflow as tf
from sklearn.metrics import adjusted_rand_score as ars
import warnings

class entropy_callback(tf.keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch, logs={}):
        c1, c2, c3 = self.get_losses(logs)
        self.alpha = (c1/c2)*0.98**epoch
        self.beta = (c1/c3)*1.01**epoch

    def get_losses(self, logs):
        c1 = logs["reconstructions_loss"]
        c2 = logs["soft_prob_loss"]
        c3 = logs["batch_ent_loss"]
        return c1, c2, c3

class probabilities_log(tf.keras.callbacks.Callback):
    def __init__(self, model, data):
        self.prob_log = []
        self.model = model
        self.data = data

    def on_epoch_end(self, epoch, logs={}):
        probs = self.model.predict_on_batch(self.data)
        self.prob_log.append(probs)

class mixae_model:
    def __init__(self, ae_args, ):
        self.ae_args = ae_args
        self.input_tensor = None
        self.alpha = tf.keras.backend.variable(0.1, dtype=tf.float32) 
        self.beta = tf.keras.backend.variable(100, dtype=tf.float32)
        self.rec = tf.keras.backend.variable(1, dtype=tf.float32)
        self.epsilon = tf.keras.backend.constant(1e-8, dtype=tf.float32)

    def compile(self, n_ae,):
        latent_samples = []
        reconstructions = []
        autoencoders = []
        for i in range(n_ae):
            ae = self.make_autoencoder()
            reconstructions.append(ae.output)
            autoencoders.append(ae)
            latent_samples.append(ae.z_seq[0])
        latent_samples = tf.keras.layers.concatenate(latent_samples, axis=-1)
        #make soft predictions 
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
        opt_args = {"lr": 1e-3, "clipnorm":10, }
        # mixae trainable model
        to_train_model = tf.keras.models.Model(
                inputs=[self.input_tensor],
                outputs=train_out,
                )
        to_train_model.compile(
                opt(**opt_args),
                loss=[
                    mixae_model.reconstruction_loss(p, self.pixel_weight),
                    mixae_model.classification_entropy,
                    mixae_model.batch_entropy,
                    #mixae_model.psuedo_uniformity,
                    ],
                loss_weights=[self.rec, self.alpha, self.beta],
                )
        # classifier model for pretrained probs
        pretrain_model = tf.keras.models.Model(
                inputs=[self.input_tensor],
                outputs=[
                    reconstructions,
                    p,
                    ]
                )
        pretrain_model.compile(
            opt(**opt_args),
            loss=[
                mixae_model.reconstruction_loss(p, self.pixel_weight),
                tf.keras.losses.categorical_crossentropy,
            ],
            loss_weights=[1, 100],
            metrics={"soft_prob": "accuracy"}
        )
        # cluster model predicting only probs 
        cluster_model = tf.keras.models.Model(
                inputs=[self.input_tensor],
                outputs=[p,]
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
        #print(full_model.summary())
        #print(p.output_shape)
        #print(reconstructions.output_shape)
        return to_train_model, pretrain_model, cluster_model, full_model

    def make_autoencoder(self,):
        positional = self.ae_args[0]
        dictargs = self.ae_args[1]
        graph_positional = self.ae_args[2]
        graph_dicts = self.ae_args[3]
        ae = ConVae(*positional, **dictargs)
        if self.input_tensor is None:
            self.input_tensor = tf.keras.layers.Input(shape=(ae.n_input,))
            self.ninput = ae.n_input
            self.pixel_weight = tf.keras.backend.constant(1/self.ninput, dtype=tf.float32)
        graph_dicts["input_tensor"] = self.input_tensor
        ae._ModelGraph(*graph_positional, **graph_dicts)
        return ae

    @staticmethod
    def reconstruction_loss(layer, weight):
        def square_error(y_true, y_pred):
            print("target", y_true.get_shape())
            print("recons", y_true.get_shape())
            y_true = tf.transpose(y_true, perm=[1, 0, 2,])
            sub = tf.keras.layers.subtract([y_true, y_pred])
            mul = tf.math.square(sub)
            print("mse_pre_sum", mul.get_shape())
            return tf.math.reduce_sum(mul, axis=-1)

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
            return tf.reduce_mean(loss_val)#*weight
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
        entropy_contents = entropy_contents - 0.9# tf.keras.backend.epsilon()
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

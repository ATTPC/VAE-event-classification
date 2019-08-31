from convolutional_VAE import ConVae
import tensorflow as tf

class mixae:
    def __init__(self, ae_args, ):
        self.ae_args = ae_args
        self.input_tensor = None
    def compile(self, n_ae,):
        latent_samples = []
        reconstructions = []
        autoencoders = []
        n_input = self.ae_args[0][-1]
        for i in range(n_ae):
            ae = self.make_autoencoder()
            reconstructions.append(ae.output)
            autoencoders.append(ae)
            latent_samples.append(ae.z_seq[0])
        latent_samples = tf.keras.layers.concatenate(latent_samples)
        #make soft predictions 
        kernel_reg = self.ae_args[3]["kernel_reg"]
        kernel_reg = kernel_reg(self.ae_args[3]["kernel_reg_strength"])
        dense_sizes = [300, 100]
        for i, d in enumerate(dense_sizes):
            if i == 0:
                p = tf.keras.layers.Dense(
                        d,
                        kernel_regularizer=kernel_reg,
                        )(latent_samples)
            else:
                p = tf.keras.layers.Dense(
                        d,
                        kernel_regularizer=kernel_reg,
                        )(p)
        p = tf.keras.layers.Dense(
                n_ae,
                kernel_regularizer=kernel_reg,
                )(p)
        p = tf.keras.layers.Softmax()(p)
        stack = tf.keras.layers.Lambda(tf.stack)
        to_train_model = tf.keras.models.Model(
                inputs=[self.input_tensor],
                outputs=[
                    stack(reconstructions),
                    ]
                )
        full_model = tf.keras.models.Model(
                inputs=[self.input_tensor],
                outputs=[
                    stack(latent_samples),
                    p,
                    stack(reconstructions),
                    ]
                )
        to_train_model.compile(
                "adam",
                loss=mixae.reconstruction_loss(p)
                )
        return to_train_model

    def make_autoencoder(self,):
        positional = self.ae_args[0]
        dictargs = self.ae_args[1]
        graph_positional = self.ae_args[2]
        graph_dicts = self.ae_args[3]
        ae = ConVae(*positional, **dictargs)
        if self.input_tensor is None:
            self.input_tensor = tf.keras.layers.Input(shape=(ae.n_input,))
        graph_dicts["input_tensor"] = self.input_tensor
        ae._ModelGraph(*graph_positional, **graph_dicts)
        return ae

    @staticmethod
    def reconstruction_loss(layer):
        def square_error(y_true, y_pred):
            y_true = tf.transpose(y_true, perm=[1, 0, 2,])
            sub = tf.keras.layers.subtract([y_true, y_pred])
            mul = tf.math.square(sub)
            return tf.math.reduce_sum(mul, axis=-1)

        def loss(y_true, y_pred):
            weighted = tf.keras.layers.multiply(
                    [
                        square_error(y_true, y_pred),
                        tf.transpose(layer, perm=[0, 1])
                    ]
                    )
            loss_val = tf.reduce_sum(weighted, axis=0)
            return tf.reduce_mean(loss_val)
        return loss


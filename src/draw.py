import numpy as np 
import tensorflow as tf
from batchmanager import BatchManager


class DRAW:
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
            batch_size,
            X,
            attn_config=None,
            mode_config=None,
            ):

        tf.reset_default_graph()
        
        self.T = T
        self.dec_size = dec_size 
        self.enc_size = enc_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size 

        self.X = X 
        self.eps = 1e-8

        self.data_shape = self.X.shape

        self.restore_mode = False
        self.simulated_mode = True
        self.use_attention = True

        if len(self.data_shape) == 3 or len(self.data_shape) == 4:
            self.n_input = self.data_shape[1] * self.data_shape[2]
            self.H = self.data_shape[1]
            self.W = self.data_shape[2]
            self.n_data = self.data_shape[0]

        else:
            print("""Wrong input dimensions expected
                    x_train to have DIM == 3 or DIM == 4 got DIM == {}""".format(len(self.data_shape)))

        if self.use_attention and attn_config is None:
            print("""If attention is used then parameters read_N, write_N and corresponding 
                    deltas must be supplied in dict attn_config""")

        elif self.use_attention:
            for key, val in attn_config.items():
                if isinstance(val, (np.ndarray, )):
                    val = tf.convert_to_tensor(val)

                setattr(self, key, val)

        
        if not mode_config is None:
            for key, val in mode_config:
                setattr(self, key, val)

        self.DO_SHARE = None
        self.compiled = False
        self.grad_op = False


    def CompileModel(
            self,
            graph_kwds=None,
            loss_kwds=None,
            ):
        """
        Paramters 
        ---------

        graph_kwds : dictionary with keywords
            initializer - one of tf.initializers uninstantiatiated

        loss_kwds : dictionary with keywords
            loss - one of tf.losses uninstantiated

        Compiles model graph with reconstruction and KL loss
        """
        
        if graph_kwds is None:
            self._ModelGraph()
        else:
            self._ModelGraph(**graph_kwds)

        if loss_kwds is None:
            self._ModeLoss()
        else:
            self._ModelLoss(**loss_kwds)

        self.compiled = True


    def _ModelGraph(
            self,
            initializer=tf.initializers.glorot_normal
            ):

        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_input))

        self.encoder = tf.nn.rnn_cell.LSTMCell(
            self.enc_size,
            state_is_tuple=True,
            activity_regularizer=tf.contrib.layers.l2_regularizer(0.01),
            initializer=initializer, 
        )

        self.decoder = tf.nn.rnn_cell.LSTMCell(
            self.dec_size,
            state_is_tuple=True,
            activity_regularizer=tf.contrib.layers.l2_regularizer(0.01),
            initializer=initializer, 
        )

        self.read = self.read_a if self.use_attention else self.readNoAttn
        self.write = self.write_a if self.use_attention else self.writeNoAttn

        self.canvas_seq = [0]*self.T
        self.z_seq = [0]*self.T
        self.dec_state_seq = [0]*self.T

        self.mus, self.logsigmas, self.sigmas = [0]*self.T, [0]*self.T, [0]*self.T

        # initial states
        h_dec_prev = tf.zeros((self.batch_size, self.dec_size))
        c_prev = tf.zeros((self.batch_size, self.n_input))

        enc_state = self.encoder.zero_state(self.batch_size, tf.float32)
        dec_state = self.decoder.zero_state(self.batch_size, tf.float32)

        # Unrolling the computational graph for the LSTM
        for t in range(self.T):
            # computing the error image
            x_hat = self.x - tf.sigmoid(c_prev)
            r = self.read(self.x, x_hat, h_dec_prev)

            h_enc, enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1))
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sample(h_enc)
            h_dec, dec_state = self.decode(dec_state, z)

            self.canvas_seq[t] = c_prev + self.write(h_dec)
            self.z_seq[t] = z
            self.dec_state_seq[t] = dec_state
            
            h_dec_prev = h_dec
            c_prev = self.canvas_seq[t]
            self.DO_SHARE = True

    def _ModelLoss(self, reconst_loss=None, include_KL=True):

        """
        Parameters
        ----------

        reconst_loss : function to compute reconstruction loss. Must take two arguments target and output
        and return object of shape (batch_size, 1)

        Computes the losses in reconstruction the target image and the KL loss in the latent expression wrt.
        the target normal distribution.
        """

        if reconst_loss is None:
            reconst_loss = self.binary_crossentropy

        x_recons = tf.sigmoid(self.canvas_seq[-1])
        #x_recons = tf.clip_by_value(self.canvas_seq[-1], 0, 1) 

        self.Lx = tf.reduce_mean(tf.reduce_sum(reconst_loss(self.x, x_recons), 1))
        #Lx = tf.losses.mean_squared_error(x, x_recons)  # tf.reduce_mean(Lx)
        #Lx = tf.losses.mean_pairwise_squared_error(x, x_recons)

        KL_loss = [0]*self.T

        for t in range(self.T):
            mu_sq = tf.square(self.mus[t])
            sigma_sq = tf.square(self.sigmas[t])
            logsigma_sq = tf.square(self.logsigmas[t])

            KL_loss[t] = tf.reduce_sum(mu_sq + sigma_sq - 2*logsigma_sq, 1)

        KL = 0.5 * tf.add_n(KL_loss) - self.T/2
        self.Lz = tf.reduce_mean(KL)

        cost = self.Lx + self.Lz if include_KL else self.Lx
        #reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #reg_var = tf.sum(reg_var)
        cost += tf.losses.get_regularization_loss()
        self.cost = cost


    def binary_crossentropy(self, t, o):
        return -(t*tf.log(o+self.eps) + (1.0-t)*tf.log(1.0-o+self.eps))


    def computeGradients(
            self,
            optimizer_class, 
            opt_args=[],
            opt_kwds={},
            ):

        """
        Parameters:
        ----------

        optimizer_class : One of tf.optimizers, uninstantiated

        opt_args : arguments to the optimizer, ordered

        opt_kwds : dictionary of keyword arguments to optimizer

        Computes the losses for the model. Train method is contingent on this being
        called first. 

        Takes an tf.optimizers class and corresponding arguments and kwdargs to be instantiated. 
        """

        if not self.compiled:
            print("please compile model first")
            return 1

        optimizer = optimizer_class(*opt_args, **opt_kwds) 
        grads = optimizer.compute_gradients(self.cost)

        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)

        self.train_op = optimizer.apply_gradients(grads)

        self.fetches = []
        self.fetches.extend([
            self.Lx,
            self.Lz,
            self.z_seq,
            self.dec_state_seq,
            self.train_op
            ])

        self.grad_op = True


    def storeResult(
            self, 
            sess,
            feed_dict,
            data_dir,
            model_dir
            ):

        """
        Paramters 
        ---------

        sess : a tensorflow session used in trainig 

        feed_dict : dictionary with batch of inputs 

        data_dir : directory to save reconstructed samples 

        model_dir : directory to save model checkpoint

        Runs an single batch through the model and saves reconstructions and 
        a model checkpoint
        """

        print()
        print("Saving model and canvasses")

        canvasses = sess.run(self.canvas_seq, feed_dict)
        canvasses = np.array(canvasses)
        references = np.array(feed_dict[self.x])
        
        filename = data_dir+"/simulated/canvasses.npy" if self.simulated_mode else data_dir+"/canvasses.npy"
        np.save(filename, canvasses)

        model_fn = model_dir+"/draw_attn.ckpt" if self.use_attention else model_dir+"/draw_no_attn.ckpt" 
        if self.simulated_mode:
            model_fn = model_dir+"/simulated/draw_attn.ckpt" if self.use_attention else model_dir+"/simulated/draw_no_attn.ckpt"

        self.saver.save(sess, model_fn)

        ref_fn = data_dir+"/simulated/references.npy" if self.simulated_mode else data_dir+"/references.npy"
        np.save(ref_fn, references)


    def train(
            self,
            sess,
            epochs,
            data_dir,
            model_dir,
            earlystopping=True,
            checkpoint_fn=None,
            ):

        """
        Paramters
        ---------


        """
        
        if not (self.compiled and self.grad_op):
            print("cannot train before model is compiled and gradients are computed")
            return

        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()

        train_iters = self.data_shape[0] // self.batch_size
        n_mvavg = 2
        moving_average = [0] * (epochs // n_mvavg)
        best_average = 1e5
        to_average = [0]*n_mvavg
        ta_which = 0
        all_lx = np.zeros(epochs)
        all_lz = np.zeros(epochs)

        for i in range(epochs):
            if self.restore_mode:
                checkpoint_fn = model_dir+"/draw_attn.ckpt" if self.use_attention else model_dir+"/draw_no_attn.ckpt"
                if simulated_mode:
                    checkpoint_fn = model_dir+"/simulated/draw_attn.ckpt" if self.use_attention else model_dir+"/simulated/draw_no_attn.ckpt"

                saver.restore(sess, checkpoint_fn)
                break

            Lxs = [0]*train_iters
            Lzs = [0]*train_iters

            bm_inst = BatchManager(self.n_data, self.batch_size, self.n_input)

            for j in range(train_iters):
                batch = self.X[bm_inst.fetchMinibatch()]
                batch = batch.reshape(self.batch_size, self.n_input)

                feed_dict = {self.x: batch}
                results = sess.run(self.fetches, feed_dict)
                Lxs[j], Lzs[j], _, _, _, = results

        #        with tf.variable_scope("sigma", reuse=DO_SHARE):
        #            w = tf.get_variable("w",)
        #            print(sess.run(w))
        #
            all_lz[i] = tf.reduce_mean(Lzs).eval()
            all_lx[i] = tf.reduce_mean(Lxs).eval()

            print("Epoch {} | Lx = {:5.2f} | Lz = {:5.2f} \r".format(
                    i,
                    all_lx[i],
                    all_lz[i]
                    ),
                    end="",
                    )

            if all_lz[i] < 0:
                print("broken training")
                print("Lx = ", all_lx[i])
                print("Lz = ", all_lz[i])

                break

            if np.isnan(all_lz[i]) or np.isnan(all_lz[i]):
                break
            
            if i >= n_mvavg: 
                to_average[ta_which] = tf.reduce_mean(
                        tf.reduce_sum(all_lx[i - n_mvavg: i] + all_lz[i - n_mvavg: i])).eval()
                ta_which += 1

            if (1 + i) % n_mvavg == 0 and i >= n_mvavg:
                ta_which = 0

                mvavg_index = i // n_mvavg
                moving_average[mvavg_index] = tf.reduce_mean(to_average).eval()

                if earlystopping:
                    do_earlystop = (i // n_mvavg) > 1

                    if moving_average[mvavg_index - 1] < moving_average[mvavg_index] and do_earlystop:
                        print("Earlystopping")

                        return all_lx, all_lz 

                to_average = [0] * n_mvavg
                self.storeResult(sess, feed_dict, data_dir, model_dir)

        return all_lx, all_lz


    def encode(self, state, input):
        with tf.variable_scope("encoder", reuse=self.DO_SHARE):
            return self.encoder(input, state)


    def decode(self, state, input):
        with tf.variable_scope("decoder", reuse=self.DO_SHARE):
            return self.decoder(input, state)


    def linear(
            self,
            x, 
            output_dim,
            regularizer=tf.contrib.layers.l2_regularizer,
            lmbd=0.1
            ):

        w = tf.get_variable("w", [x.get_shape()[1], output_dim],
                            regularizer=regularizer(lmbd),
                            )

        b = tf.get_variable(
            "b",
            [output_dim],
            initializer=tf.constant_initializer(0.0),
            regularizer=regularizer(lmbd))

        return tf.matmul(x, w) + b


    def readNoAttn(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)


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
            mu = self.linear(h_enc, self.latent_dim, lmbd=0.1)

        with tf.variable_scope("sigma", reuse=self.DO_SHARE):
            sigma = self.linear(h_enc, self.latent_dim,
                        lmbd=0.1,
                        regularizer=tf.contrib.layers.l2_regularizer)

            sigma = tf.clip_by_value(sigma, 1, 10)
            logsigma = tf.log(sigma)

        return (mu + sigma*e, mu, logsigma, sigma)


    def writeNoAttn(self, h_dec):
        with tf.variable_scope("write", reuse=self.DO_SHARE):
            return self.linear(h_dec, self.n_inputs)


    def attn_params(self, scope, h_dec, N):
        with tf.variable_scope(scope, reuse=self.DO_SHARE):
            tmp = self.linear(h_dec, 4, regularizer=tf.contrib.layers.l1_regularizer)
            gx, gy, logsigma_sq, loggamma = tf.split(
                tmp, 4, 1)

        sigma_sq = tf.exp(logsigma_sq)

        if scope == "write":
            delta = self.delta_w #tf.exp(logdelta)
        else:
            delta = self.delta_r

        gamma = tf.exp(loggamma)

        gx = (self.H + 1)/2 * (gx + 1)
        gy = (self.W + 1)/2 * (gy + 1)
        #delta = (H - 1)/(N - 1) * delta

        return gx, gy, sigma_sq, delta, gamma


    def filters(self, gx, gy, sigma_sq, delta, gamma, N):
        i = tf.convert_to_tensor(np.arange(N, dtype=np.float32))

        mu_x = gx + (i - N/2 - 0.5) * delta  # batch_size, N
        mu_y = gy + (i - N/2 - 0.5) * delta
        # print(mu_x.get_shape(), gx.get_shape(), i.get_shape())
        a = tf.convert_to_tensor(np.arange(self.H, dtype=np.float32))
        b = tf.convert_to_tensor(np.arange(self.W, dtype=np.float32))

        A, MU_X = tf.meshgrid(a, mu_x)  # batch_size, N * H
        B, MU_Y = tf.meshgrid(b, mu_y)

        A = tf.reshape(A, [self.batch_size, N, self.H])
        B = tf.reshape(B, [self.batch_size, N, self.W])

        MU_X = tf.reshape(MU_X, [self.batch_size, N, self.H])
        MU_Y = tf.reshape(MU_Y, [self.batch_size, N, self.W])

        sigma_sq = tf.reshape(sigma_sq, [self.batch_size, 1, 1])

        Fx = tf.exp(- tf.square(A - MU_X)/(2*sigma_sq))
        Fy = tf.exp(- tf.square(B - MU_Y)/(2*sigma_sq))

        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 1, keepdims=True), self.eps)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 1, keepdims=True), self.eps)

        return Fx, Fy


    def read_attn(self, x, xhat, h_dec_prev, Fx, Fy, gamma, N):
        Fx_t = tf.transpose(Fx, perm=[0, 2, 1])

        x = tf.reshape(x, [self.batch_size, 128, 128])
        xhat = tf.reshape(xhat, [self.batch_size, 128, 128])

        FyxFx_t = tf.reshape(tf.matmul(Fy, tf.matmul(x, Fx_t)), [-1, N*N])
        FyxhatFx_t = tf.reshape(tf.matmul(Fy, tf.matmul(x, Fx_t)), [-1, N*N])

        return gamma * tf.concat([FyxFx_t, FyxhatFx_t], 1)


    def write_attn(self, h_dec, Fx, Fy, gamma):

        with tf.variable_scope("writeW", reuse=self.DO_SHARE):
            w = self.linear(h_dec, self.write_N_sq, tf.contrib.layers.l1_regularizer, lmbd=1e-5)
        
        w = tf.reshape(w, [self.batch_size, self.write_N, self.write_N])
        Fy_t = tf.transpose(Fy, perm=[0, 2, 1])

        tmp = tf.matmul(w, Fx)
        tmp = tf.reshape(tf.matmul(Fy_t, tmp), [-1, self.H*self.H])

        return tmp/tf.maximum(gamma, self.eps)


    def read_a(self, x, xhat, h_dec_prev):
        params = self.attn_params("read", h_dec_prev, self.read_N)
        Fx, Fy = self.filters(*params, self.read_N)
        return self.read_attn(x, xhat, h_dec_prev, Fx, Fy, params[-1], self.read_N)


    def write_a(self, h_dec):
        params_m = self.attn_params("write", h_dec, self.write_N)
        Fx_m, Fy_m = self.filters(*params_m, self.write_N)
        return self.write_attn(h_dec, Fx_m, Fy_m, params_m[-1])

    def generateLatent(self, sess, save_dir, X_tup):
        """
        Parameters
        ----------

        sess : a tf.InteractiveSession instance

        save_dir : directory to save files to
        
        X_tup: tuple of training data and test data, cannot have len 1, cannot have len 1

        Generates latent expressions, decoder states and reconstructions on both the training and test set.
        """

        lat_vals = []
        recons_vals = []
        decoder_states = []

        for i in range(2):

            X = X_tup[i]
            n_latent = (X.shape[0]//self.batch_size)*self.batch_size
            latent_values = np.zeros((self.T, n_latent, self.latent_dim))
            dec_state_array = np.zeros((self.T, 2, n_latent, self.dec_size))
            reconstructions = np.zeros((self.T, n_latent, self.H*self.W))

            for i in range(X.shape[0]//self.batch_size):
                start = i * self.batch_size
                end = (i+1) * self.batch_size
                to_feed = X[start:end].reshape((self.batch_size, self.H*self.W))
                feed_dict = {self.x: to_feed}

                _, _, z_seq, dec_state_seq, _, = sess.run(self.fetches, feed_dict)

                canvasses = sess.run(self.canvas_seq, feed_dict)

                latent_values[:, start:end, :] = z_seq
                reconstructions[:, start:end, :] = np.array(canvasses)
                dec_state_array[:, :, start:end, :] = dec_state_seq
            
            lat_vals.append(latent_values)
            recons_vals.append(reconstructions)
            decoder_states.append(dec_state_array)

        for i in range(2):

            fn = "train_latent.npy" if i == 0 else "test_latent.npy"
            r_fn = "train_reconst.npy" if i == 0 else "test_reconst.npy"
            dec_fn = "train_decoder_states.npy" if i == 0 else "test_decoder_states.npy"

            l = lat_vals[i]
            r = recons_vals[i]
            d = decoder_states[i]
            
            if not self.simulated_mode:
                np.save(save_dir+"/latent/" + fn, l)
                np.save(save_dir+"/" + r_fn, r)
                np.save(save_dir+"/" + dec_fn, d)
            else:
                np.save(save_dir+"/simulated/latent/" + fn, l)
                np.save(save_dir+"/simulated/"+ r_fn, r)
                np.save(save_dir+"/simulated/" + dec_fn, d)


    def generateSamples(self, save_dir, rerun_latent=False, load_dir=None):
        n_samples = self.batch_size
        z_seq = [0]*self.T

        if rerun_latent:
            if load_dir is None:
                print("To reconstruct please pass a directory from which to load samples and states")
                return

            latent_fn = load_dir+"/simulated/latent/train_latent.npy" if simulated_mode else load_dir+"/latent/train_latent.npy"
            dec_state_fn = load_dir+"/simulated/train_decoder_states.npy" if simulated_mode else load_dir+"/train_decoder_states.npy"

            decoder_states = np.load(dec_state_fn)
            latent_samples = np.load(latent_fn)
        
        dec_state = self.decoder.zero_state(self.batch_size, tf.float32)
        h_dec = tf.zeros((self.batch_size, self.dec_size))
        c_prev = tf.zeros((self.batch_size, self.n_input))

        for t in range(self.T):

            if not rerun_latent:
                mu, sigma = (0, 1) if t<1 else (0., 1)
                sample = np.random.normal(mu, sigma, (self.batch_size, self.latent_dim)).astype(np.float32)
            else:
                sample = latent_samples[t, batch_size:2*batch_size, :].reshape((self.batch_size, self.latent_dim)).astype(np.float32)

            z_seq[t] = sample
            z = tf.convert_to_tensor(sample)
            
            if rerun_latent:
                dec_state = decoder_states[t, :, self.batch_size:2*self.batch_size, :].reshape((2, self.batch_size, dec_size)).astype(np.float32)
                dec_state = tf.nn.rnn_cell.LSTMStateTuple(dec_state[0], dec_state[1])

            h_dec, dec_state = self.decode(dec_state, z)
            self.canvas_seq[t] = (c_prev+self.write(h_dec)).eval()

            h_dec_prev = h_dec
            c_prev = self.canvas_seq[t]

        canvasses = np.array(self.canvas_seq)

        if not self.simulated_mode:
            np.save(save_dir+"/generated/samples.npy", canvasses)
            np.save(save_dir+"/generated/latent.npy", np.array(z_seq))
        else:
            np.save(save_dir+"/simulated/generated/samples.npy", canvasses)
            np.save(save_dir+"/simulated/generated/latent.npy", np.array(z_seq))


if __name__ == "__main__":

    T = 9
    enc_size = 10
    dec_size = 11
    latent_dim = 8

    batch_size = 12
    train_data = np.zeros((5600, 128, 128, 1))
    test_data = np.zeros((1100, 128, 128, 1))

    delta_write = 10
    delta_read = 10 
    
    array_delta_w = np.zeros((batch_size, 1))
    array_delta_w.fill(delta_write)
    array_delta_w = array_delta_w.astype(np.float32)

    array_delta_r = np.zeros((batch_size, 1))
    array_delta_r.fill(delta_read)
    array_delta_r = array_delta_r.astype(np.float32)

    attn_config = {
                "read_N": 10,
                "write_N": 10,
                "write_N_sq": 10**2,
                "delta_w": array_delta_w,
                "delta_r": array_delta_r,
            }


    draw_model = DRAW(
            T,
            dec_size,
            enc_size,
            latent_dim,
            batch_size,
            train_data,
            attn_config
            )
    
    graph_kwds = {
            "initializer": tf.initializers.glorot_normal
            }

    loss_kwds = {
            "reconst_loss": None
            }

    draw_model.CompileModel(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-1,]
    opt_kwds = {
            "beta1": 0.5,
            }

    draw_model.computeGradients(opt, opt_args, opt_kwds)
    
    sess = tf.InteractiveSession()

    epochs = 2
    data_dir = "../data"
    model_dir = "../models"

    draw_model.train(sess, epochs, data_dir, model_dir, )

    draw_model.generateLatent(sess, "../drawing", (train_data, test_data))

    draw_model.generateSamples("../drawing", "../drawing")

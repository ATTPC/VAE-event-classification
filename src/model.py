import tensorflow as tf
import signal
import numpy as np
import sys

from keras import backend as K
from sklearn.metrics import accuracy_score, f1_score

from batchmanager import BatchManager


class LatentModel:

    def __init__(
            self,
            X,
            latent_dim,
            beta,
            mode_config,
    ):

        tf.reset_default_graph()
        signal.signal(signal.SIGINT, self.signal_handler)

        self.beta = beta
        self.latent_dim = latent_dim

        self.X = X
        self.eps = 1e-8

        self.data_shape = self.X.shape

        if len(self.data_shape) == 3 or len(self.data_shape) == 4:
            self.H = self.data_shape[1]
            self.W = self.data_shape[2]
            self.ch = self.data_shape[3]
            self.n_input = self.H*self.W*self.ch
            self.n_data = self.data_shape[0]
        else:
            print("""Wrong input dimensions expected
                    x_train to have DIM == 3 or DIM == 4 got DIM == {}""".format(len(self.data_shape)))

        if mode_config is not None:
            for key, val in mode_config.items():
                setattr(self, key, val)
        else:
            raise ValueError("mode_config must be dict, got None")

        self.compiled = False
        self.grad_op = False

    def train(
            self,
            sess,
            epochs,
            data_dir,
            model_dir,
            minibatch_size,
            save_checkpoints=True,
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

        K.set_session(sess)

        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()

        # set session as self attribute to be available from sigint call
        self.sess = sess
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        train_iters = self.data_shape[0] // minibatch_size
        n_mvavg = 5
        moving_average = [0] * (epochs // n_mvavg)
        to_average = [0]*n_mvavg
        ta_which = 0
        all_lx = np.zeros(epochs)
        all_lz = np.zeros(epochs)

        if self.train_classifier:
            train_interval = 10
            clf_batch_size = self.X_c.shape[0]//(train_iters//train_interval)
            all_clf_loss = np.zeros(epochs)

        for i in range(epochs):
            if self.restore_mode:
                self.saver.restore(sess, checkpoint_fn)
                break

            Lxs = []
            Lzs = []
            logloss_train = []

            bm_inst = BatchManager(self.n_data, minibatch_size)

            if self.train_classifier:
                clf_bm_inst = BatchManager(self.X_c.shape[0], clf_batch_size)

            """
            Epoch train iteration
            """

            for j, ind in enumerate(bm_inst):
                batch = self.X[ind]
                batch = batch.reshape(np.size(ind), self.n_input)

                # self.batch_size: minibatch_size}
                feed_dict = {self.x: batch, }
                if self.scale_kl:
                    feed_dict[self.kl_scale] = np.array([i/epochs, ])

                results = sess.run(self.fetches, feed_dict)
                Lx, Lz, _, _, _, = results

                Lxs.append(Lx)
                Lzs.append(Lz)

                if self.train_classifier:
                    if (j % train_interval) == 0:

                        batch_ind = next(clf_bm_inst)
                        #batch_ind = np.random.randint(0, self.X_c.shape[0], size=(minibatch_size, ))
                        clf_batch = self.X_c[batch_ind]
                        clf_batch = clf_batch.reshape(
                            np.size(batch_ind), self.n_input)

                        t_batch = self.Y_c[batch_ind]

                        # self.batch_size: minibatch_size}
                        clf_feed_dict = {self.x: clf_batch,
                                         self.y_batch: t_batch, }
                        clf_cost, _ = sess.run(
                            self.clf_fetches, clf_feed_dict, options=run_opts)
                        logloss_train.append(clf_cost)

            tmp = np.average(Lzs)
            all_lz[i] = tmp
            all_lx[i] = tf.reduce_mean(Lxs).eval()

            """Compute classifier performance """

            if self.train_classifier:

                all_clf_loss[i] = tf.reduce_mean(logloss_train).eval()

                train_tup = (self.X_c, self.Y_c)
                test_tup = (self.X_c_test, self.Y_c_test)
                scores = [0, 0]

                for k, tup in enumerate([train_tup, test_tup]):

                    score = 0
                    clf_batch = 100
                    X, Y = tup
                    tot = X.shape[0]
                    clf_bm = BatchManager(tot, clf_batch)

                    for bi in clf_bm:
                        n_bi = bi.shape[0]
                        to_pred = X[bi].reshape((n_bi, self.n_input))
                        targets = Y[bi]

                        tmp = np.array(self.score(
                            sess,
                            to_pred,
                            targets,
                            metric=f1_score,
                            metric_kwds={"average": None, "labels": [0, 1, 2]}))
                        score += (n_bi/tot) * tmp

                    scores[k] = score

                print("Epoch {} | Lx = {:5.2f} | Lz = {:5.2f} | clf cost {:5.2f} | \
                        train score {}  | test score {}".format(
                    i,
                    all_lx[i],
                    all_lz[i],
                    all_clf_loss[i],
                    scores[0],
                    scores[1]
                ),
                )
            else:
                print("Epoch {} | Lx = {:5.2f} | Lz = {:5.2f} \r".format(
                    i,
                    all_lx[i],
                    all_lz[i]
                ),
                    end="",
                )

            """
            if all_lz[i] < 0:
                print("broken training")
                print("Lx = ", all_lx[i])
                print("Lz = ", all_lz[i])
                break
            """

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

                if save_checkpoints:
                    self.storeResult(sess, feed_dict, data_dir, model_dir, i)

        return all_lx, all_lz

    def compile_model(
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
            self._ModelLoss()
        else:
            self._ModelLoss(**loss_kwds)
        self.compiled = True

    def compute_gradients(
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

        if self.train_classifier:

            classifier_grads = optimizer.compute_gradients(
                self.classifier_cost)

            for i, (g, v) in enumerate(classifier_grads):
                if g is not None:
                    classifier_grads[i] = (tf.clip_by_norm(g, 5), v)

            self.classifier_op = optimizer.apply_gradients(classifier_grads)
            self.clf_fetches = [self.classifier_cost, self.classifier_op]

        self.grad_op = True

    def _ModelGraph(self,):
        raise NotImplementedError(
            "could not compile pure latent class LatentModel")

    def _ModelLoss(self,):
        raise NotImplementedError(
            "could not compile pure virtual class LatentModel")

    def predict(self, sess, X):
        tmp = {self.x: X, }  # self.batch_size: X.shape[0]}
        return np.argmax(sess.run(self.logits, tmp), 1)

    def predict_proba(self, sess, X):
        tmp = {self.x:  X, }  # self.batch_size: X.shape[0]}
        return sess.run(self.logits, tmp)

    def score(self, sess, X, y, metric=accuracy_score, metric_kwds={}):
        predicted = self.predict(sess, X)
        t = np.argmax(y, 1)
        return metric(t, predicted, **metric_kwds)

    def binary_crossentropy(self, t, o):
        return -(t*tf.log(o+self.eps) + (1.0-t)*tf.log(1.0-o+self.eps))

    def compute_kernel(self, x, y):
        """
        Copied from Shengjia Zhao:
        http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html
        """
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack(
            [x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack(
            [1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def compute_mmd(self, x, y, sigma_sqr=1.0):
        """
        Copied from Shengjia Zhao:
        http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html
        """
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def storeResult(
            self,
            sess,
            feed_dict,
            data_dir,
            model_dir,
            i
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
        print("Saving model and canvasses | epoch: {}".format(i))

        canvasses = sess.run(self.canvas_seq, feed_dict)
        canvasses = np.array(canvasses)
        references = np.array(feed_dict[self.x])
        epoch = "_epoch" + str(i)

        filename = data_dir+"/simulated/canvasses"+epoch + \
            ".npy" if self.simulated_mode else data_dir+"/canvasses"+epoch+".npy"
        np.save(filename, canvasses)

        model_fn = model_dir+"/draw_attn"+epoch + \
            ".ckpt" if self.use_attention else model_dir+"/draw_no_attn"+epoch+".ckpt"
        if self.simulated_mode:
            model_fn = model_dir+"/simulated/draw_attn"+epoch + \
                ".ckpt" if self.use_attention else model_dir + \
                "/simulated/draw_no_attn"+epoch+".ckpt"

        self.saver.save(sess, model_fn)

        ref_fn = data_dir+"/simulated/references"+epoch + \
            ".npy" if self.simulated_mode else data_dir+"/references"+epoch+".npy"
        np.save(ref_fn, references)

    def linear(
        self,
        x,
        output_dim,
        regularizer=tf.contrib.layers.l2_regularizer,
        lmbd=0.1,
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

    def generate_latent(self, sess, save_dir, X_tup, save=True):
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

        for j, X in enumerate(X_tup):
            n_latent = X.shape[0]
            latent_values = np.zeros((self.T, n_latent, self.latent_dim))

            is_seq_model = hasattr(self, "n_decoder_cells")
            if is_seq_model:
                dec_state_array = np.zeros(
                    (self.T, self.n_decoder_cells, 2, n_latent, self.dec_size))
            else:
                dec_state_array = [0,]
            reconstructions = np.zeros((self.T, n_latent, self.H*self.W))
            latent_bm = BatchManager(n_latent, 100)

            #print("T", self.T, "dec size", self.dec_size)
            #print("full dec_seq shape", dec_state_array.shape)

            for ind in latent_bm:

                to_feed = X[ind].reshape((np.size(ind), self.H*self.W))
                feed_dict = {self.x: to_feed,
                             self.batch_size: to_feed.shape[0]}
                to_run = [self.z_seq, self.dec_state_seq, self.canvas_seq]

                z_seq, dec_state_seq, canvasses = sess.run(to_run, feed_dict)
                dec_state_seq = np.array(dec_state_seq)

                #print("dec_state shape", dec_state_seq.shape)

                latent_values[:, ind, :] = z_seq
                reconstructions[:, ind, :] = canvasses
                if is_seq_model:
                    dec_state_seq = dec_state_seq
                    dec_state_array[:, :, :, ind, :] = dec_state_seq

            lat_vals.append(latent_values)
            recons_vals.append(reconstructions)
            decoder_states.append(dec_state_array)

        if save:
            for i, X in enumerate(X_tup):

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
                    np.save(save_dir+"/simulated/" + r_fn, r)
                    np.save(save_dir+"/simulated/" + dec_fn, d)

        else:
            return lat_vals, recons_vals, decoder_states

    def generate_samples(self, save_dir, n_samples=100, rerun_latent=False, load_dir=None):

        z_seq = [0]*self.T

        if rerun_latent:
            if load_dir is None:
                print(
                    "To reconstruct please pass a directory from which to load samples and states")
                return

            latent_fn = load_dir+"/simulated/latent/train_latent.npy" if self.simulated_mode else load_dir + \
                "/latent/train_latent.npy"
            dec_state_fn = load_dir+"/simulated/train_decoder_states.npy" if self.simulated_mode else load_dir + \
                "/train_decoder_states.npy"

            decoder_states = np.load(dec_state_fn)
            latent_samples = np.load(latent_fn)

        dec_state = self.decoder.zero_state(n_samples, tf.float32)
        h_dec = tf.zeros((n_samples, self.dec_size))
        c_prev = tf.zeros((n_samples, self.n_input))

        for t in range(self.T):

            if not rerun_latent:
                mu, sigma = (0, 1) if t < 1 else (0., 1)
                sample = np.random.normal(
                    mu, sigma, (n_samples, self.latent_dim)).astype(np.float32)
            else:
                sample = latent_samples[t, batch_size:2*batch_size, :].reshape(
                    (n_samples, self.latent_dim)).astype(np.float32)

            z_seq[t] = sample
            z = tf.convert_to_tensor(sample)

            if rerun_latent:
                dec_state = decoder_states[t, :, self.batch_size:2*self.batch_size, :].reshape(
                    (2, self.batch_size, dec_size)).astype(np.float32)
                dec_state = tf.nn.rnn_cell.LSTMStateTuple(
                    dec_state[0], dec_state[1])

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

    def signal_handler(self, signal, frame):
        """Function to be called when Ctrl + C is hit"""

        if self.train_classifier:
            self.generate_latent(self.sess, "~/tmp", (self.X_c, ))

        self.sess.close()
        sys.exit(0)

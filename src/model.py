import tensorflow as tf
import signal
import numpy as np
import sys

from keras import backend as K
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import  adjusted_rand_score, normalized_mutual_info_score

from batchmanager import BatchManager
from sklearn.cluster import MiniBatchKMeans

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

    def clustering_layer(self, inputs):
        tmp = tf.square( tf.expand_dims(inputs, axis=1) - self.clusters)
        q = 1.0 / (1.0 + (tf.reduce_sum(
                                tmp,
                                axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q

    def pretrain(
            self,
            sess,
            epochs,
            minibatch_size,
            ):

        print("Pretraining .....")
        self.be_patient = False
        self.prev_loss = 0
        all_lx = np.zeros(epochs)
        smooth_loss = np.zeros(epochs)
        #writer = tf.summary.FileWriter("../loss_records/tensorboard/pretrain", sess.graph) 
        for i in range(epochs):
            bm = BatchManager(self.X.shape[0], minibatch_size)
            lxs = []
            for bi in bm:
                n_bi = bi.shape[0]
                to_run = self.X[bi].reshape((n_bi, self.n_input))
                feed_dict = {self.x: to_run}
                lx, _ = sess.run(self.pretrain_fetch, feed_dict)
                lxs.append(lx)
            
            cur_lx = np.average(lxs)
            all_lx[i] = cur_lx
            print("Lx: {}".format(i),  all_lx[i])

            if (i % 10 == 0):
                self.storeResult(sess, feed_dict, "../drawing", "../models", i)
                """
                summary = sess.run([self.merged,], feed_dict=feed_dict) 
                writer.write_summary(summary, i)
                """
            to_earlystop = self.earlystopping(
                                smooth_loss,
                                all_lx,
                                None, 
                                i
                                )
            if to_earlystop:
                return


    def run_large(
            self,
            sess,
            to_run,
            data,
            input_tensor=None,
            batch_size=100,
            ):

        if input_tensor == None:
            input_tensor = self.x

        to_shape = list(to_run.get_shape())
        to_shape[0] = data.shape[0]
        ret_array = np.zeros(to_shape)
        bm = BatchManager(data.shape[0], batch_size)

        for bi in bm:
            n_bi = bi.shape[0]
            feed_dict = {input_tensor:data[bi].reshape((n_bi, self.n_input))}
            ret_array[bi] = sess.run(to_run, feed_dict)

        return ret_array

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
            run=0,
    ):
        """
        Paramters
        ---------


        """

        if not (self.compiled and self.grad_op):
            print("cannot train before model is compiled and gradients are computed")
            return

        K.set_session(sess)

        self.performance = tf.placeholder(tf.float32, shape=(), name="score")
        tf.summary.scalar("performance", self.performance)
        self.merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                "../loss_records/tensorboard/run_{}".format(run),
                sess.graph)

        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run() 
        # set session as self attribute to be available from sigint call
        self.sess = sess
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        train_iters = self.data_shape[0] // minibatch_size
        all_lx = np.zeros(epochs)
        all_lz = np.zeros(epochs)
        smooth_loss = np.zeros(epochs)

        log_every = 9

        if self.train_classifier:
            train_interval = 10
            clf_batch_size = self.X_c.shape[0]//(train_iters//train_interval)
            all_clf_loss = np.zeros(epochs)

        if self.include_KM:
            self.pretrain_clustering(sess, minibatch_size)

        print("starting training..")
        for i in range(epochs):
            if self.restore_mode:
                self.saver.restore(sess, checkpoint_fn)
                break

            Lxs = []
            Lzs = []
            logloss_train = []
            self.be_patient=False
            bm_inst = BatchManager(self.n_data, minibatch_size)

            if self.train_classifier:
                clf_bm_inst = BatchManager(self.X_c.shape[0], clf_batch_size)

            """
            Update target distribution and check cluster performance
            """
            if self.include_KM:
                if (i % self.update_interval) == 0:
                    to_break, performance = self.update_clusters(
                                                    i, 
                                                    sess,
                                                    data_dir,
                                                    model_dir,
                                                    )

                    if to_break:
                        break
            else:
                performance = np.average([Lxs[-1], Lzs[-1]])

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
                if self.include_KM:
                    feed_dict[self.p] = self.P[ind]

                results = sess.run(self.fetches, feed_dict)
                Lx, Lz, _, _, _, = results

                Lxs.append(Lx)
                Lzs.append(Lz)

                if self.train_classifier:
                    if (j % train_interval) == 0:
                        batch_loss = self.train_classifier(sess, clf_bm_inst, run_opts)
                        logloss_train.append(batch_loss)

            try:
                Lzs = np.array(Lzs)
                tmp = np.average(Lzs)
                all_lz[i] = tmp
            except ValueError:
                all_lz[i] = 1e5

            all_lx[i] = tf.reduce_mean(Lxs).eval()

            """Compute classifier performance """

            if self.train_classifier:
                scores = self.evaluate_classifier(sess, all_clf_loss, logloss_train, i)
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
                print("nan loss value")
                break

            if (1 + i) % log_every == 0 and i >= 0:
                "log performance to tensorboard"
                feed_dict[self.performance] = performance
                summary = sess.run(self.merged, feed_dict=feed_dict)
                writer.add_summary(summary, i)

                if earlystopping:
                    to_earlystop = self.earlystopping(
                            smooth_loss,
                            all_lx,
                            all_lz,
                            i)

                    if to_earlystop:
                        break

                if save_checkpoints:
                    self.storeResult(sess, feed_dict, data_dir, model_dir, i)

        return all_lx, all_lz

    def earlystopping(self, smooth_loss, all_lx, all_lz=None, i=0):
        earlystop_beta = 0.95
        patience=5
        
        if all_lz is None:
            loss = all_lx[i]
        else:
            loss =  np.average([all_lx[i], all_lz[i]])
        smooth_loss[i] = (1 - earlystop_beta) *loss
        smooth_loss[i] += earlystop_beta*self.prev_loss
        self.prev_loss = smooth_loss[i]

        retval = 0
        if i > 10:
            if smooth_loss[i] > smooth_loss[i-1]:
                if not self.be_patient:
                    self.patient_i = i
                self.be_patient = True
            if smooth_loss[i] < smooth_loss[i-1]:
                self.be_patient = False

        if self.be_patient and (i - self.patient_i) == patience: 
            change = np.diff(smooth_loss[self.patient_i:  i])
            mean_change = change.mean()
            print("Earlystopping Mean", mean_change)
            print("changes", change)
            print("values", smooth_loss[self.patient_i: i])
            print("----------")
            self.be_patient = False
            if mean_change > 0:
                retval = 1 

        return retval

    def update_clusters(
            self,
            i,
            sess,
            data_dir,
            model_dir
            ):

        tot_samp = self.X.shape[0]
        q = np.zeros((tot_samp, self.n_clusters))
        clst_bm = BatchManager(tot_samp, 100)
        retval = 0

        for bi in clst_bm:
            n_bi = bi.shape[0]
            to_pred = self.X[bi].reshape((n_bi, self.n_input))
            feed_dict = {self.x: to_pred}
            q[bi] = sess.run(self.q, feed_dict)

        self.P = self.t_distribution(q)
        all_pred = q.argmax(1)

        if i == 0:
            self.y_prev = all_pred
        else:
            precent_changed = self.label_change(self.y_prev, all_pred)
            print()
            print("precent_changed: ", precent_changed)
            if precent_changed  < self.delta:
                self.storeResult(sess, feed_dict, data_dir, model_dir, i)
                retval = 1
            self.y_prev = all_pred

        if self.labelled_data != None:
            y_pred, targets = self.predict_cluster(sess,)
            ars = adjusted_rand_score(targets, y_pred)
            nmi = normalized_mutual_info_score(targets, y_pred)
            performance = ars
            print( ) 
            print("Confusion matrix: " )
            print(confusion_matrix(targets, y_pred))
            print("ARS : {:.3f} ' NMI: {:.3f} ".format(ars, nmi))

        return retval, ars

    def predict_cluster(self, sess):
        x_label = self.labelled_data[0]
        n_x = x_label.shape[0]
        x_label = np.reshape(x_label, (n_x, self.n_input))
        targets = self.labelled_data[1]
        q_label = self.run_large(sess, self.q, x_label)
        y_pred = q_label.argmax(1)
        return y_pred, targets

    def pretrain_clustering(self, sess, minibatch_size):
        if self.pretrain_simulated:
            print("Simulated pretrain....")
            self.train_simulated(sess, 15, minibatch_size)

        self.pretrain(sess, self.pretrain_epochs, minibatch_size)

        print("Training K-means..... ")
        km = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    n_init=50,
                    batch_size=150,
                    reassignment_ratio=0.5,
                    max_iter=1000
                    )
        
        print("Compute z")
        z = self.run_large(sess, self.z_seq[0], self.X,)
        print("Fit k-means")
        km.fit(z)
        print("assign clusters")
        self.clusters.load(km.cluster_centers_, sess)

    def train_simulated(self, sess, epochs, minibatch_size):
        self.clf_fetches = self.compute_classifier_cost(self.optimizer)

        for i in range(epochs):
            clf_bm_inst = BatchManager(self.X_c.shape[0], minibatch_size)
            all_clf_cost =  []
            all_clf_acc = []
            for batch_ind in clf_bm_inst:
                clf_cost, clf_acc = self.run_clf_batch(sess, batch_ind, )
                all_clf_cost.append(clf_cost)
                all_clf_acc.append(clf_acc)

            print(
                "clf cost", i, np.mean(all_clf_cost), 
                " | clf_acc", i, np.mean(all_clf_acc),
                )

        return all_clf_cost

    def run_clf_batch(self, sess, batch_ind ):
        #batch_ind = np.random.randint(0, self.X_c.shape[0], size=(minibatch_size, ))
        clf_batch = self.X_c[batch_ind]
        clf_batch = clf_batch.reshape(
                        np.size(batch_ind), self.n_input)
        t_batch = self.Y_c[batch_ind]

        # self.batch_size: minibatch_size}
        clf_feed_dict = {self.x: clf_batch,
                            self.y_batch: t_batch, }
        clf_cost, clf_acc, _ = sess.run(
            self.clf_fetches, clf_feed_dict, )
        return clf_cost, clf_acc

    def evaluate_classifier(
            self,
            sess,
            all_clf_loss,
            logloss_train,
            i
            ):

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
        return scores

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
        self.optimizer = optimizer
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

        if self.include_KM:
            lx_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
            lx_grads = lx_optimizer.compute_gradients(self.Lx)

            for i, (g, v) in enumerate(lx_grads):
                if g is not None:
                    lx_grads[i] = (tf.clip_by_norm(g, 5), v)

            lx_train_op = lx_optimizer.apply_gradients(lx_grads)
            self.pretrain_fetch = [self.Lx, lx_train_op]

        if self.train_classifier:
            self.clf_fetches = self.compute_classifier_cost(optimizer)

        self.grad_op = True

    def compute_classifier_cost(self, optimizer):
        classifier_grads = optimizer.compute_gradients(
            self.classifier_cost)

        for i, (g, v) in enumerate(classifier_grads):
            if g is not None:
                classifier_grads[i] = (tf.clip_by_norm(g, 5), v)

        self.classifier_op = optimizer.apply_gradients(classifier_grads)
        clf_fetches = [self.classifier_cost, self.clf_acc, self.classifier_op]
        return clf_fetches


    def _ModelGraph(self,):
        raise NotImplementedError(
            "could not compile pure latent class LatentModel")

    def _ModelLoss(self,):
        raise NotImplementedError(
            "could not compile pure virtual class LatentModel")

    def label_change(self, a, b):
        diff = a-b
        nonzero = np.nonzero(diff)[0]
        return len(nonzero)/len(a)

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

    def mse(self, t, o):
        return (t -  o)**2

    def t_distribution(self, q):
        w = q**2 / q.sum(0)
        return (w.T / w.sum(1)).T

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

    def variable_summary(self, var):
        """
        Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var, )
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


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

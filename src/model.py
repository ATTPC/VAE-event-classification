


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
            self.n_input = self.data_shape[1] * self.data_shape[2]
            self.H = self.data_shape[1]
            self.W = self.data_shape[2]
            self.n_data = self.data_shape[0]

        if not mode_config is None:
            for key, val in mode_config.items():
                setattr(self, key, val)
        else:
            raise ValueError("mode_config must be dict, got None")

        self.compiled = False
        self.grad_op = False

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
            self._ModeLoss()
        else:
            self._ModelLoss(**loss_kwds)

        self.compiled = True


    def _ModelGraph(self,):
        raise NotImplementedError("could not compile pure latent class LatentModel")


    def _ModelLoss(self,):
        raise NotImplementedError("could not compile pure virtual class LatentModel")


    def predict(self, sess, X):
        tmp = {self.x: X,}# self.batch_size: X.shape[0]}
        return np.argmax(sess.run(self.logits, tmp), 1)


    def predict_proba(self, sess, X):
        tmp = {self.x:  X, }#self.batch_size: X.shape[0]}
        return sess.run(self.logits, tmp)


    def score(self, sess, X, y, metric=accuracy_score, metric_kwds={}):
        predicted = self.predict(sess, X)
        t  = np.argmax(y, 1)
        return metric(t, predicted, **metric_kwds)


    def binary_crossentropy(self, t, o):
        return -(t*tf.log(o+self.eps) + (1.0-t)*tf.log(1.0-o+self.eps))



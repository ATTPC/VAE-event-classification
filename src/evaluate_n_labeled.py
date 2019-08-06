from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from batchmanager import BatchManager
from joblib import Parallel, delayed
import scipy

import numpy as np

def n_labeled_data(x_set, y_set, sample_size, dataset_names, ):
    means = [0,]*len(x_set)
    stds  = [0,]*len(x_set)
    for i in range(len(x_set)):
        print("Evaluating dataset: ", i)
        x = x_set[i]
        x = x if len(x.shape) == 2 else x.reshape((x.shape[0], -1))
        y = y_set[i]
        if scipy.sparse.issparse(y):
            y = y.toarray()
        y = y if len(y.shape) == 1 else y.argmax(-1)
        means[i], stds[i] = n_labeled_estimator(x, y, sample_size, classes=np.unique(y))
    return means, stds

def n_labeled_estimator(x, y, sample_size, classes=3):
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size=0.7)
    n_train = x_tr.shape[0]
    n_iter = 100
    performance = np.zeros((n_iter, (n_train // sample_size) + 1))
    lw = loop_wrapper(performance, x_te, y_te, n_train, sample_size)
    
    Parallel(n_jobs=10)([delayed(lw)(x_tr, y_tr, i) for i in range(n_iter)])
    
    mean_performances = lw.performance.mean(axis=0)
    std_performances = lw.performance.std(axis=0)
    return mean_performances, std_performances

class loop_wrapper:
    def __init__(self, performance, x_te, y_te, n_train, sample_size):
        self.performance = performance
        self.x_te = x_te
        self.y_te = y_te
        self.n_train = n_train
        self.sample_size = sample_size
        self.i = 0
        
    def __call__(self, x_tr, y_tr, i):
        bm = BatchManager(self.n_train, self.sample_size)
        x_samp = None
        for j, ind in enumerate(bm):
            if x_samp is None :
                x_samp = x_tr[ind]
                y_samp = y_tr[ind]
            else:
                x_samp = np.concatenate(
                    [x_samp, x_tr[ind]],
                    axis=0
                )
                y_samp = np.concatenate(
                    [y_samp, y_tr[ind]],
                    axis=0
                )
            lr = fit_model(x_samp, y_samp)
            self.performance[i, j] = f1_score(self.y_te, lr.predict(self.x_te), average="macro")

def fit_model(x_samp, y_samp): 
    lr = LogisticRegression(
    class_weight="balanced",
    multi_class="ovr",
    solver="newton-cg",
    max_iter=1000,
    penalty="l2"
    )
    lr.fit(x_samp, y_samp)
    return lr

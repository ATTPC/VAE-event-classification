#!/usr/bin/env python3
import h5py
import numpy as np
import scipy
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder


def DataLoader(file_location):
    fileobj = h5py.File(file_location, "r")
    X_t = fileobj["train_features"]
    y_t = fileobj["train_targets"]

    X_v = fileobj["test_features"]
    y_v = fileobj["test_targets"]

    return X_t, y_t, X_v, y_v

def load_mnist(size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1)/255
    x_test= np.expand_dims(x_test, -1)/255
    x_tot = np.concatenate([x_train, x_test])
    return x_train, x_test, y_test

def load_real_vgg(size):
    size = str(size)
    vgg_run_130 = np.load("../data/real/vgg_images/run_0130_label_False_size_"+size+".npy")
    vgg_run_150 = np.load("../data/real/vgg_images/run_0150_label_False_size_"+size+".npy")
    vgg_run_190 = np.load("../data/real/vgg_images/run_0150_label_False_size_"+size+".npy")
    vgg_run_210 = np.load("../data/real/vgg_images/run_0210_label_False_size_"+size+".npy")
    run_130 = np.load("../data/real/images/run_0130_label_False_size_"+size+".npy")
    run_150 = np.load("../data/real/images/run_0150_label_False_size_"+size+".npy")
    run_190 = np.load("../data/real/images/run_0150_label_False_size_"+size+".npy")
    run_210 = np.load("../data/real/images/run_0210_label_False_size_"+size+".npy")
    vgg_x_train = np.concatenate([vgg_run_130, vgg_run_150, vgg_run_190, vgg_run_210])
    x_train = np.concatenate([run_130, run_150, run_190, run_210])
    x_lab_tr = np.load("../data/real/vgg_images/train_size_"+size+".npy")
    x_lab_te = np.load("../data/real/vgg_images/test_size_"+size+".npy")
    x_test = np.concatenate((x_lab_tr, x_lab_te))
    y_tr = np.load("../data/real/targets/train_targets_size_"+size+".npy")
    y_te = np.load("../data/real/targets/test_targets_size_"+size+".npy")
    y_test = np.concatenate((y_tr, y_te))
    if scipy.sparse.issparse(y_test):
        y_test = y_test.toarray()
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return vgg_x_train, x_train, x_test, y_test

def load_real_event(size, event="0210"):
    event = str(event)
    run_210 = np.load("../data/real/images/run_"+event+"_label_False_size_"+size+".npy")
    x_train = run_210 #np.concatenate([run_130, run_150, run_190, run_210])
    #x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = np.load("../data/real/images/run_"+event+"_label_True_size_"+size+".npy")
    y_test = np.load("../data/real/targets/run_"+event+"_targets_size_"+size+".npy")
    if scipy.sparse.issparse(y_test):
        y_test = y_test.toarray()
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return x_train, x_test, y_test

def load_real_vgg_event(size, event="0210"):
    event = str(event)
    run_210 = np.load("../data/real/images/run_"+event+"_label_False_size_"+size+".npy")
    vgg_run_210 = np.load("../data/real/vgg_images/run_"+event+"_label_False_size_"+size+".npy")
    x_train = run_210 #np.concatenate([run_130, run_150, run_190, run_210])
    x_train = x_train.reshape((x_train.shape[0], -1))
    vgg_x_train = vgg_run_210
    x_test = np.load("../data/real/vgg_images/run_"+event+"_label_True_size_"+size+".npy")
    y_test = np.load("../data/real/targets/run_"+event+"_targets_size_"+size+".npy")
    if scipy.sparse.issparse(y_test):
        y_test = y_test.toarray()
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return vgg_x_train, x_train, x_test, y_test

def load_clean_vgg_event(size, event="0210"):
    event = str(event)
    run_210 = np.load("../data/clean/images/run_"+event+"_label_False_size_"+size+".npy")
    vgg_run_210 = np.load("../data/clean/vgg_images/run_"+event+"_label_False_size_"+size+".npy")
    x_train = run_210 #np.concatenate([run_130, run_150, run_190, run_210])
    x_train = x_train.reshape((x_train.shape[0], -1))
    vgg_x_train = vgg_run_210
    x_test = np.load("../data/clean/vgg_images/run_"+event+"_label_True_size_"+size+".npy")
    y_test = np.load("../data/clean/targets/run_"+event+"_targets_size_"+size+".npy")
    if scipy.sparse.issparse(y_test):
        y_test = y_test.toarray()
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return vgg_x_train, x_train, x_test, y_test

def load_clean_event(size, event="0210"):
    event = str(event)
    run_210 = np.load("../data/clean/images/run_"+event+"_label_False_size_"+size+".npy")
    x_train = run_210 #np.concatenate([run_130, run_150, run_190, run_210])
    x_test = np.load("../data/clean/images/run_"+event+"_label_True_size_"+size+".npy")
    y_test = np.load("../data/clean/targets/run_"+event+"_targets_size_"+size+".npy")
    if scipy.sparse.issparse(y_test):
        y_test = y_test.toarray()
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return x_train, x_test, y_test


def load_clean_vgg(size):
    size = str(size)
    vgg_run_130 = np.load("../data/clean/vgg_images/run_0130_label_False_size_"+size+".npy")
    vgg_run_150 = np.load("../data/clean/vgg_images/run_0150_label_False_size_"+size+".npy")
    vgg_run_190 = np.load("../data/clean/vgg_images/run_0150_label_False_size_"+size+".npy")
    vgg_run_210 = np.load("../data/clean/vgg_images/run_0210_label_False_size_"+size+".npy")
    run_130 = np.load("../data/clean/images/run_0130_label_False_size_"+size+".npy")
    run_150 = np.load("../data/clean/images/run_0150_label_False_size_"+size+".npy")
    run_190 = np.load("../data/clean/images/run_0150_label_False_size_"+size+".npy")
    run_210 = np.load("../data/clean/images/run_0210_label_False_size_"+size+".npy")
    vgg_x_train = np.concatenate([vgg_run_130, vgg_run_150, vgg_run_190, vgg_run_210])
    x_train = np.concatenate([run_130, run_150, run_190, run_210])
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = np.load("../data/clean/vgg_images/train_size_"+size+".npy")
    y_test = np.load("../data/clean/targets/train_targets_size_"+size+".npy")
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return vgg_x_train, x_train, x_test, y_test
def load_clean(size):
    size = str(size)
    run_130 = np.load("../data/clean/images/run_0130_label_False_size_"+size+".npy")
    run_150 = np.load("../data/clean/images/run_0150_label_False_size_"+size+".npy")
    run_190 = np.load("../data/clean/images/run_0150_label_False_size_"+size+".npy")
    run_210 = np.load("../data/clean/images/run_0210_label_False_size_"+size+".npy")
    x_train = np.concatenate([run_130, run_150, run_190, run_210])
    x_test = np.load("../data/clean/images/train_size_"+size+".npy")
    y_test = np.load("../data/clean/targets/train_targets_size_"+size+".npy")
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return x_train, x_test, y_test

def load_real(size):
    size = str(size)
    run_130 = np.load("../data/real/images/run_0130_label_False_size_"+size+".npy")
    run_150 = np.load("../data/real/images/run_0150_label_False_size_"+size+".npy")
    run_190 = np.load("../data/real/images/run_0150_label_False_size_"+size+".npy")
    run_210 = np.load("../data/real/images/run_0210_label_False_size_"+size+".npy")
    x_train = np.concatenate([run_130, run_150, run_190, run_210])
    x_test = np.load("../data/real/images/train_size_"+size+".npy")
    y_test = np.load("../data/real/targets/train_targets_size_"+size+".npy")
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(3)]).fit_transform(y_test)
    return x_train, x_test, y_test

def load_simulated(size):
    x_train = np.load("../data/simulated/images/pr_train_simulated.npy")
    x_test = np.load("../data/simulated/images/pr_test_simulated.npy")
    x_train = np.concatenate([x_train, x_test])
    y_train = np.load("../data/simulated/targets/train_targets.npy")
    y_test = np.load("../data/simulated/targets/test_targets.npy")
    y_test = y_test.reshape((-1, 1))
    y_test = OneHotEncoder(categories=[range(2)]).fit_transform(y_test)
    return x_train, x_test, y_test

if __name__ == "__main__":
    #file_location = "~/Documents/github/VAE-event-classification/data/real/packaged/x-y/proton-carbon-junk-noise.h5"
    #a = DataLoader(file_location)
    x, xt, yt = load_real("80")
    print(yt.shape)

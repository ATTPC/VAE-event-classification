#!/usr/bin/env python3
import h5py
import numpy as np
from tensorflow.keras.datasets import mnist


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

def load_clean(size):
    size = str(size)
    run_130 = np.load("../data/clean/images/run_0130_label_False_size_"+size+".npy")
    run_150 = np.load("../data/clean/images/run_0150_label_False_size_"+size+".npy")
    run_190 = np.load("../data/clean/images/run_0150_label_False_size_"+size+".npy")
    run_210 = np.load("../data/clean/images/run_0210_label_False_size_"+size+".npy")
    x_train = np.concatenate([run_130, run_150, run_190])
    x_test = np.load("../data/clean/images/train_size_"+size+".npy")
    y_test = np.load("../data/clean/targets/train_targets_size_"+size+".npy")
    return x_train, x_test, y_test

def load_real(size):
    #Labelled data for testing
    with h5py.File("../data/images.h5", "r") as fo:
        train_targets = np.array(fo["train_targets"])
        test_targets = np.array(fo["test_targets"])
    all_0130 = np.load("../data/processed/all_0130.npy")
    all_0210 = np.load("../data/processed/all_0210.npy")
    x_train = np.concatenate([all_0130, all_0210])
    #train_data = np.load("../data/processed/train.npy")
    test_data = np.load("../data/processed/test.npy")
    train_data = np.load("../data/processed/train.npy")
    x_test = train_data
    y_test = train_targets 
    return x_train, x_test, y_test

def load_simulated(size):
    x_train = np.load("../data/simulated/pr_train_simulated.npy")
    x_test = np.load("../data/simulated/pr_test_simulated.npy")
    x_train = np.concatenate([x_train, x_test])
    y_train = np.load("../data/simulated/train_targets.npy")
    y_test = np.load("../data/simulated/test_targets.npy")
    return x_train, x_test, y_test

if __name__ == "__main__":

    file_location = "~/Documents/github/VAE-event-classification/data/real/packaged/x-y/proton-carbon-junk-noise.h5"
    a = DataLoader(file_location)

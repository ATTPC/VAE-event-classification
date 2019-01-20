#!/usr/bin/env python3

from data_loader import DataLoader
from skimage import util
import numpy as np

file_location = "/home/solli-comphys/github/VAE-event-classification/data/real/packaged/x-y/proton-carbon-junk-noise.h5"
X_train, y_train, X_test, y_test = DataLoader(file_location)

def rgb2gray(rgb):

    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

X_train = rgb2gray(X_train)/255
for i, x in enumerate(X_train):
    X_train[i] = util.invert(x)

X_test = rgb2gray(X_test)/255
for i, x in enumerate(X_test):
    X_test[i] = util.invert(x)

threshhold_indices_train = X_train < 1e-2
threshhold_indices_test = X_test < 1e-2

X_train[threshhold_indices_train] = 0
X_test[threshhold_indices_test] = 0

# xStd = np.std(X_train[np.nonzero(X_train)])
# X_train = X_train/xStd
# X_test = X_test/xStd


X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

t_fn = "/home/solli-comphys/github/VAE-event-classification/data/processed/train.npy"
te_fn = "/home/solli-comphys/github/VAE-event-classification/data/processed/test.npy"

np.save(t_fn, X_train)
np.save(te_fn, X_test)

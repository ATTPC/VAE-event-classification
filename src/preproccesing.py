#!/usr/bin/env python3

from data_loader import DataLoader
from skimage import util
import sys
import numpy as np

file_location = "../data/images.h5"
test_simulated = "../data/simulated/test_data.npy"
train_simulated = "../data/simulated/train_data.npy"

train_simulated = np.load(train_simulated)
test_simulated = np.load(test_simulated)

nonzero_train = np.nonzero(train_simulated)
nonzero_test = np.nonzero(test_simulated)

all_nonzero = np.concatenate([
                train_simulated[nonzero_train],
                test_simulated[nonzero_test],    
                ], axis = 0)
maxval = np.max(all_nonzero)
minval = np.min(all_nonzero)

b = (0.1 - minval/maxval) * (1 / (1 - minval/maxval))
a = 1/maxval - b/maxval

unitmap = lambda x: a*x + b 

train_simulated[nonzero_train] = unitmap(train_simulated[nonzero_train])
test_simulated[nonzero_test] = unitmap(test_simulated[nonzero_test])

# print(np.max(train_simulated))
np.save("../data/simulated/pr_train_simulated.npy", train_simulated)
np.save("../data/simulated/pr_test_simulated.npy", test_simulated)
sys.exit()
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

t_fn = "../data/processed/train.npy"
te_fn = "../data/processed/test.npy"

np.save(t_fn, X_train)
np.save(te_fn, X_test)

import numpy as np
import h5py
from skimage import util

which = "0210"

with h5py.File(
    "../data/proton-carbon-junk-noise-unlabeled-" + which + ".h5", "r"
) as fo:
    X_train = np.array(fo["images"])


def rgb2gray(rgb):

    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


X_train = rgb2gray(X_train) / 255

for i, x in enumerate(X_train):
    X_train[i] = util.invert(x)

threshhold_indices_train = X_train < 1e-2

X_train[threshhold_indices_train] = 0

# xStd = np.std(X_train[np.nonzero(X_train)])
# X_train = X_train/xStd
# X_test = X_test/xStd


X_train = X_train.reshape(X_train.shape + (1,))

t_fn = "../data/processed/all_" + which + ".npy"

np.save(t_fn, X_train)

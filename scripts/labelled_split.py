from sklearn.model_selection import train_test_split
import numpy as np
import os


arrays = []
targets = []
data = "clean"
size = "128"
tmp = [arrays, targets]
fps = ["../data/" + data + "/images", "../data/" + data + "/targets"]

fp = fps[0]
for (dirpath, dirnames, fnames) in os.walk(fp):
    for fname in fnames:
        if "True" in fname and size in fname:
            tmp[0].append(np.load(fp + "/" + fname))
            print(fname, " loaded")

fp = fps[1]
runs = ["0130", "0210"]
for run in runs:
    fn = fp + "/run_" + run + "_targets_size_" + size + ".npy"
    targets.append(np.load(fn))

arrays = np.concatenate(arrays)
targets = np.expand_dims(np.concatenate(targets), -1)
x_train, x_test, y_train, y_test = train_test_split(arrays, targets, test_size=0.2)

np.save(fps[0] + "/train_size_" + size + ".npy", x_train)
np.save(fps[0] + "/test_size_" + size + ".npy", x_test)
np.save(fps[1] + "/train_targets_size_" + size + ".npy", y_train)
np.save(fps[1] + "/test_targets_size_" + size + ".npy", y_test)

import tensorflow as tf
import numpy as np
import h5py
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch

from randomsearch_run import run

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data = "clean"

print("PID", os.getpid())

if data=="simulated":
    x_train = np.load("../data/simulated/pr_train_simulated.npy")
    x_test = np.load("../data/simulated/pr_test_simulated.npy")

    y_train = np.load("../data/simulated/train_targets.npy")
    y_test = np.load("../data/simulated/test_targets.npy")
elif data == "real":
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
elif data=="clean":
    run_130 = np.load("../data/clean/images/run_0130_label_False_size_80.npy")
    run_150 = np.load("../data/clean/images/run_0150_label_False_size_80.npy")
    run_190 = np.load("../data/clean/images/run_0150_label_False_size_80.npy")
    x_train = np.concatenate([run_130, run_150, run_190])
    x_test = np.load("../data/clean/images/train_size_80.npy")
    y_test = np.load("../data/clean/targets/train_targets_size_80.npy")

n = 1000
try:
    os.mkdir("../randomsearch_simulated/run_{}".format(run))
except:
    pass

with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run+1))

rs = RandomSearch(x_train, x_test, y_test, ConVaeGenerator, architecture="ours")
rs.search(n, 1000, "../randomsearch_"+data+"/run_{}/".format(run))



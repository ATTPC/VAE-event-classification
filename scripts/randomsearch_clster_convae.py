import tensorflow as tf
import numpy as np
import h5py
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch

from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
data = "real"

print("PID", os.getpid())

if data=="simulated":
    x_train = np.load("../data/simulated/pr_train_simulated.npy")
    x_test = np.load("../data/simulated/pr_test_simulated.npy")

    y_train = np.load("../data/simulated/train_targets.npy")
    y_test = np.load("../data/simulated/test_targets.npy")
if data == "real":
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

n = 1000
try:
    os.mkdir("../randomsearch_simulated/run_{}".format(run))
except:
    pass

with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run+1))

rs = RandomSearch(x_train, x_test, y_test, ConVaeGenerator, architecture="ours")
rs.search(n, 1000, "../randomsearch_"+data+"/run_{}/".format(run))



import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch

from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data = "simulated"

print("PID", os.getpid())

if data=="simulated":
    x_train = np.load("../data/simulated/pr_train_simulated.npy")
    x_test = np.load("../data/simulated/pr_test_simulated.npy")

    y_train = np.load("../data/simulated/train_targets.npy")
    y_test = np.load("../data/simulated/test_targets.npy")

n = 100
print("OG IMG DIM", x_train.shape[1])
try:
    os.mkdir("../randomsearch_simulated/run_{}".format(run))
except:
    pass

with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run+1))

rs = RandomSearch(x_train, x_test, y_test, ConVaeGenerator, architecture="own")
rs.search(n, 100, "../randomsearch_simulated/run_{}/".format(run))



import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data = "simulated"

if data=="simulated":
    x_train = np.load("../data/simulated/pr_train_simulated.npy")[0:200]
    x_test = np.load("../data/simulated/pr_test_simulated.npy")[0:100]

    y_train = np.load("../data/simulated/train_targets.npy")[0:200]
    y_test = np.load("../data/simulated/test_targets.npy")[0:100]

n = 3
print("OG IMG DIM", x_train.shape[1])
rs = RandomSearch(x_train, x_test, y_test, ConVaeGenerator)
rs.search(n, 100, "../randomsearch_simulated/")

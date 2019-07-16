import tensorflow as tf
import numpy as np
import h5py
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch
from data_loader import load_clean

from randomsearch_run import run

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data = "clean"
size = "80"
x_train, x_test, y_test = load_clean(size)

print("PID", os.getpid())

n = 1000
with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run+1))

rs = RandomSearch(x_train, x_test, y_test, ConVaeGenerator, architecture="ours")
rs.search(n, 1000, "../randomsearch_"+data+"/run_{}/".format(run))



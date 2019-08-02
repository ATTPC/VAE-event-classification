import tensorflow as tf
import numpy as np
import h5py
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch
from data_loader import *
from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data = "realevent"
size = "128"
data = "realevent"
method = "clster"

if data == "real":
    x_train, x_test, y_test = load_real(size)
elif data == "realevent":
    which = "0210"
    x_train, x_test, y_test = load_real_event(size, which)
else:
    x_train, x_test, y_test = load_clean(size)

print("PID", os.getpid())

n = 1000
savedir = "../randomsearch_convae_"+data+"_"+size+"_"+method+"/run_{}".format(run)
with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run+1))

savedir = "../randomsearch_convae_"+data+"_"+size+"_"+method+"/run_{}".format(run)
rs = RandomSearch(x_train, x_test, y_test, ConVaeGenerator, architecture="ours")
rs.search(n, 1000, savedir)



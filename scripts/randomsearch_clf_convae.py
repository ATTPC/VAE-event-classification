import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch
from data_loader import load_clean, load_real
from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("PID: ", os.getpid())
size = 80
data = "real"
if data == "real":
    x_train, x_test, y_test = load_real(size)
else:
    x_train, x_test, y_test = load_clean(size)

x_train = x_train[np.random.randint(0, x_train.shape[0], size=(20000,))]
n = 1000
savedir = "../randomsearch_convae_"+data+"_{}_clf/run_{}".format(size, run)
with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run+1))

rs = RandomSearch(
        x_train,
        x_test,
        y_test,
        ConVaeGenerator,
        clustering=False,
        architecture="ours",
        )
rs.search(n, 150, savedir)

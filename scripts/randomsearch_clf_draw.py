import tensorflow as tf
import numpy as np
import os

import sys

sys.path.append("../src")
from draw_generator import DRAWGenerator
from randomsearch import RandomSearch
from data_loader import load_clean, load_simulated
from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("PID: ", os.getpid())
size = 80
x_train, x_test, y_test = load_simulated(size)
# x_train = x_train[np.random.randint(0, x_train.shape[0], size=(20000,))]

n = 1000
savedir = "../randomsearch_draw_simulated_clf/run_{}".format(run)
with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run + 1))

rs = RandomSearch(
    x_train, x_test, y_test, DRAWGenerator, clustering=False, architecture="ours"
)
rs.search(n, 150, savedir)

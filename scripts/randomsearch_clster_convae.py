import tensorflow as tf
import numpy as np
import h5py
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch
import data_loader as dl
from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
size = "128"
data = "simulated"
method = "clster"
architecture = "ours"
use_dd = True
reconst = "net_charge"

if "vgg" in data:
    use_vgg = True
else:
    use_vgg = False

if reconst == "hist":
    dd_loader = dl.load_simulated_hist
else:
    dd_loader = dl.load_simulated_netcharge

if data == "real":
    x_train, x_test, y_test = dl.load_real(size)
elif data == "clean":
    x_train, x_test, y_test = dl.load_clean(size)
elif data == "realevent":
    which = "0210"
    x_train, x_test, y_test = dl.load_real_event(size, which)
elif data == "simulated":
    x_train, x_test, y_test = dl.load_simulated(size,)
    if use_dd:
        dd_train, dd_test = dd_loader(size)

print("PID", os.getpid())

n = 1000
savedir = "../results/randomsearch_convae_"+data+"_"+size+"_"+method+"/run_{}".format(run)
with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run+1))

if use_vgg:
    rs = RandomSearch(
            vgg_x_train,
            x_test,
            y_test,
            ConVaeGenerator,
            clustering=True,
            architecture=architecture,
            use_vgg_repr=use_vgg,
            target_images=x_train,
            use_dd=use_dd,
            dd_targets=dd_train
            )
else:
    rs = RandomSearch(
            x_train,
            x_test,
            y_test,
            ConVaeGenerator,
            clustering=True,
            architecture=architecture,
            use_dd=use_dd,
            dd_targets=dd_train
            )
rs.search(n, 150, savedir)

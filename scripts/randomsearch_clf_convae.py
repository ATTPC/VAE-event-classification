import tensorflow as tf
import numpy as np
import os

import sys

sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch
import data_loader as dl
from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("PID: ", os.getpid())
size = "128"
data = "realevent"
architecture = "static"
method = "clf"
use_dd = True
q_hist_x_train = None

if "vgg" in data:
    use_vgg = True
else:
    use_vgg = False

if data == "real":
    x_train, x_test, y_test = dl.load_real(size)
elif data == "realevent":
    which = "0210"
    x_train, x_test, y_test = dl.load_real_event(size, which)
    if use_dd:
        # q_hist_x_train, q_hist_x_test = dl.load_realevent_hist(size)
        q_hist_x_train, q_hist_x_test = dl.load_realevent_netcharge(size)
        print(q_hist_x_train.shape)
elif data == "cleanevent":
    which = "0210"
    x_train, x_test, y_test = dl.load_clean_event(size, which)
elif data == "vgg_realevent":
    which = "0210"
    vgg_x_train, x_train, x_test, y_test = dl.load_real_vgg_event(size, which)
    if use_dd:
        q_hist_x_train, q_hist_x_test = dl.load_realevent_hist(size)
elif data == "vgg_cleanevent":
    which = "0210"
    vgg_x_train, x_train, x_test, y_test = dl.load_clean_vgg_event(size, which)
elif data == "vgg_simulated":
    vgg_x_train, x_train, x_test, y_test = dl.load_vgg_simulated(size)
else:
    x_train, x_test, y_test = dl.load_clean(size)

n = 1000
savedir = (
    "../results/randomsearch_convae_"
    + data
    + "_"
    + size
    + "_"
    + method
    + "/run_{}".format(run)
)
with open("randomsearch_run.py", "w") as fo:
    fo.write("run={}".format(run + 1))

if use_vgg:
    rs = RandomSearch(
        vgg_x_train,
        x_test,
        y_test,
        ConVaeGenerator,
        clustering=False,
        architecture="ours",
        use_vgg_repr=use_vgg,
        target_images=x_train,
        use_dd=use_dd,
        dd_targets=q_hist_x_train,
    )
else:
    rs = RandomSearch(
        x_train,
        x_test,
        y_test,
        ConVaeGenerator,
        clustering=False,
        architecture=architecture,
        use_dd=use_dd,
        dd_targets=q_hist_x_train,
    )
rs.search(n, 150, savedir)

import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append("../src")
from convae_generator import ConVaeGenerator
from randomsearch import RandomSearch
from data_loader import load_clean, load_real
from data_loader import load_real_vgg_event, load_clean_vgg_event
from data_loader import load_real_event, load_clean_event
from randomsearch_run import run

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("PID: ", os.getpid())
size = "128"
use_vgg = False
data = "vgg_cleanevent"
method = "clf"

if data == "real":
    x_train, x_test, y_test = load_real(size)
elif data == "realevent":
    which = "0210"
    x_train, x_test, y_test = load_real_event(size, which)
elif data == "cleanevent":
    which = "0210"
    x_train, x_test, y_test = load_clean_event(size, which)
elif data == "vgg_realevent":
    which = "0210"
    use_vgg = True
    vgg_x_train, x_train, x_test, y_test = load_real_vgg_event(size, which)
elif data == "vgg_cleanevent":
    which = "0210"
    use_vgg = True
    vgg_x_train, x_train, x_test, y_test = load_clean_vgg_event(size, which)
else:
    x_train, x_test, y_test = load_clean(size)

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
            clustering=False,
            architecture="ours",
            use_vgg_repr=use_vgg,
            target_images=x_train
            )
else:
    rs = RandomSearch(
            x_train,
            x_test,
            y_test,
            ConVaeGenerator,
            clustering=False,
            architecture="ours",
            )
rs.search(n, 150, savedir)

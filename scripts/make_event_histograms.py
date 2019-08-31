import numpy as np
import sys
import os
sys.path.append("../src")

from data_loader import *
from event_representations import make_histograms, make_net_count


data = ["simulated/", "real/", "clean/"]
represent = "hist"
size = "128"
base_fp = "../data/"

if represent == "hist":
    repr_func = make_histograms
    to_dir = "q_histograms/"
elif represent == "net_charge": 
    repr_func = make_net_count
    to_dir = "net_charge/"

for d in data:
    fp = base_fp+d+"images/"
    to_fp = base_fp+d+ to_dir
    finished_files = os.listdir(to_fp)
    hist_list = []
    for fn in os.listdir(fp):
        if size in fn or "simulated" in d:
            if not fn in finished_files:
                imgs = np.load(fp+fn)
                interval = [6e-1, 1]
                if "simulated" in d:
                    interval = [2.9e-1, .9]
                hists = repr_func(imgs, interval=[6e-1, 1])
                if len(hists.shape) == 1:
                    hists = hists.reshape((-1, 1))
                hist_list.append(hists)
                print("Loaded: ", fn)  

    means = np.zeros(len(hist_list))
    for i in range(len(hist_list)):
        means[i] = hist_list[i].sum()/hist_list[i].shape[0]

    mean = np.average(means)
    for h, fn in zip(hist_list, os.listdir(fp)): 
        np.save(to_fp+fn, h/mean)
        print("Saved!", fn)


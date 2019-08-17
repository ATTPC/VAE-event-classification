import sys

sys.path.append("../src")
from make_classification_table import make_performance_table

import numpy as np
import os

vgg = False
alternate = ""
dd = True
prefix = "vgg_" if vgg else ""
prefix += "dd_" if dd else ""
dir_path = "../data/latent/clf_latent/"
if dd:
    fnames = ["static_histogram", "static_netcharge"]
x_set = []
y_set = []
for i, which in enumerate(fnames):
    fn = prefix + which + "_latent.npy"
    target_fn = prefix + which + "_targets.npy"
    x_set.append(np.load(dir_path + fn))
    y_set.append(np.load(dir_path + target_fn))

dataset_names = [
    #        ["Proton", "Carbon"],
    ["Proton", "Carbon", "Other"],
    ["Proton", "Carbon", "Other"],
]

rows = ["Histogram", "Net charge"]
df = make_performance_table(x_set, y_set, dataset_names, rows=rows)
savedir = "/home/robersol/github/thesis/chapters/results/classification/plots/"
fn = savedir + alternate + prefix + "convae_clf_table.tex"
with open(fn, "w+") as fo:
    df.to_latex(fo, escape=False)

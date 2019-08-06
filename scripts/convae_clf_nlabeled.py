import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import os 

import sys
sys.path.append("../src")
from evaluate_n_labeled import n_labeled_data

vgg = False
alternate = ""
prefix = "vgg_" if vgg else ""

x_set = np.load("../data/latent/clf_latent/" + alternate + prefix + "data_repr.npy")
y_set = np.load("../data/latent/clf_latent/targets.npy")
dataset_names = [
 ['Proton', 'Carbon', 'All'],
 ['Proton', 'Carbon', 'Other', 'All'],
 ['Proton', 'Carbon', 'Other', 'All']
 ]
means, stds = n_labeled_data(x_set, y_set, 50, dataset_names)
cm = matplotlib.cm.get_cmap("magma")
colors = [cm(0.3), cm(0.6), cm(0.85)]
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(12, 8))
data_t = ["Simulated", "Filtered", "Full"]
error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
for i in range(len(means)):
    x = [100*i for i in range(1, means[i].shape[0])]
    ax.errorbar(
        x=x,
        y=means[i][:-1],
        yerr=stds[i][:-1],
        color=colors[i],
        fmt=".",
        label=data_t[i],
        ms=10,
        **error_kw
    )
    

plt.legend(loc="best")
ax.set_xlabel("N labeled samples")
ax.set_ylabel("f1 test score ")
#ax.set_ylim((0.5, 1.2))
ticks = np.arange(0.5, 1.1, 0.1)
labels = ["{:.1f}".format(i) if i <= 1 else "" for i in ticks]
#ax.set_xticks(x)
ax.set_yticks(ticks=ticks, )
ax.set_yticklabels(labels=labels)
print(ticks)
print(x)
repo_str = "/home/robersol/github/thesis/chapters/results/classification/"
if vgg:
    fn = repo_str + "plots/"+alternate+"vgg_ac_n_samples"
else: 
    fn = repo_str + "plots/"+alternate+"ac_n_samples"

plt.savefig(fn+".png")
plt.savefig(fn+".pdf")

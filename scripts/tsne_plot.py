from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

cm = matplotlib.cm.get_cmap("magma")
vgg = False 
alternate = ""
prefix = "vgg_" if vgg else ""
x_set = np.load("../data/latent/clf_latent/" + alternate + prefix + "data_repr.npy")
targets = np.load("../data/latent/clf_latent/targets.npy")

pca_representations = []
for latent in x_set:
    pca_representations.append(PCA(50).fit_transform(
        latent.reshape((latent.shape[0], -1))
        ))

tsne_representations = []
for pca_rep in pca_representations:
    tsne_representations.append(TSNE(2, perplexity=15).fit_transform(pca_rep))
fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
class_names = ["Proton", "Carbon", "Other"]
datasets = ["Simulated", "Filtered", "Full"]
colors = [cm(0.3), cm(0.6), cm(0.85)]

for i, tsne_rep in enumerate(tsne_representations):
    if len(targets[i].shape) == 2:
        classes = targets[i].argmax(-1)
    else:
        classes = targets[i]
    for c in np.unique(classes):
        w = classes == c
        s = tsne_rep[w]
        ax[i].scatter(s[:,0], s[:,1], alpha=0.5, label=class_names[c], color=colors[c])
        ax[i].set_title(datasets[i])
    if i == 1:
        lgd = ax[i].legend(
            bbox_to_anchor=(-0.95, 1.02, 1., .102),
            markerscale=3,
            fontsize=30,
            ncol=3,
            loc="lower left",
            borderpad=2,
            frameon=False
                    )
dirname = "/home/robersol/github/thesis/chapters/results/classification/plots/"
fn = prefix + alternate + "ac_tsne"
plt.savefig(dirname + fn + ".pdf",
        bbox_extra_artists=(lgd,), bbox_inches='tight'
        )
plt.savefig(dirname + fn + ".png",
        bbox_extra_artists=(lgd,), bbox_inches='tight'
        )

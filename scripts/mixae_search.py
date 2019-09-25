#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix, accuracy_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os
from sklearn.utils.fixes import comb
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
import data_loader as dl
from batchmanager import BatchManager
from mixae import mixae_model, entropy_callback, probabilities_log
import sys
sys.path.append("../src/")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def plot_confusion_matrix(
        validation_targets,
        predicted_values,
        title="",
        true_cols=None,
        ax=None
):
    cm_int = confusion_matrix(validation_targets, predicted_values)

    if type(ax) == type(None):
        fig, ax = plt.subplots()

    ax.set_title(title, fontsize=20)
    sns.heatmap(cm_int, annot=True, ax=ax)
    if not true_cols is None:
        ax.set_yticklabels(true_cols)
    ax.set_ylabel('True label', fontsize=14)
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if type(ax) == type(None):
        return fig, ax


def display_events(model, n, n_display=3, what_class=0,):
    if what_class == 0:
        event = "Proton"
    elif what_class == 1:
        event = "Carbon"
    else:
        event = "Other"
    all_class = np.argwhere(y_lab == what_class)
    which_class = [int(all_class[i]) for i in range(n_display)]
    print(which_class)
    pred = model.predict_on_batch(x_lab_flat[tuple(which_class), :])
    lab_samples = pred[0]
    lab_probs = pred[1]

    cmap = matplotlib.cm.magma
    cmap.set_under(color='white')

    fig, ax = plt.subplots(ncols=n_classes+1, nrows=n_display, figsize=(15, 9))
    ax[0, 0].set_title(event+" events", fontsize=20)
    for i in range(n_display):
        w = which_class[i]
        ax[i][0].imshow(x_lab_flat[w].reshape(img_shape), cmap=cmap, vmin=1e-4)
        ax[i][0].axis("off")
        for j in range(n_classes):
            ax[i][j+1].imshow(lab_samples[j]
                              [i].reshape(img_shape), cmap=cmap, vmin=1e-5)
            title = r"$p_{}".format(j)
            title += r"= {:.3f}$".format(float(lab_probs[i, j]))
            ax[i][j+1].set_title(title, fontsize=14)
            ax[i][j+1].axis("off")
    base_fn = "/home/robersol/github/thesis/chapters/results/clustering/plots/"
    fn = base_fn + event + "_"+data+"_mixae_reconst_{}".format(n)
    plt.savefig(fn+".pdf")
    plt.savefig(fn+".png")
    plt.close()


# In[3]:
data = "mnist"

if data == "mnist":
    (x_full, y_tr), (x_labeled, y_lab) = tf.keras.datasets.mnist.load_data()
    x_full = (np.expand_dims(x_full, -1)/255)
    x_labeled = np.expand_dims(x_labeled, -1)/255
    y_lab_onehot = OneHotEncoder().fit_transform(y_lab.reshape(-1, 1))
    n_samples = x_full.shape[0]
elif data == "real":
    x_full, x_labeled, y_lab = dl.load_real_event("128")
    y_lab_onehot = np.copy(y_lab)
    y_lab = y_lab.argmax(1)
    n_samples = x_full.shape[0]
    img_shape = x_full.shape[1:] if len(
        x_full.shape) == 3 else x_full.shape[1:-1]
elif data == "clean":
    x_full, x_labeled, y_lab = dl.load_clean_event("128")
    y_lab_onehot = np.copy(y_lab)
    y_lab = y_lab.argmax(1)
    n_samples = x_full.shape[0]
    img_shape = x_full.shape[1:] if len(
        x_full.shape) == 3 else x_full.shape[1:-1]
else:
    x_full = np.load("../data/simulated/images/sim_images.npy")
    y_full = np.load("../data/simulated/images/sim_targets.npy")
    x_full, x_labeled, y_tr, y_lab = train_test_split(
        x_full, y_full, shuffle=True)
    n_samples = x_full.shape[0]

n_classes = len(np.unique(y_lab))
img_shape = x_full.shape[1:] if len(x_full.shape) == 3 else x_full.shape[1:-1]

n_layers = 4
latent_dim = 8
kernel_architecture = [3, 3, 3, 3]
filter_architecture = [64, 32, 16, 8, ]
strides_architecture = [2, ]*n_layers
pooling_architecture = [0, ]*n_layers

mode_config = {
    "simulated_mode": False,  # deprecated, to be removed
    "restore_mode": False,  # indicates whether to load weights
    "include_KL": False,  # whether to compute the KL loss over the latent space
    "include_MMD": False,  # same as above, but MMD
    # same as above, but K-means. See thesis for a more in-depth treatment of these
    "include_KM": False,
    "batchnorm": False,  # whether to include batch-normalization between layers
    "use_vgg": False,  # whether the input data is from a pre-trained model
    "use_dd": False,  # whether to use the dueling-decoder objective
}

reg_strength = 1e-6
ae_args = [
    [
        n_layers,
        filter_architecture,
        kernel_architecture,
        strides_architecture,
        pooling_architecture,
        latent_dim,
        x_full.shape,
    ],
    {
        "mode_config": mode_config
    },
    [],
    {
        "kernel_reg_strength": reg_strength,
        "kernel_reg": tf.keras.regularizers.l2,
        "activation": "lrelu"
        # "output_activation": "sigmoid"
    }
]

x_full_flat = x_full.reshape(n_samples, -1)
x_lab_flat = x_labeled.reshape(x_labeled.shape[0], -1)
print(x_full_flat.max(), x_full_flat.min())

earlystopping = tf.keras.callbacks.EarlyStopping(
    "batch_ent_loss",
    min_delta=1e-1,
    patience=2,
    # restore_best_weights=True,
    # baseline=0.30
)
earlystopping_pre = tf.keras.callbacks.EarlyStopping(
    "val_loss", min_delta=1e-4, patience=2)


def update_wrapper(self, func):
    def func_wrapper(epoch, logs={}):
        # print()
        # print("old min delta", self.min_delta)
        self.min_delta *= (0.96*2/3)**epoch
        # print("New min_delta", self.min_delta)
        return func(epoch, logs)
    return func_wrapper


earlystopping.update_wrapper = update_wrapper
earlystopping.on_epoch_end = earlystopping.update_wrapper(
    earlystopping, earlystopping.on_epoch_end)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

prob_logs = []
histories = []
models = []
runs = 100
repeats = 5
single = False
alphas = [0.1,  1, ]
betas = [100, 3000]
thetas = [1, 4]
parameter_comb = []
perf_list = []
max_ari = 0.2
for i in range(runs):
    alpha = np.random.uniform(alphas[0], alphas[1])
    beta = np.random.uniform(betas[0], betas[1])
    theta = 10**(np.random.randint(thetas[0], thetas[1]))
    for j in range(repeats):
        tf.keras.backend.clear_session()
        mix_obj = mixae_model(
            ae_args, alpha, beta, x_full_flat.shape[0]/theta)
        m, mp, clst, _ = mix_obj.compile(n_classes)
        pl = probabilities_log(clst, n_classes, x_lab_flat)
        n_eps = 10
        unsuper_hist = m.fit(
            x_full_flat,
            [
                np.expand_dims(x_full_flat, 1),
                np.zeros(n_samples),
                np.zeros(n_samples)
            ],
            batch_size=150,
            epochs=n_eps,
            callbacks=[
                # entropy_callback(mix_obj.alpha, mix_obj.beta),
                pl,
                earlystopping,
            ],
            verbose=0
        )
        if single:
            for j in range(n_classes):
                try:
                    display_events(m, i, what_class=j)
                except ValueError:
                    pass
        models.append(m)
        prob_logs.append(pl)
        histories.append(unsuper_hist)
        parameter_comb.append([alpha, beta, theta])
        ari_vals = []
        acc_vals = []
        for j, pred in enumerate(pl.prob_log):
            clf_pred = pred.argmax(1)
            ari_vals.append(adjusted_rand_score(y_lab, clf_pred))
            acc_vals.append(acc(y_lab, clf_pred))
        perf_list.append([max(ari_vals), max(acc_vals), parameter_comb[-1]])
        if max(ari_vals) > max_ari:
            print(max(ari_vals), max(acc_vals), parameter_comb[-1])

        fn = "../results/mixae/"+data+"/randomsearch_param.txt"
        with open(fn, "a+") as fo:
            fo.write("ari, acc, alpha, beta, theta \n")
            for p in perf_list:
                fo.write("{},".format(p[0]))
                fo.write("{},".format(p[1]))
                for param in p[2]:
                    fo.write("{},".format(param))
                fo.write("\n")

        if max(ari_vals) < max_ari and i == 0:
            break

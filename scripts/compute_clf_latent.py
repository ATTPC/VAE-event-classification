import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append("../src/")
from convolutional_VAE import ConVae
import data_loader as dl

print("PID", os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
vgg = False
dd = True
prefix = "vgg_" if vgg else ""
prefix += "dd_" if dd else ""
if vgg:
    alternate = ""
    data_loaders = [
        dl.load_vgg_simulated,
        dl.load_clean_vgg_event,
        dl.load_real_vgg_event,
    ]
    base_str = "../results/randomsearch_convae_"
    hyperparameters = [
        np.load(base_str + "vgg_simulated_128_clf/run_165/hyperparam_vals_ours.npy")[
            204
        ],
        np.load(base_str + "vgg_cleanevent_128_clf/run_159/hyperparam_vals_ours.npy")[
            73
        ],
        np.load(base_str + "vgg_realevent_128_clf/run_146/hyperparam_vals_ours.npy")[
            210
        ],
    ]
    if alternate == "tmp":
        hyperparameters[1] = hyperparameters[2].copy()
if dd:
    alternate = ""
    base_str = "../results/randomsearch_convae_realevent_128_clf/"
    data_loaders = [dl.load_real_event, dl.load_real_event]
    hyperparameters = [
        np.load(base_str + "/run_175/hyperparam_vals_static.npy")[7],
        np.load(base_str + "/run_188/hyperparam_vals_static.npy")[31],
    ]
    dense_layers = [2, 1]
    dd_target_loaders = [dl.load_realevent_hist, dl.load_realevent_netcharge]
    filenames = ["static_histogram", "static_netcharge"]

else:
    alternate = ""
    data_loaders = [dl.load_simulated, dl.load_clean_event, dl.load_real_event]
    base_str = "../results/randomsearch_convae_"
    hyperparameters = [
        np.load(base_str + "simulated_128_clf/run_96hyperparam_vals_ours.npy")[13],
        np.load(base_str + "cleanevent_128_clf/run_160/hyperparam_vals_ours.npy")[84],
        np.load(base_str + "realevent_128_clf/run_133/hyperparam_vals_ours.npy")[116],
    ]
    if alternate == "tmp":
        hyperparameters[1] = hyperparameters[2].copy()

which = [0, 1]
x_set = []
y_set = []
for i in range(len(data_loaders)):
    if not i in which:
        continue
    epochs = 2000
    batch_size = 150
    config = hyperparameters[i]
    conv_config = config[0]
    filter_architecture = conv_config[0]
    kernel_architecture = conv_config[1]
    strides_architecture = conv_config[2]
    pooling_config = conv_config[3]
    n_layers = conv_config[4]

    parameters_config = config[1]
    beta = parameters_config[0]
    eta = parameters_config[1]
    beta1 = parameters_config[2]
    beta2 = parameters_config[3]
    latent_dim = parameters_config[4]
    # sampling_dim = parameters_config[5]

    mode_config = config[2]
    clustering_config = config[3]

    parameters_config = config[1]
    beta = parameters_config[0]
    eta = parameters_config[1]
    beta1 = parameters_config[2]
    beta2 = parameters_config[3]
    latent_dim = parameters_config[4]
    print("Configuration")
    print("Conv params :")
    print(filter_architecture)
    print(kernel_architecture)
    print(strides_architecture)
    print(pooling_config)
    print("Mode config: ")
    print(mode_config)
    print("Parameters:")
    print(beta, latent_dim, eta, beta1, beta2)

    loader = data_loaders[i]
    if vgg:
        x_train, target_imgs, x_test, y_test = loader("128")
        mode_config["use_vgg"] = True
    else:
        x_train, x_test, y_test = loader("128")
        target_imgs = None
        mode_config["use_vgg"] = False
    y_set.append(y_test)
    cvae = ConVae(
        n_layers,
        filter_architecture,
        kernel_architecture,
        strides_architecture,
        pooling_config,
        latent_dim,
        x_train,
        beta=beta,
        mode_config=mode_config,
        clustering_config=clustering_config,
        labelled_data=[x_test, y_test],
        target_imgs=target_imgs,
    )
    if vgg:
        if dd:
            dense = parameters_config[-2]
        else:
            dense = parameters_config[-1]
        cvae.dense_layers = dense
    if dd:
        dd_train, dd_test = dd_target_loaders[i]("128")
        cvae.dd_targets = dd_train
        cvae.lmbd = parameters_config[-1]
        cvae.dd_dense = dense_layers[i]
    loss = config[-2]
    if loss is None:
        out_act = "sigmoid"
    else:
        out_act = None
    activation = config[-1]
    print("Loss and act params : ")
    print(loss, activation, out_act)
    graph_kwds = {"activation": activation, "output_activation": out_act}
    loss_kwds = {"reconst_loss": loss}

    opt = tf.train.AdamOptimizer
    opt_args = [eta]
    opt_kwds = {"beta1": beta1}

    cvae.compile_model(graph_kwds, loss_kwds)
    cvae.compute_gradients(opt, opt_args, opt_kwds)
    sess = tf.InteractiveSession()
    lx, lz = cvae.train(
        sess, epochs, batch_size, earlystopping=True, save_checkpoints=0, verbose=1
    )
    x_set.append(cvae.run_large(sess, cvae.z_seq[0], x_test))
    sess.close()
    print()
    print(x_set[-1].shape)

dir_path = "../data/latent/clf_latent/"
data_suff = "_latent.npy"
target_suff = "_targets.npy"

for i, which in enumerate(filenames):
    fn = prefix + which + data_suff
    target_fn = prefix + which + target_suff
    np.save(dir_path + fn, x_set[i])
    np.save(dir_path + target_fn, y_set[i])

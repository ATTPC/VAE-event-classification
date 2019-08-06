import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append("../src/")
from convolutional_VAE import ConVae
from data_loader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
vgg = False
if vgg:
    alternate=""
    data_loaders = [
        load_vgg_simulated,
        load_clean_vgg_event, 
        load_real_vgg_event,
    ]
    base_str = "../results/randomsearch_convae_"
    hyperparameters = [
            np.load(base_str+"vgg_simulated_128_clf/run_165/hyperparam_vals_ours.npy")[300],
            np.load(base_str+"vgg_cleanevent_128_clf/run_159/hyperparam_vals_ours.npy")[73],
            np.load(base_str+"vgg_realevent_128_clf/run_146/hyperparam_vals_ours.npy")[113],
            ]
    if alternate == "tmp":
        hyperparameters[1] = hyperparameters[2].copy()
else:
    alternate=""
    data_loaders = [
        load_simulated,
        load_clean_event, 
        load_real_event,
    ]
    base_str = "../results/randomsearch_convae_"
    hyperparameters = [
            np.load(base_str+"simulated_128_clf/run_96hyperparam_vals_ours.npy")[13],
            np.load(base_str+"cleanevent_128_clf/run_160/hyperparam_vals_ours.npy")[169],
            np.load(base_str+"realevent_128_clf/run_133/hyperparam_vals_ours.npy")[25],
            ]
    if alternate == "tmp":
        hyperparameters[1] = hyperparameters[2].copy()

x_set = []
y_set = []
for i in range(len(data_loaders)):
    epochs = 2
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
    #sampling_dim = parameters_config[5]

    mode_config = config[2]
    clustering_config = config[3]

    parameters_config = config[1]
    beta = parameters_config[0]
    eta = parameters_config[1]
    beta1 = parameters_config[2]
    beta2 = parameters_config[3]
    latent_dim = parameters_config[4]

    dl = data_loaders[i]
    if vgg:
        x_train, target_imgs, x_test, y_test = dl("128")
        mode_config["use_vgg"] = True
    else:
        x_train, x_test, y_test = dl("128")
        x_train = x_train[:200]
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
            beta=10,
            mode_config=mode_config,
            clustering_config=clustering_config,
            labelled_data=[x_test, y_test],
            target_imgs=target_imgs
            )
    if vgg:
        cvae.dense_layers = parameters_config[-1]
    loss = config[-2]
    if loss is None:
        out_act = "sigmoid"
    else:
        out_act = None
    activation = config[-1]
    graph_kwds = {"activation": activation ,"output_activation": out_act}
    loss_kwds = {"reconst_loss": loss}
    cvae.compile_model(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [eta, ]
    opt_kwds = {
        "beta1": beta1
    }

    cvae.compute_gradients(opt,)
    sess = tf.InteractiveSession() 
    lx, lz = cvae.train(
            sess,
            epochs,
            batch_size,
            earlystopping=True,
            save_checkpoints=0,
            verbose=1
            )
    x_set.append(cvae.run_large(sess, cvae.z_seq[0], x_test))
    sess.close()

x_set = np.array(x_set)
y_set = np.array(y_set)
dir_path = "../data/latent/clf_latent/"
prefix = "vgg_" if vgg else ""
fn = alternate + prefix + "data_repr.npy"
target_fn = "targets.npy"
np.save(dir_path + fn, x_set)
np.save(dir_path + target_fn, y_set)

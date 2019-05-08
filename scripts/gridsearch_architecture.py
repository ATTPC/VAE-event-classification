
import sys
sys.path.append("../src")

import numpy as np
import tensorflow as tf

from sklearn.model_selection import  cross_val_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import h5py

import os

from draw import DRAW
from hypperparam_generator import make_hyperparam

from counter import *

print("PID :", os.getpid())

def longform_latent(latent,):
    longform_samples = np.zeros((
                        latent.shape[1], 
                        latent.shape[0]*latent.shape[2]
                        ))

    latent_dim = latent.shape[2]

    for i, evts in enumerate(latent):
        longform_samples[:, i*latent_dim:(i+1)*latent_dim] = evts

    return longform_samples


def compute_accuracy(X, y, Xtest, ytest):

    model = LogisticRegression(
            penalty="l2",
            solver="newton-cg",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=1000,
        )

    mlp = MLPClassifier(
            max_iter=1000,
            batch_size=100,
            hidden_layer_sizes=(80, 30, 10),
            early_stopping=True,
            learning_rate_init=0.001
        )

    model.fit(X, y)
    mlp.fit(X, y)

    train_score_lr = f1_score(y, model.predict(X), average=None)
    test_score_lr = f1_score(ytest, model.predict(Xtest), average=None)
    lr_scores = (train_score_lr, test_score_lr)

    train_score_mlp = f1_score(y, mlp.predict(X), average=None)
    test_score_mlp = f1_score(ytest, mlp.predict(Xtest), average=None)
    mlp_scores = (train_score_mlp, test_score_mlp)

    return lr_scores, mlp_scores


delta = 0.98
N = 65
batch_size = 150
enc_size = 500 

num_simulations = 100
epochs = 20

read_N = N
write_N = N

delta_write = delta
delta_read = delta

treshold_value = 0.4
treshold_data = False

all_0130 = np.load("../data/processed/all_0130.npy")
train_data = np.load("../data/processed/train.npy")
test_data = np.load("../data/processed/test.npy")

with h5py.File("../data/images.h5", "r") as fo:
    train_targets = np.array(fo["train_targets"])
    test_targets = np.array(fo["test_targets"])

#all_0210 = np.load("../data/processed/all_0210.npy")
#all_data = np.concatenate([all_0130, all_0210])

all_data = all_0130

data_ind = np.arange(all_data.shape[0])
chosen_in = np.random.choice(data_ind, 20000, replace=False)
all_data = all_data[chosen_in]

if treshold_data:
	all_data[all_data < treshold_value] = 0

hyperparam_vals = [
        "T, dec_state_size, latent_dim, loss_type, rw_type, \
        n_enc, n_dec, beta", ]

loss_record = np.zeros((num_simulations, 2, epochs))
clf_record = np.zeros((num_simulations, 2, 2, 3))

hp_gen = make_hyperparam()

for i in range(num_simulations):

    T, state_size, latent_dim, loss_type, rw_type, n_enc, n_dec, beta = hp_gen.generate_config()
    hyperparam_vals.append([T, state_size, latent_dim, loss_type, rw_type, n_enc, n_dec, beta])

    use_attention = False
    use_conv = False

    if rw_type == "attention":
        use_attention = True
    if rw_type =="conv":
        use_conv = True

    use_kl = False
    use_MMD = False

    if loss_type == "MMD":
        use_MMD = True
    elif loss_type == "KL":
        use_kl = True

    dec_size = state_size

    attn_config = {
                "read_N": read_N,
                "write_N": write_N,
                "write_N_sq": write_N**2,
                "delta_w": delta,
                "delta_r": delta,
            }

    if not use_attention:
        attn_config = None

    mode_config = {
            "simulated_mode": False,
            "restore_mode": False,
            "include_KL": use_kl,
            "include_MMD": use_MMD
            }

    draw_model = DRAW(
            T,
            dec_size,
            enc_size,
            latent_dim,
            all_data,
            beta=beta,
            attn_config=attn_config,
            mode_config=mode_config,
            use_conv=use_conv,
            use_attention=use_attention
            )

    graph_kwds = {
            "initializer": tf.initializers.glorot_normal,
            "n_encoder_cells": n_enc,
            "n_decoder_cells": n_dec,
            }

    loss_kwds = {
            "reconst_loss": None,
            }

    draw_model.CompileModel(graph_kwds, loss_kwds)

    opt = tf.train.AdamOptimizer
    opt_args = [1e-3,]
    opt_kwds = {
            "beta1": 0.5,
            }

    draw_model.computeGradients(opt, opt_args, opt_kwds)

    sess = tf.InteractiveSession()

    data_dir = "../drawing"
    model_dir = "../models"

    lx, lz, = draw_model.train(
                            sess,
                            epochs,
                            data_dir,
                            model_dir,
                            batch_size,
                            save_checkpoints=False)

    if any(np.isnan(lx)) or any(np.isnan(lz)):
            continue

    print()
    loss_record[i][0] = lx
    loss_record[i][1] = lz

    draw_model.X = train_data

    latent_values , _, _ = draw_model.generate_latent(sess, "../drawing", (train_data, test_data), save=False)
    latent_train = longform_latent(latent_values[0])
    latent_test = longform_latent(latent_values[1])

    lr_scores, mlp_scores = compute_accuracy(
                                latent_train, train_targets,
                                latent_test, test_targets
                                )


    print("--------------------")
    print("Dec size: ", dec_size, " Latent dim : ", latent_dim, " loss : ", loss_type)
    print("rw_type: ", rw_type, " T :", T, "beta : ", beta )
    print("####### TRAIN scores #######")
    print("logreg_score", lr_scores[0], "mlp_scores", mlp_scores[0])
    print("####### TEST scores #######")
    print("logreg_score", lr_scores[1], "mlp_scores", mlp_scores[1])
    print("---------------------")

    clf_record[i][0] = lr_scores
    clf_record[i][1] = mlp_scores
    sess.close()

    np.save("../loss_records/architecture_opt{}.npy".format(RUN_COUNT) ,loss_record)
    np.save("../loss_records/arch_hyperparam_vals_{}.npy".format(RUN_COUNT),hyperparam_vals)
    np.save("../loss_records/arch_score_vals{}.npy".format(RUN_COUNT), clf_record)

with open("counter.py", "w") as fo:
    fo.write("RUN_COUNT = {}".format(RUN_COUNT + 1))

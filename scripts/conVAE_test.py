import sys
sys.path.append("../src")

from convolutional_VAE import ConVae
import numpy as np
import keras as ker
import os


print("PID: ", os.getpid())
input_dim = (128, 128, 1)
X = np.load("../data/processed/all_0130.npy")

n_layers = 4 
filter_architecture = [20, 40, 10, 5]
kernel_arcitecture = [7, 5, 3, 2]
strides_architecture = [4, 2, 2, 1]

latent_dim = 3
batch_size = 50

earlystop = ker.callbacks.EarlyStopping(
                    monitor='loss',
                    min_delta=0.1,
                    patience=0,
                    verbose=0,
                    mode='auto',
                    restore_best_weights=True
                )

callbacks = [earlystop, ]


cvae = ConVae(
        input_dim,
        n_layers,
        filter_architecture,
        kernel_arcitecture,
        strides_architecture,
        latent_dim,
        batch_size,
        X,
        )

cvae.CompileModel()
cvae.CompileLoss()
cvae.train(epochs=100, callbacks=callbacks)


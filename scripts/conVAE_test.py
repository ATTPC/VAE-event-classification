import sys
sys.path.append("../src")

from convolutional_VAE import ConVae
import numpy as np


input_dim = (128, 128, 1)
X = np.zeros((500, 128, 128, 1))

n_layers = 2 
filter_architecture = [50, 40]
kernel_arcitecture = [7, 5,]
strides_architecture = [4, 2]

latent_dim = 3
batch_size = 50

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
cvae.train()

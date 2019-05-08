import numpy as np
import random

class make_hyperparam:

    def __init__(self):

        self.T = np.arange(2, 44, 4)

        self.dec_state_sizes = np.arange(100, 2500, 100)
        #self.dec_state_sizes = np.arange(2, 20, 10)
        self.latent_sizes = np.arange(5, 150, 10)
        #self.latent_sizes = np.arange(2, 4, 1)

        self.losses = ["MMD", "KL", ]
        self.read_write = ["attention", "conv", ]

        self.n_enc_cells = np.arange(1, 2)
        self.n_dec_cells = np.arange(1, 2)

        self.betas = np.arange(0, 4)
        self.betas = 10**self.betas

    def generate_config(self,):

        T = np.random.choice(self.T)
        dec_state_size = np.random.choice(self.dec_state_sizes)    
        latent_dim = np.random.choice(self.latent_sizes)
        loss = random.choice(self.losses)
        rw_method = random.choice(self.read_write)

        n_enc = np.random.choice(self.n_enc_cells)
        n_dec = np.random.choice(self.n_dec_cells)

        beta  = np.random.choice(self.betas)

        return T, dec_state_size, latent_dim, loss, rw_method, n_enc, n_dec, beta

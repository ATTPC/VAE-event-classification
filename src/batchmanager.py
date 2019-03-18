import numpy as np


class BatchManager:

    def __init__(self, nSamples, batch_size, n_input):
        self.available = np.arange(0, nSamples)
        self.nSamples = nSamples

        self.batch_size = batch_size
        self.n_input = n_input

    def FetchMinibatch(self, ):
        tmp_ind = np.random.randint(0, self.nSamples, size=(batch_size, ))
        ind = self.available[tmp_ind]

        self.available = np.delete(self.available, tmp_ind)
        self.nSamples = len(self.available)
        
        return ind
        #return self.X[ind].reshape(self.batch_size, self.n_pixels)

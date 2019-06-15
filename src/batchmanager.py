import numpy as np


class BatchManager:

    def __init__(self, nSamples, batch_size):
        self.available = list(np.arange(0, nSamples))
        self.nSamples = nSamples
        self.batch_size = batch_size

    def __next__(self, ):

        if self.nSamples == 0:
            raise StopIteration()

        if self.nSamples > self.batch_size:
            ind = np.random.choice(self.available, self.batch_size, replace=False)

        else: 
            ind = np.random.choice(self.available, self.nSamples, replace=False)
            

        for i in ind:
            self.available.remove(i)

        self.nSamples = len(self.available)
        return ind
        
        #return self.X[ind].reshape(self.batch_size, self.n_pixels)


    def __iter__(self, ):
        return self



if __name__ == "__main__":
    import sys
    
    n_samp = 10 
    batch_size = 3
    itercount = 0

    bm = BatchManager(n_samp, batch_size)

    for  batch_ind in bm:

        print("-------------####-------------")
        
        itercount += 1
        if itercount > 10:
            sys.exit(0)

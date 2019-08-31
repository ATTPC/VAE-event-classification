import numpy as np

def make_histograms(x, bins=40, interval=[1e-1, 1]):
    intervals = np.linspace(interval[0], interval[1], bins)
    flat_x = x.reshape((x.shape[0], -1))
    hist_x = np.zeros((x.shape[0], bins))
    for i in range(1, bins):
        mask = flat_x <= intervals[i]
        mask = np.logical_and(mask, flat_x > intervals[i-1])
        hist_x[:, i] = mask.sum(1)

    return hist_x

def make_net_count(x, **kwargs):
    flat_x = x.reshape((x.shape[0], -1))
    sum_x = flat_x.sum(1)
    return sum_x

import numpy as np
import cv2
import matplotlib.pyplot as plt

train_data = np.load("../data/processed/train.npy")

i = 1
which = [10+i, 1+i, 3+i, 4+i, 13+i, 20+i, ]
s = np.squeeze(train_data[which]).astype(np.float64)
s_dim = s.shape

fig, axs = plt.subplots(nrows=len(which), ncols=2, figsize=(10, 10))


for i, row in enumerate(axs):
    source = (255*s[i]/s[i].max()).astype(np.uint8)
    source = cv2.resize(source, None, fx=0.35, fy=0.35)

    ds = cv2.fastNlMeansDenoising(
        source, None, 17, 21, 7)

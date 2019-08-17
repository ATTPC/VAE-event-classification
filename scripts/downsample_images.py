import numpy as np
import cv2

base = "../data/"
data_locations = [
    base + "processed/train.npy",
    base + "processed/all_0130.npy",
    base + "processed/test.npy",
]

perc = 0.35

for loc_fn in data_locations:
    data = np.load(loc_fn)
    s = np.squeeze(data)

    w = s.shape[1]
    h = s.shape[2]

    to = np.zeros((s.shape[0], int(np.round(w * perc)), int(np.round(h * perc)), 1))

    for i in range(s.shape[0]):
        source = s[i]
        max_val = source.max()
        source = (255 * source / max_val).astype(np.uint8)
        source = cv2.resize(source, None, fx=perc, fy=perc)

        to[i] = np.expand_dims((source * max_val / 255).astype(np.float64), -1)

    to_fn = loc_fn[:3] + loc_fn[3:].replace(".", "_{}.".format(perc * 100))
    print(to_fn)
    np.save(to_fn, to)

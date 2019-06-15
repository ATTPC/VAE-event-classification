from sklearn.model_selection import train_test_split
import numpy as np

size = 80
runs = ["0130", "0210"]
data = []
targets = []

for r in runs:
    data.append(np.load("../data/clean/images/run_{}_label_True_size_{}.npy".format(r, size)))
    targets.append(np.load("../data/clean/targets/run_{}_targets.npy".format(r, size)))

data = np.concatenate(data)
targets = np.concatenate(targets)


train_x, test_x, train_y, test_y = train_test_split(data, targets, test_size=0.2)
np.save("../data/clean/images/train_size_{}.npy".format(size), train_x)
np.save("../data/clean/images/test_size_{}.npy".format(size), test_x)

np.save("../data/clean/targets/train_targets_size_{}.npy".format(size), train_y)
np.save("../data/clean/targets/test_targets_size_{}.npy".format(size), test_y)

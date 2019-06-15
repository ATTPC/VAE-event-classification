import numpy as np
from sklearn.model_selection import train_test_split

targets_0130 = np.load("../data/clean/targets/run_0130_targets.npy")
targets_0210 = np.load("../data/clean/targets/run_0210_targets.npy")
all_targets = np.concatenate([targets_0130, targets_0210])

labeled_0130 = np.load("../data/clean/images/run_0130_label_True.npy")
labeled_0210 = np.load("../data/clean/images/run_0210_label_True.npy")
all_labeled = np.concatenate([labeled_0130, labeled_0210])

train_data, test_data, train_targets, test_targets = train_test_split(
                                                        all_labeled,
                                                        all_targets,
                                                        test_size=0.2
                                                        )
np.save("../data/clean/images/train.npy", train_data)
np.save("../data/clean/images/test.npy", test_data)
np.save("../data/clean/targets/train_targets.npy", train_targets)
np.save("../data/clean/targets/test_targets.npy", test_targets)

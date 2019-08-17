import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, make_scorer

pd.set_option("display.max_colwidth", -1)


class f1_class_n:
    def __init__(self, which, labels):
        self.which = which
        self.labels = labels

    def __call__(self, model, x, y_true):
        y_pred = model.predict(x)
        return f1_score(y_true, y_pred, average=None, labels=self.labels)[self.which]


def compute_oos_performance(perf_dict, classes):
    df = pd.DataFrame()
    train_means = {}
    train_stds = {}
    test_means = {}
    test_stds = {}
    for key, item in perf_dict.items():
        if "test" in key:
            class_ = key.split("_")[1]
            test_means[class_] = item.mean()
            test_stds[class_] = item.std()
            print(class_)
        elif "train" in key:
            class_ = key.split("_")[1]
            train_means[class_] = item.mean()
            train_stds[class_] = item.std()
    for l in [train_means, train_stds, test_means, test_stds]:
        vals = list(l.values())
        l["All"] = np.mean(vals)
    return train_means, train_stds, test_means, test_stds, classes


def model(x, y, classes):
    cv = 5
    perf_dict = cross_validate(
        LogisticRegression(
            class_weight="balanced",
            multi_class="ovr",
            solver="newton-cg",
            max_iter=100000,
            penalty="l2",
        ),
        x,
        y,
        cv=cv,
        scoring={
            "{}".format(classes[n]): f1_class_n(n, np.arange(len(classes)))
            for n in range(len(classes))
        },
        n_jobs=cv,
    )
    return perf_dict


def make_performance_table(x_set, y_set, dataset_names, rows=None, columns=None):
    if columns is None:
        columns = ["Proton", "Carbon", "Other", "All"]
    if rows is None:
        rows = ["Simulated", "Filtered", "Full"]
    perf_array = np.zeros((len(rows), len(columns)), dtype=object)
    for i in range(len(rows)):
        classes = dataset_names[i]
        x = x_set[i]
        x = x if len(x.shape) == 2 else x.reshape((x.shape[0], -1))
        y = y_set[i]
        y = y if len(y.shape) == 1 else y.argmax(-1)
        print(x.shape)
        perf_dict = model(x, y, classes)
        classes.append("All")
        train_means, train_stds, test_means, test_stds, classes = compute_oos_performance(
            perf_dict, classes
        )
        for j in range(4):
            if len(test_means) == 3 and j == 3:
                continue
            # print(test_means)
            mean = test_means[classes[j]]
            std = test_stds[classes[j]]
            perf_str = r"$\underset{{\num{{+- {:.3e} }}  }}{{\num{{ {:.3g} }} }}$".format(
                std, mean
            )
            if len(test_means) == 3 and j == 2:
                perf_array[i, j] = "N/A"
                perf_array[i, j + 1] = perf_str
            else:
                perf_array[i, j] = perf_str
    df = pd.DataFrame(perf_array, columns=columns, index=rows)
    return df

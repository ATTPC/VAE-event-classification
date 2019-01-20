#!/usr/bin/env python3
import h5py

def DataLoader(file_location):
    fileobj = h5py.File(file_location, "r")
    X_t = fileobj["train_features"]
    y_t = fileobj["train_targets"]

    X_v = fileobj["test_features"]
    y_v = fileobj["test_targets"]

    return X_t, y_t, X_v, y_v

if __name__ == "__main__":

    file_location = "/home/solli-comphys/github/VAE-event-classification/data/real/packaged/x-y/proton-carbon-junk-noise.h5"
    a = DataLoader(file_location)

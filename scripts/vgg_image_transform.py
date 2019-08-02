import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
import sys
import os

print("PID: ", os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def vgg_model(input_dim):
    input_layer = Input(shape=input_dim)
    vgg = VGG16(include_top=False, input_tensor=input_layer)
    #o = Flatten()(vgg.output)
    model = Model(inputs=input_layer, outputs=vgg.output)
    for l in model.layers:
        l.trainable = False
    return model

data = ["simulated/", "real/", "clean/"]
size = "128"
base_fp = "../data/"
model = vgg_model((int(size), int(size), 3))

for d in data:
    fp = base_fp+d+"images/"
    to_fp = base_fp+d+"vgg_images/"
    finished_files = os.listdir(to_fp)
    for fn in os.listdir(fp):
        if size in fn or "simulated" in d:
            if not fn in finished_files:
                imgs = np.load(fp+fn)
                input_imgs = np.concatenate([imgs, imgs, imgs], axis=-1)
                print("Loaded: ")
                print(fn)
                output_imgs = model.predict(input_imgs)
                np.save(to_fp+fn, output_imgs)
                print("Saved!")


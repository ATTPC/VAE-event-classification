from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from keras import optimizers
from keras import callbacks
from keras import regularizers
from keras import utils

from keras.datasets import mnist

import numpy as np

input_size = (15, 3, 1)
# input_size = (28, 28, 1)
n_classes = 2

batch_size = (1,)

input_layer = Input(shape=input_size, batch_shape=batch_size + input_size)

c = Conv2D(
    filters=4, kernel_size=[2, 2], strides=1, padding="valid", activation="tanh"
)(input_layer)

c = Conv2D(filters=4, kernel_size=[2, 2], strides=1, padding="same", activation="relu")(
    c
)

c = Flatten()(c)

h = Dense(10, activation="relu", activity_regularizer=regularizers.l2(0.001))(c)

o = Dense(n_classes, activation="softmax")(h)

model_obj = Model(inputs=input_layer, outputs=o)
print(model_obj.summary())

optimizer = optimizers.Adam(lr=0.01, clipnorm=5.0)

model_obj.compile(optimizer, loss="binary_crossentropy", metrics=["accuracy"])


original_latent = np.load("../drawing/simulated/latent/train_latent.npy")
train_targets = np.load("../data/simulated/train_targets.npy")

data = original_latent.reshape((5600, -1))
# data = (data-np.average(data))/np.std(data)

# targets = OneHotEncoder(sparse=False).fit_transform(train_targets.reshape((-1, 1)))

print("--- DATA METRICS -----")
print(data.shape, train_targets.shape)
print(np.average(data), np.std(data))

print("----------------------")

X_train, X_test, y_train, y_test = train_test_split(data, train_targets)
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

X_train = X_train.reshape((-1, input_size[0], input_size[1], 1))
X_test = X_test.reshape((-1, input_size[0], input_size[1], 1))


progbar = callbacks.ProgbarLogger()
earlystop = callbacks.EarlyStopping(min_delta=0.01, patience=0)

callbacks = [progbar, earlystop]


model_obj.fit(
    x=X_train,
    y=y_train,
    batch_size=batch_size[0],
    epochs=100,
    callbacks=callbacks,
    validation_data=(X_test, y_test),
)

print(model_obj.predict(X_train[0:3], batch_size=3))

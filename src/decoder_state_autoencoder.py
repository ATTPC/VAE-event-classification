import numpy as np
from keras.utils import Sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Dropout
from keras import optimizers
from keras import regularizers
from keras import constraints

batch_size = 50
latent_dim = 3
epochs = 1
latent_repr = False

if latent_repr:
    train_fn = "../drawing/simulated/latent/train_latent.npy"
    test_fn = "../drawing/simulated/test_decoder_states.npy"
else:
    train_fn = "../drawing/simulated/train_decoder_states.npy"

X_train = np.load(train_fn)
train_shape = X_train.shape

if not latent_repr:
    X_train = np.squeeze(X_train[:, 1, :, :])

original_dim = X_train.shape
X_train = X_train.reshape((original_dim[0]*original_dim[1], original_dim[2]))
print(np.std(X_train), )

if latent_repr:
    encoding_architecture = [60, 40, 30, 10]
else:
    encoding_architecture = [1200, 600, 100, 50]

x = Input(batch_shape=(batch_size, X_train.shape[1]))
h = Dense(encoding_architecture[0], activation="tanh",
        #kernel_constraint=constraints.MinMaxNorm(
        #    min_value=encoding_architecture[0]/10, 
        #    max_value=encoding_architecture[0]*5,
        #    rate=1.0,
        #    axis=0),
        kernel_regularizer = regularizers.l1(0.5),
        )(x)

for layer in encoding_architecture[1:]:
    h = Dense(layer, activation="tanh",
                # kernel_constraint=constraints.MinMaxNorm(min_value=layer/10, max_value=layer*5, rate=1.0, axis=0),
                kernel_regularizer=regularizers.l1(0.5),
            )(h)

embedded = Dense(latent_dim, activation="linear", 
        use_bias=False,
        #kernel_constraint=constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=0),
            )(h)

encoder = Model(x, embedded)

d = Dense(encoding_architecture[-1], activation="selu",
        #kernel_constraint=constraints.MinMaxNorm(
        #    min_value=encoding_architecture[-1]/10,
        #    max_value=encoding_architecture[-1]*5,
        #    rate=1.0,
        #    axis=0),
        kernel_regularizer = regularizers.l1(0.5),
        )(embedded)

decoding_architecture = encoding_architecture[:-1]
decoding_architecture.reverse()

for layer in decoding_architecture:
    d = Dense(layer, activation="tanh",
        #kernel_constraint=constraints.MinMaxNorm(min_value=layer/10, max_value=layer*5, rate=1.0, axis=0),
        kernel_regularizer=regularizers.l1(0.5),
            )(d)


out = Dense(X_train.shape[1], activation="linear")(d)

autoencoder = Model(x, out)

sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, clipvalue=5)
adadelta = optimizers.Adadelta(clipvalue=5)

autoencoder.compile(optimizer=adadelta, loss="mean_squared_error")

noised_X = X_train + np.random.normal(loc=0, scale=0.01, size=X_train.shape)
print(noised_X.shape)

autoencoder.fit(noised_X, X_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_train, X_train)
        )

encoded_states = encoder.predict(noised_X)
encoded_states = encoded_states.reshape((original_dim[0], original_dim[1], latent_dim))

if latent_repr:
    np.save("../data/latent_encoded_states.npy", encoded_states)
    encoder.save("../models/latent_state_encoder.h5")
else:
    np.save("../data/decoder_encoded_states.npy", encoded_states)
    encoder.save("../models/decoder_state_encoder.h5")

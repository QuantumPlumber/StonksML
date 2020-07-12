import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

# from ..DataHandling import DataGenerator

minutes_in_trading_day = 60 * 6.5
sequence_minutes = 30
prediction_minutes = 1
print(prediction_minutes)
features = 1  # open, high, low, close, volume, time
predictions = 2  # open, high, low, close,

SandP_input = tf.keras.layers.Input(shape=(sequence_minutes, features), name='SandP_input')

net = tf.keras.layers.Flatten(name='flatten_input')(SandP_input)

num_nets = 8
for i in np.arange(num_nets):
    net = tf.keras.layers.Dense(units=(num_nets + 1 - i) * 30 * prediction_minutes * features, activation='tanh',
                                name='dense_{}'.format(i))(net)

net = tf.keras.layers.Dense(units=prediction_minutes * predictions, activation=None, name='dense_out')(net)
out = tf.keras.layers.Reshape((prediction_minutes, predictions), name='output')(net)

keras_model = tf.keras.Model(inputs=SandP_input, outputs=out)


def loss_fn(y_truth, y_pred):
    centroid = y_pred[:, :, 0:1]
    #sigma = y_pred[:, :, 1:2]
    sigma = y_truth[:, :, 0:1] / 1

    arg_error = ((centroid - y_truth[:, :, 0:1]) ** 2 / (2 * sigma ** 2))
    gauss_error = -tf.math.exp(-arg_error)
    centroid_error = (centroid - y_truth[:, :, 0:1]) ** 2
    #sigma_error = tf.math.abs((2 * sigma ** 2) - ((centroid - y_truth[:, :, 0:1]) ** 2))

    total_error = gauss_error

    # debug test function (simpler)
    # gaussian_error = 1 - tf.math.exp(- (sigma - y_truth[:, :, 0:1]) ** 2)
    return total_error


sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
# RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# Adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)

keras_model.compile(optimizer=sgd,
                    loss={'output': loss_fn},
                    loss_weights=[1.])

# direct = './multilayer_cnn_2'
direct = '.'
filename = direct + '/' + 'Dense_Model.png'
plot_model(keras_model, to_file=filename, show_shapes=True)

filename = direct + '/' + 'Dense_Model.json'
f = open(filename, 'w')
f.write(keras_model.to_json())
f.close()

keras_model.summary()

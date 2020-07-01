import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

# from ..DataHandling import DataGenerator

num_stocks = 500
minutes_in_trading_day = 60 * 6.5
sequence_minutes = 30
features = 5 + 1  # open, high, low, close, time
input_vector_size = num_stocks * features  # the size of the flattened input data

SandP_input = tf.keras.layers.Input(shape=(sequence_minutes, features), name='SandP_input')

lstm_width = 100
lstm_1 = tf.keras.layers.LSTM(units=lstm_width, return_sequences=True, name='lstm_1')
lstm_2 = tf.keras.layers.LSTM(units=lstm_width, return_sequences=True, name='lstm_2')
lstm_3 = tf.keras.layers.LSTM(units=lstm_width, return_sequences=False, name='lstm_3')
dense_2 = tf.keras.layers.Dense(units=30, activation='relu', name='dense')
reshape_1 = tf.keras.layers.Reshape((5, 6), name='output')

net = lstm_1(SandP_input)
net = lstm_2(net)
net = lstm_3(net)
net = dense_2(net)
out = reshape_1(net)

RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# Adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)

keras_model = tf.keras.Model(inputs=SandP_input, outputs=out)

keras_model.compile(optimizer=RMSprop,
                    loss={'output': 'mean_squared_error'},
                    loss_weights=[1.])

# direct = './multilayer_cnn_2'
direct = '.'
filename = direct + '/' + 'model.png'
plot_model(keras_model, to_file=filename, show_shapes=True)

filename = direct + '/' + 'model.json'
f = open(filename, 'w')
f.write(keras_model.to_json())
f.close()

keras_model.summary()
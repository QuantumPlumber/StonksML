import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

from ..DataHandling import DataGenerator

DataGenerator.dataset_creator()

num_stocks = 500
minutes_in_trading_day = 60*6.5
sequence_minutes = 30
features = 5+1  # open, high, low, close, time
input_vector_size = num_stocks*features # the size of the flattened input data

SandP_input = tf.keras.layers.Input(shape=(sequence_minutes, input_vector_size), name='SandP_input')

lstm_1 = tf.keras.layers.LSTM(units=sequence_minutes, name='lstm')
dense_2 = tf.keras.layers.Dense(units=1, activation='relu', name='output')

net = lstm_1(SandP_input)
out = dense_2(net)

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

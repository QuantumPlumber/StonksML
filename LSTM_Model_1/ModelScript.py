import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

num_stocks = 500
minutes_in_trading_day = 60 * 6.5
sequence_minutes = 30
features = 5 + 1  # open, high, low, close, time
input_vector_size = num_stocks * features  # the size of the flattened input data

SandP_input = tf.keras.layers.Input(shape=(sequence_minutes, features), name='SandP_input')
net = SandP_input


lstm_width = 100
num_nets = 4
for i in np.arange(num_nets):
    net = tf.keras.layers.LSTM(units=lstm_width, return_sequences=True, name='lstm_{}'.format(i))(net)

net = tf.keras.layers.LSTM(units=lstm_width, return_sequences=False, name='lstm_end')(net)
net = tf.keras.layers.Dense(units=30, activation='relu', name='dense_out')(net)
out = tf.keras.layers.Reshape((5, 6), name='output')(net)

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
#RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# Adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)

keras_model = tf.keras.Model(inputs=SandP_input, outputs=out)

keras_model.compile(optimizer=sgd,
                    loss={'output': 'mean_squared_error'},
                    loss_weights=[1.])

#direct = './multilayer_cnn_2'
direct = '.'
filename = direct + '/' + 'Recurrent_Model.png'
plot_model(keras_model, to_file=filename, show_shapes=True)

filename = direct + '/' + 'Recurrent_Model.json'
f = open(filename, 'w')
f.write(keras_model.to_json())
f.close()

keras_model.summary()

from KerasModels import ReccurrentCNN
from DataHandling import DataGenerator
import tensorflow as tf
import numpy as np
import importlib
import h5py

filename = './DataHandling/datafile.hdf5'
datafile = h5py.File(filename, 'r')

train_data = datafile['train_sequences'][0:int(1e5), ...]
predict_data = datafile['pred_sequences'][0:int(1e5), ...]

print(train_data.shape)

'''
dataset_size = train_data.shape[0]
test_cut = 5
test_cut_loc = dataset_size // test_cut

train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:-test_cut_loc], predict_data[0:-test_cut_loc]))
test_dataset = tf.data.Dataset.from_tensor_slices((train_data[-test_cut_loc:0], predict_data[-test_cut_loc:0]))

dataset = tf.data.Dataset.from_tensor_slices((train_data, predict_data))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(64)
'''

ReccurrentCNN.keras_model.fit(x=train_data, y=predict_data, batch_size=64, epochs=5, validation_split=.2)

ReccurrentCNN.keras_model.save('saved_model.h5')

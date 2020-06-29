from KerasModels import ReccurrentCNN
from DataHandling import DataGenerator
import tensorflow as tf
import numpy as np
import importlib

importlib.reload(DataGenerator)

(train_data, predict_data) = DataGenerator.dataset_creator(filedirectory='D:/StockData/',
                                                           num_tot_seq=50000,
                                                           num_seq_per_day=10,
                                                           train_seq_len=30,
                                                           pred_seq_len=5)

print(train_data.shape)

dataset_size = train_data.shape[0]
test_cut = 5
test_cut_loc = dataset_size // test_cut

train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:-test_cut_loc], predict_data[0:-test_cut_loc]))
test_dataset = tf.data.Dataset.from_tensor_slices((train_data[-test_cut_loc:0], predict_data[-test_cut_loc:0]))

dataset = tf.data.Dataset.from_tensor_slices((train_data, predict_data))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(64)

ReccurrentCNN.keras_model.fit(x=train_data, y=predict_data, batch_size=64, epochs=5, validation_split=.2)

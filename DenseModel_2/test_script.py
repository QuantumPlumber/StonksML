import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

model_file = './FC_model.h5'
print(model_file)
model = tf.keras.models.load_model(model_file)
model.summary()

filename = '../DataHandling/datafile_long_rescale_2std.hdf5'
filename = '../DataHandling/datafile_volume_scale.hdf5'
datafile = h5py.File(filename, 'r')

num_sequences = datafile['train_sequences'].shape[0]
print(num_sequences)
train_cut = .8
print(train_cut)
train_index = int(train_cut * num_sequences)

train_data = np.sum(datafile['train_sequences'][train_index:, :, [0, 1, 2, 3]],axis=2, keepdims=True) / 4
predict_data = np.sum(datafile['pred_sequences'][train_index:, 0:1, 0:4],axis=2, keepdims=True) / 4

train_shape = train_data.shape
print(train_data.shape)

sequences_to_predict = 10
sequence_indices = np.random.choice(np.arange(train_shape[0]), size=sequences_to_predict)

ground_truth = predict_data[sequence_indices, ...]
print(ground_truth.shape)
ground_train = train_data[sequence_indices, ...]
predictions = model.predict(ground_train)
print(predictions.shape)

columns = predictions.shape[-1]
fig, axes = plt.subplots(nrows=sequences_to_predict, ncols=columns, sharex=True)
print(axes.shape)

for row in np.arange(sequences_to_predict):
    for col in np.arange(columns):
        #axes[row, col].plot(np.concatenate((ground_train[row, :, col], ground_truth[row, :, col])), 'k')
        #axes[row, col].plot(np.concatenate((ground_train[row, :, col], predictions[row, :, col])), 'r')

        axes[row].plot(np.concatenate((ground_train[row, :], ground_truth[row, :])), 'k')
        axes[row].plot(np.concatenate((ground_train[row, :], predictions[row, :])), 'r')
plt.show()
# input('press any key to continue..')

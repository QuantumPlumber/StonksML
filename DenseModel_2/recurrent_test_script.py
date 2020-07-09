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

train_data = datafile['train_sequences'][train_index:, :, [0, 1, 2, 3, 5]]
predict_data = datafile['pred_sequences'][train_index:, :, 0:4]

train_shape = train_data.shape
print(train_data.shape)

sequences_to_predict = 10
sequence_indices = np.random.choice(np.arange(train_shape[0]), size=sequences_to_predict)

ground_truth = predict_data[sequence_indices, ...]
print(ground_truth.shape)
chosen_train_data = train_data[sequence_indices, ...]

dummy_recur_data = np.copy(chosen_train_data)
recurrent_predictions = np.zeros_like(ground_truth)
dummy_data = np.copy(chosen_train_data)
predictions = np.zeros_like(ground_truth)
for j in np.arange(ground_truth.shape[1]):
    recurrent_predictions[:, j:j+1, :] = model.predict(dummy_recur_data)

    # now replace the last value of the input with the prediction
    dummy_recur_data[:, :-1, :] = dummy_recur_data[:, 1:, :]
    dummy_recur_data[:, -1:, 0:4] = recurrent_predictions[:, j:j+1, :]

    predictions[:, j:j+1, :] = model.predict(dummy_data)

    # now replace the last value of the input with the prediction
    dummy_data[:, :-1, :] = dummy_data[:, 1:, :]
    dummy_data[:, -1:, 0:4] = ground_truth[:, j:j+1, :]

columns = ground_truth.shape[-1]
fig, axes = plt.subplots(nrows=sequences_to_predict, ncols=columns, sharex=True)
print(axes.shape)

for row in np.arange(sequences_to_predict):
    for col in np.arange(columns):
        axes[row, col].plot(np.concatenate((chosen_train_data[row, :, col], ground_truth[row, :, col])), 'k')
        axes[row, col].plot(np.concatenate((chosen_train_data[row, :, col], predictions[row, :, col])), 'r')
        axes[row, col].plot(np.concatenate((chosen_train_data[row, :, col], recurrent_predictions[row, :, col])), 'b')

plt.show()
# input('press any key to continue..')

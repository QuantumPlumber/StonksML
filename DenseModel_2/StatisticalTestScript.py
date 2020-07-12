import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

'''
predict the entire test set to see what the model distribution is. 

'''

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

#train_data_for_model = datafile['train_sequences'][train_index:, :, [0, 1, 2, 3]]
train_data = np.sum(datafile['train_sequences'][train_index:, :, [0, 1, 2, 3]], axis=2, keepdims=True) / 4
predict_data = np.sum(datafile['pred_sequences'][train_index:, 0:1, 0:4], axis=2, keepdims=True) / 4
print(train_data.shape)
print(predict_data.shape)
predictions = model.predict(train_data)
print(predictions.shape)

ground_truth_offsets = train_data[:, -1, :] - predict_data[:, 0, :]
avg = np.sum(ground_truth_offsets) / ground_truth_offsets.shape[0]
std = (np.sum((ground_truth_offsets - avg) ** 2) / ground_truth_offsets.shape[0]) ** .5
bin_values, bin_locs = np.histogram(ground_truth_offsets, range=(-4 * std, 4 * std), bins=100)
# bin_values = bin_values / np.sum(bin_values)  # normalize bin values
mid_locs = (bin_locs[:-1] + bin_locs[1:]) / 2
coeff = np.polyfit(mid_locs, np.log(np.abs(bin_values)), deg=10)
ground_truth_fit = np.exp(np.polyval(coeff, mid_locs))
truth_area = np.sum(ground_truth_fit*(mid_locs[1] - mid_locs[0]))
print('total ground truth distribution area: {}'.format(truth_area))
plt.plot(mid_locs, ground_truth_fit)

predictions_offsets = train_data[:, -1, :] - predictions[:, 0, :]
avg = np.sum(predictions_offsets) / predictions_offsets.shape[0]
std = (np.sum((predictions_offsets - avg) ** 2) / predictions_offsets.shape[0]) ** .5
bin_values, bin_locs = np.histogram(predictions_offsets, range=(-4 * std, 4 * std), bins=100)
# bin_values = bin_values / np.sum(bin_values)  # normalize bin values
predict_mid_locs = (bin_locs[:-1] + bin_locs[1:]) / 2
dummy_log = np.log(np.abs(bin_values))
dummy_log[np.isinf(dummy_log)] = 0
coeff = np.polyfit(predict_mid_locs, dummy_log, deg=10)
predict_truth_fit = np.exp(np.polyval(coeff, predict_mid_locs))
predict_area = np.sum(predict_truth_fit*(predict_mid_locs[1] - predict_mid_locs[0]))
print('total prediction distribution area: {}'.format(predict_area))

print('ratio of areas: {}'.format(predict_area/truth_area))

plt.plot(predict_mid_locs, predict_truth_fit)
plt.show()

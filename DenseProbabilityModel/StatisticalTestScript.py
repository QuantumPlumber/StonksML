import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

'''
predict the entire test set to see what the model distribution is. 

'''


def loss_fn(y_pred, y_truth):
    gaussian_error = tf.math.exp(
        - tf.math.pow(tf.math.subtract(y_pred[:, :, 0:1], y_truth[:, :, 0:1]), 2) / tf.math.multiply(y_pred[:, :, 1:2],
                                                                                                     2.0))

    # gaussian_error = 1 - tf.math.exp(- (y_pred[:, :, 0:1] - y_truth[:, :, 0:1]) ** 2 / (2*y_pred[:, :, 1:2]))

    gaussian_error = 1 - tf.math.exp(- (y_pred[:, :, 0:1] - y_truth[:, :, 0:1]) ** 2)
    return gaussian_error

model_file = './FC_model.h5'
print(model_file)
model = tf.keras.models.load_model(model_file, custom_objects={'loss_fn': loss_fn})
model.summary()

filename = '../DataHandling/datafile_long_rescale_2std.hdf5'
filename = '../DataHandling/datafile_volume_scale.hdf5'
datafile = h5py.File(filename, 'r')

num_sequences = datafile['train_sequences'].shape[0]
print(num_sequences)
train_cut = .8
print(train_cut)
train_index = int(train_cut * num_sequences)

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
plt.plot(mid_locs, ground_truth_fit, color='g')

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
plt.plot(predict_mid_locs, predict_truth_fit, color='b')


difference_offsets = predict_data[:, 0, :] - predictions[:, 0, :]
avg = np.sum(difference_offsets) / difference_offsets.shape[0]
std = (np.sum((difference_offsets - avg) ** 2) / difference_offsets.shape[0]) ** .5
bin_values, bin_locs = np.histogram(difference_offsets, range=(-4 * std, 4 * std), bins=100)
# bin_values = bin_values / np.sum(bin_values)  # normalize bin values
diff_mid_locs = (bin_locs[:-1] + bin_locs[1:]) / 2
dummy_log = np.log(np.abs(bin_values))
dummy_log[np.isinf(dummy_log)] = 0
coeff = np.polyfit(diff_mid_locs, dummy_log, deg=10)
diff_truth_fit = np.exp(np.polyval(coeff, diff_mid_locs))
diff_area = np.sum(diff_truth_fit*(diff_mid_locs[1] - diff_mid_locs[0]))
print('total diff distribution area: {}'.format(diff_area))
plt.plot(diff_mid_locs, diff_truth_fit, color='k')

print('ratio of areas: {}'.format(predict_area/truth_area))

plt.plot(predict_mid_locs, predict_truth_fit)
plt.show()
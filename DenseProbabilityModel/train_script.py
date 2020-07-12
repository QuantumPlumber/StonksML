import h5py
import numpy as np
import DenseProbabilityModel.DenseModel as ModelFile

# filename = './DataHandling/datafile2.hdf5'
filename = '../DataHandling/datafile_long_rescale_2std.hdf5'
# filename = '../DataHandling/datafile_volume_scale.hdf5'
datafile = h5py.File(filename, 'r')

# cut dataset for train and validation
num_sequences = datafile['train_sequences'].shape[0]
train_cut = .8
train_index = int(train_cut * num_sequences)

#
train_data = np.sum(datafile['train_sequences'][0:train_index, :, [0, 1, 2, 3]], axis=2, keepdims=True) / 4
train_data[np.isnan(train_data)] = 0
train_data[np.isinf(train_data)] = 0
predict_data = np.sum(datafile['pred_sequences'][0:train_index, 0:1, 0:4], axis=2, keepdims=True) / 4
predict_data[np.isnan(predict_data)] = 0
predict_data[np.isinf(predict_data)] = 0

max_outliers = 10 * train_data.max()
min_outliers = 10 * train_data.min()
predict_data[predict_data > max_outliers] = max_outliers
predict_data[predict_data < min_outliers] = min_outliers

total_offsets = np.concatenate(
    (train_data[:, 1:, :] - train_data[:, :-1, :], predict_data[:, 1:, :] - predict_data[:, :-1, :]),
    axis=None)
avg = np.sum(total_offsets) / total_offsets.shape[0]
std = (np.sum((total_offsets - avg) ** 2) / total_offsets.shape[0]) ** .5
bin_values, bin_locs = np.histogram(total_offsets, range=(-4 * std, 4 * std), bins=100)
bin_values = bin_values / np.sum(bin_values)  # normalize bin values
mid_locs = (bin_locs[:-1] + bin_locs[1:]) / 2
coeff = np.polyfit(mid_locs, np.log(np.abs(bin_values)), deg=10)
step_size = np.sum(train_data[:, -1, :] - predict_data[:, 0, :], axis=1).flatten() / train_data.shape[-1]
scaling_factors = np.exp(np.polyval(coeff, step_size))
scaling_factors = scaling_factors/scaling_factors.max()
weights = 1 / scaling_factors
weights[weights > 100] = 100
weights = np.ones_like(scaling_factors) # turn off weights


print(train_data.shape)

ModelFile.keras_model.fit(x=train_data, y=predict_data, sample_weight=weights,
                          shuffle=True, batch_size=64, epochs=5,
                          validation_split=.2)

ModelFile.keras_model.save('FC_model.h5')

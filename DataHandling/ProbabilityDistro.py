import h5py
import matplotlib.pyplot as plt
import numpy as np

'''
Generates a weighting function for a dataset based on the gaussian fit to the histogram of the step size 
between minutes
'''

filename = './datafile2.hdf5'
filename = './datafile_long_rescale_2std.hdf5'
filename = './datafile_volume_scale.hdf5'
datafile = h5py.File(filename, 'r')

for key in datafile.keys():
    print(key)
    print(datafile[key].attrs['total_entries'])
    print(datafile[key].shape)

columns = datafile[key].shape[-1]
print(columns)

entries = datafile[key].shape[0]
print(entries)

all_train_data = datafile['train_sequences'][:, :, 0:4]
print('train data max {}'.format(all_train_data.max()))
print('train data min {}'.format(all_train_data.min()))
all_predict_data = datafile['pred_sequences'][:, :, 0:4]
print('predict data max {}'.format(all_predict_data.max()))
print('predict data min {}'.format(all_predict_data.min()))

gross_values = all_predict_data[all_predict_data > 10 * all_train_data.max()]
print(gross_values.shape)

nan_indices = np.isnan(all_train_data)
print('number of NaN indices: {}'.format(np.sum(nan_indices)))
all_train_data[nan_indices] = 0
nan_indices = np.isnan(all_predict_data)
print('number of NaN indices: {}'.format(np.sum(nan_indices)))
all_predict_data[nan_indices] = 0

location_of_bad_values = np.nonzero(all_predict_data > 10 * all_train_data.max())

all_predict_data[all_predict_data > 10 * all_train_data.max()] = 10 * all_train_data.max()
all_predict_data[all_predict_data < 10 * all_train_data.min()] = 10 * all_train_data.min()

print('train data max {}'.format(all_train_data.max()))
print('train data min {}'.format(all_train_data.min()))

inf_indices = np.isinf(all_train_data)
print('number of inf indices: {}'.format(np.sum(inf_indices)))
all_train_data[inf_indices] = 0
inf_indices = np.isinf(all_predict_data)
print('number of inf indices: {}'.format(np.sum(inf_indices)))
all_predict_data[inf_indices] = 0

print('predict data max {}'.format(all_predict_data.max()))
print('predict data min {}'.format(all_predict_data.min()))

total_offsets = np.concatenate(
    (all_train_data[:, 1:, :] - all_train_data[:, :-1, :], all_predict_data[:, 1:, :] - all_predict_data[:, :-1, :]),
    axis=None)

print(total_offsets.shape)
print(total_offsets.max())
print(total_offsets.min())
avg = np.sum(total_offsets) / total_offsets.shape[0]
print('average of dataset movement: {}'.format(avg))
std = (np.sum((total_offsets - avg) ** 2) / total_offsets.shape[0]) ** .5
print('std of dataset movement: {}'.format(std))

plt.figure()
bin_values, bin_locs, _ = plt.hist(total_offsets, range=(-4 * std, 4 * std), bins=100)

bin_values = bin_values / np.sum(bin_values)  # normalize bin values

mid_locs = (bin_locs[:-1] + bin_locs[1:]) / 2
coeff = np.polyfit(mid_locs, np.log(np.abs(bin_values)), deg=10)

print('coefficients are (2nd, 1st, 0th):')
print(coeff)

plt.figure(figsize=(6, 6))
plt.plot(mid_locs, bin_values)
plt.plot(mid_locs, np.exp(np.polyval(coeff, mid_locs)))

plt.show()

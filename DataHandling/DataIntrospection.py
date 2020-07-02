import h5py
import matplotlib.pyplot as plt
import numpy as np

filename = './datafile2.hdf5'
filename = './datafile_long.hdf5'
datafile = h5py.File(filename, 'r')

for key in datafile.keys():
    print(key)
    print(datafile[key].attrs['total_entries'])
    print(datafile[key].shape)

columns = datafile[key].shape[-1]
print(columns)

entries = datafile[key].shape[0]
print(entries)

plot_number = 10

fig, axes = plt.subplots(nrows=plot_number, ncols=columns, sharex=True)

all_train_data = datafile['train_sequences'][:, :, :]
nan_indices = np.isnan(all_train_data)
print(np.sum(nan_indices))

while True:
    seq_choices = np.random.choice(np.arange(start=int(.8*entries), stop=int(entries)), size=plot_number)
    print(seq_choices)

    for i, seq in enumerate(seq_choices):
        for col in np.arange(columns):
            plotter = np.concatenate((datafile['train_sequences'][seq, :, col], datafile['pred_sequences'][seq, :, col]))
            axes[i, col].clear()
            axes[i, col].plot(plotter)

    plt.show()
    signal = input('press n for a new plot \n press any other key to exit..')

    if signal != 'n':
        break

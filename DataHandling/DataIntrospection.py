import h5py
import matplotlib.pyplot as plt
import numpy as np

filename = './datafile.hdf5'
datafile = h5py.File(filename, 'r')

for key in datafile.keys():
    print(key)

columns = datafile[key].shape[-1]
print(columns)

entries = datafile[key].shape[0]
print(entries)

plot_number = 10

fig, axes = plt.subplots(nrows=plot_number, ncols=columns, sharex=True)

seq_choices = np.random.choice(np.arange(entries), size=plot_number)

for i, seq in enumerate(seq_choices):
    for col in np.arange(columns):
        plotter = np.concatenate((datafile['train_sequences'][seq, :, col], datafile['pred_sequences'][seq, :, col]))
        axes[i, col].plot(plotter)

plt.show()
input('press any key to continue..')

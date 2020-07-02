import h5py
import numpy as np
import DenseModel_1.DenseModel as ModelFile

#filename = './DataHandling/datafile2.hdf5'
filename = './DataHandling/datafile_long.hdf5'
datafile = h5py.File(filename, 'r')

num_sequences = datafile['train_sequences'].shape[0]
train_cut = .8
train_index = int(train_cut*num_sequences)

train_data = datafile['train_sequences'][0:train_index, ...]
train_data[np.isnan(train_data)] = 0
predict_data = datafile['pred_sequences'][0:train_index, ...]
predict_data[np.isnan(predict_data)] = 0

print(train_data.shape)

ModelFile.keras_model.fit(x=train_data, y=predict_data, shuffle=True, batch_size=64, epochs=5, validation_split=.2)

ModelFile.keras_model.save('FC_model.h5')


import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt


def loss_fn(y_truth, y_pred):
    centroid = y_pred[:, :, 0:1]
    sigma = y_pred[:, :, 1:2]

    centroid_error = 1 - tf.math.exp(-((centroid - y_truth[:, :, 0:1]) ** 2 / (2 * sigma ** 2)))
    #centroid_error = (centroid - y_truth[:, :, 0:1]) ** 2
    sigma_error = sigma ** 2

    total_error = centroid_error + sigma_error

    # debug test function (simpler)
    # gaussian_error = 1 - tf.math.exp(- (sigma - y_truth[:, :, 0:1]) ** 2)
    return total_error


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

train_shape = train_data.shape
print(train_data.shape)

sequences_to_predict = 10
sequence_indices = np.random.choice(np.arange(train_shape[0]), size=sequences_to_predict)

ground_truth = predict_data[sequence_indices, ...]
print(ground_truth.shape)
ground_train = train_data[sequence_indices, ...]
predictions = model.predict(ground_train)
print(predictions.shape)

columns = 1
fig, axes = plt.subplots(nrows=sequences_to_predict, ncols=columns, sharex=True)
print(axes.shape)

for row in np.arange(sequences_to_predict):
    for col in np.arange(columns):
        train = ground_train[row, :, 0]
        truth = ground_truth[row, 0, 0]
        centroid = predictions[row, 0, 0]
        #sigma = predictions[row, 0, 1]
        sigma = truth / 1
        gauss_domain = np.linspace(start=centroid - 4*sigma, stop=centroid + 4*sigma, num=100)

        predict_level = np.exp(-((centroid - truth) ** 2 / (2 * sigma ** 2)))
        predict_progression = np.linspace(start=0, stop=predict_level * 30. / 31., num=30)

        predict_gauss = np.exp(-((centroid - gauss_domain) ** 2 / (2 * sigma ** 2)))

        axes[row].plot(train, predict_progression, 'k')
        axes[row].plot(truth, predict_level, '.b')
        axes[row].plot(gauss_domain, predict_gauss, 'b')
        axes[row].plot(centroid, 1.0, '.r')


plt.show()
# input('press any key to continue..')

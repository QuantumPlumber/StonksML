import os
import h5py
import arrow
import numpy as np

'''
    A generator to grab daily day files from the Data repository return them
'''


def days_in_directory(filedirectory='D:/StockData/', ticker='SPY'):
    file_number = 0
    unique_dates = []

    for direntry in os.scandir(filedirectory):
        result_dict = {}
        meta_result_dict = {}

        if direntry.is_dir():
            continue

        if direntry.is_file():
            filepath = direntry.path
            # print(filepath)
            try:
                datafile = h5py.File(filepath, 'r')
            except:
                print('could not open file: {}'.format(filepath))
                continue

        if ticker not in list(datafile.keys()):
            print('no \'SPY\' data in file: {}'.format(filepath))
            continue

        time_data = datafile[ticker]['datetime'][...]
        if time_data.shape[0] == 0:
            print('no \'SPY\' data in file: {}'.format(filepath))
            continue
        else:
            print('\'SPY\' data found in file: {}'.format(filepath))

        mid_day = arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).to('America/New_York')
        date = mid_day.date()

        start_date = arrow.Arrow(year=2020, month=4, day=21, tzinfo='America/New_York')
        end_date = arrow.now('America/New_York')

        print('isoweekday is: {}'.format((mid_day.isoweekday())))
        if mid_day.isoweekday() not in [1, 2, 3, 4, 5]:
            print('not a weekday')
            continue
        else:
            if date not in unique_dates and mid_day.is_between(start=start_date, end=end_date, bounds='()'):
                file_number += 1
                unique_dates.append(date)
            else:
                continue

        datafile.close()

    return file_number, unique_dates


def data_file_generator(filedirectory='D:/StockData/', ticker='SPY'):
    result_list = []
    meta_result_list = []

    VIX_path = filedirectory + 'S&P_500_VIX_2020-05-10'
    if os.path.exists(VIX_path):
        print('opening datafile:')
        print(VIX_path)
        VIX_datafile = h5py.File(VIX_path, 'r')
    else:
        print('VIX file does not exist!')

    VIX_datetime = VIX_datafile['VIX9D']['datetime'][...]
    VIX_dates = []
    for dd in VIX_datetime:
        VIX_dates.append(arrow.get(dd * 1e-3).to('America/New_York').date())

    VIX_volatility = (VIX_datafile['VIX9D']['open'][...] + VIX_datafile['VIX9D']['high'][...] +
                      VIX_datafile['VIX9D']['low'][...] + VIX_datafile['VIX9D']['close'][...]) / 4.

    unique_dates = []
    for direntry in os.scandir(filedirectory):
        result_dict = {}
        meta_result_dict = {}

        if direntry.is_dir():
            continue

        if direntry.is_file():
            filepath = direntry.path
            print(filepath)
            try:
                datafile = h5py.File(filepath, 'r')
            except:
                print('could not open file: {}'.format(filepath))
                continue

        if ticker not in list(datafile.keys()):
            print('no \'SPY\' data in file: {}'.format(filepath))
            continue

        time_data = datafile[ticker]['datetime'][...]
        if time_data.shape[0] == 0:
            print('no \'SPY\' data in file: {}'.format(filepath))
            continue

        mid_day = arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).to('America/New_York')
        date = mid_day.date()

        start_date = arrow.Arrow(year=2020, month=4, day=21, tzinfo='America/New_York')
        end_date = arrow.now('America/New_York')

        if arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).isoweekday() not in [1, 2, 3, 4, 5]:
            print('not a weekday')
            continue
        else:
            if date not in unique_dates and mid_day.is_between(start=start_date, end=end_date, bounds='()'):
                unique_dates.append(date)

                count = 0
                for dd in VIX_dates:
                    if dd == date:
                        break
                    count += 1

                yield datafile, str(date), VIX_volatility[count]
            else:
                continue


def trading_minute(t):
    # print(t.shape)
    tradeable = np.zeros_like(t, dtype=np.bool)
    # print(tradeable.shape)

    start_of_trading_minute = 9 * 60 + 30
    end_of_trading_minute = 16 * 60
    total_minutes = end_of_trading_minute - start_of_trading_minute

    trade_time = arrow.get(t[0] * 1e-3).to('America/New_York')
    timeseries_t_0 = trade_time.hour * 60 + trade_time.minute  # in minutes from open

    current_minute = timeseries_t_0 - start_of_trading_minute

    trading_minute = []
    for i in np.arange(t.shape[0]):
        trading_minute.append(current_minute)
        current_minute += 1

    return np.array(trading_minute) / total_minutes


def daily_sequence_generator(filedirectory='D:/StockData/',
                             num_seq_per_day=10,
                             train_seq_len=30,
                             pred_seq_len=5):
    tot_seq_len = train_seq_len + pred_seq_len
    total_seq_shape = (tot_seq_len, 6)  # open, high, low, close, vol, datetime

    train_seq_shape = (num_seq_per_day, train_seq_len, 6)  # open, high, low, close, vol, datetime
    pred_seq_shape = (num_seq_per_day, pred_seq_len, 6)  # open, high, low, close, vol, datetime

    unique_dates = []
    file_is_new = False

    for direntry in os.scandir(filedirectory):
        if direntry.is_dir():
            continue

        if direntry.is_file():
            filepath = direntry.path
            print(filepath)
            try:
                datafile = h5py.File(filepath, 'r')
                file_is_new = False
            except:
                print('could not open file: {}'.format(filepath))
                continue

        tickers = datafile.keys()

        for ticker in tickers:
            time_data = datafile[ticker]['datetime'][...]
            if time_data.shape[0] == 0:
                break

            mid_day = arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).to('America/New_York')
            date = mid_day.date()
            print(date)

            if arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).isoweekday() not in [1, 2, 3, 4, 5]:
                # print('not a weekday')
                continue
            else:
                if date not in unique_dates:
                    unique_dates.append(date)
                    file_is_new = True
                    break

        if not file_is_new:  # don't process if this file contains duplicate data.
            continue

        for ticker in tickers:
            # print(ticker)
            if ticker is not 'SPY':
                try:
                    open_data = datafile[ticker]['open'][...]
                    high_data = datafile[ticker]['high'][...]
                    low_data = datafile[ticker]['low'][...]
                    close_data = datafile[ticker]['close'][...]
                    volum_data = datafile[ticker]['volume'][...]
                    time_data = datafile[ticker]['datetime'][...]
                except:
                    continue

                if time_data.shape[0] == 0:
                    continue

                if time_data[0] < 1:
                    # print('no data for {}'.format(ticker))
                    continue

                if arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).isoweekday() not in [1, 2, 3, 4, 5]:
                    # print('not a weekday')
                    continue

                seq_starts = np.random.choice(np.arange(time_data.shape[0] - tot_seq_len), size=num_seq_per_day)
                # print(seq_starts)

                train_sequences = np.zeros(shape=train_seq_shape)
                pred_sequences = np.zeros(shape=pred_seq_shape)
                dummy_seq = np.zeros(shape=total_seq_shape)
                for i, start in enumerate(seq_starts):
                    dummy_seq[:, 0] = open_data[start:start + tot_seq_len]
                    dummy_seq[:, 1] = high_data[start:start + tot_seq_len]
                    dummy_seq[:, 2] = low_data[start:start + tot_seq_len]
                    dummy_seq[:, 3] = close_data[start:start + tot_seq_len]
                    dummy_seq[:, 4] = volum_data[start:start + tot_seq_len]
                    dummy_seq[:, 5] = trading_minute(time_data[start:start + tot_seq_len])

                    # shift and scale according to the train data
                    avg = np.sum(dummy_seq[:, 0:5], axis=0, keepdims=True) / tot_seq_len
                    std_dev = (np.sum((dummy_seq[:, 0:5] - avg) ** 2, axis=0, keepdims=True) / tot_seq_len) ** .5
                    dummy_seq[:, 0:5] = (dummy_seq[:, 0:5] - avg) / std_dev

                    train_sequences[i, :, :] = dummy_seq[0:train_seq_len, :]
                    pred_sequences[i, :, :] = dummy_seq[train_seq_len:, :]

                train_sequences[np.isnan(train_sequences)] = 0 # get rid of nan
                pred_sequences[np.isnan(pred_sequences)] = 0

            yield train_sequences, pred_sequences

        datafile.close()


def dataset_creator(filedirectory='D:/StockData/',
                    num_tot_seq=1000,
                    num_seq_per_day=10,
                    train_seq_len=30,
                    pred_seq_len=5):
    seq_gen = daily_sequence_generator(filedirectory=filedirectory,
                                       num_seq_per_day=num_seq_per_day,
                                       train_seq_len=train_seq_len,
                                       pred_seq_len=pred_seq_len)
    train_list = []
    predict_list = []
    for i in np.arange(num_tot_seq // num_seq_per_day):
        train_train, pred_pred = next(seq_gen)

        train_list.append(train_train)
        predict_list.append(pred_pred)

    train_output = np.stack(train_list, axis=0)
    predict_output = np.stack(predict_list, axis=0)

    return train_output, predict_output


def datafile_creator(output_file_name='datafile.hdf5',
                     filedirectory='D:/StockData/',
                     num_tot_seq=1000,
                     num_seq_per_day=10,
                     train_seq_len=30,
                     pred_seq_len=5):
    datafile = h5py.File('./' + output_file_name)

    total_sequences = (num_tot_seq // num_seq_per_day) * num_seq_per_day  # must be divisible
    print('total sequences to be generated: {}'.format(total_sequences))
    train_shape = (total_sequences, train_seq_len, 6)
    train_hdf5 = datafile.create_dataset(name='train_sequences', shape=train_shape)
    predict_shape = (total_sequences, pred_seq_len, 6)
    pred_hdf5 = datafile.create_dataset(name='pred_sequences', shape=predict_shape)

    seq_gen = daily_sequence_generator(filedirectory=filedirectory,
                                       num_seq_per_day=num_seq_per_day,
                                       train_seq_len=train_seq_len,
                                       pred_seq_len=pred_seq_len)
    train_list = []
    predict_list = []
    i = 0
    for train_train, predict_predict in seq_gen:
        cut_bot = i * num_seq_per_day
        cut_top = (i + 1) * num_seq_per_day

        if cut_top >= total_sequences:
            break

        train_hdf5[cut_bot:cut_top, ...] = train_train
        pred_hdf5[cut_bot:cut_top, ...] = predict_predict

        i += 1

        #print(i)
        #print((cut_bot, cut_top))

    train_hdf5.attrs['total_entries'] = np.array(cut_top)
    train_hdf5.attrs['total_days'] = np.array(i)
    pred_hdf5.attrs['total_entries'] = np.array(cut_top)
    pred_hdf5.attrs['total_days'] = np.array(i)

    datafile.close()


if __name__ == '''__main__''':
    train_dataset, predict_dataset = dataset_creator(filedirectory='D:/StockData/',
                                                     num_tot_seq=1000,
                                                     num_seq_per_day=10,
                                                     train_seq_len=30,
                                                     pred_seq_len=5)

    print(train_dataset.shape)
    print(predict_dataset.shape)
    # print(np.max(train_dataset[1, :, :], axis=1))
    # print(train_dataset[50, :, :])

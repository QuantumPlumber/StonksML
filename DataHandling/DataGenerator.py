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

    trade_time = arrow.get(t[0] * 1e-3).to('America/New_York')
    timeseries_t_0 = trade_time.hour * 60 + trade_time.minute  # in minutes from open

    current_minute = timeseries_t_0 - start_of_trading_minute

    trading_minute = []
    for i in np.arange(t.shape[0]):
        trading_minute.append(current_minute)
        current_minute += 1

    return np.array(trading_minute)


def dataset_creator(filedirectory='D:/StockData/',
                    num_tot_seq=1000,
                    num_seq_per_day=10,
                    train_seq_len=30,
                    pred_seq_len=5):
    tot_seq_len = train_seq_len + pred_seq_len

    train_seq_shape = (num_seq_per_day, train_seq_len, 6)  # open, high, low, close, vol, datetime
    train_sequences = np.empty(shape=train_seq_shape)

    pred_seq_shape = (num_seq_per_day, pred_seq_len, 6)
    pred_sequences = np.empty(shape=pred_seq_shape)

    unique_dates = []
    file_is_new = False

    num_seq_recorded = 0
    skip = 0
    for direntry in os.scandir(filedirectory):
        if skip < 0:
            skip += 1
            print(skip)
            continue

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
                continue

            mid_day = arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).to('America/New_York')
            date = mid_day.date()

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
            if ticker is not 'SPY':
                time_data = datafile[ticker]['datetime'][...]
                if time_data.shape[0] == 0:
                    continue

                if time_data[0] < 1:
                    print('no data for {}'.format(ticker))
                    continue

                if arrow.get(time_data[time_data.shape[0] // 2] * 1e-3).isoweekday() not in [1, 2, 3, 4, 5]:
                    # print('not a weekday')
                    continue

                seq_starts = np.random.choice(np.arange(time_data.shape[0] - tot_seq_len),
                                              size=num_seq_per_day)

                train_dummy_seq = np.zeros(shape=train_seq_shape)
                pred_dummy_seq = np.zeros(shape=pred_seq_shape)
                for i, start in enumerate(seq_starts):
                    train_dummy_seq[i, :, 0] = datafile[ticker]['open'][start:start + train_seq_len]
                    train_dummy_seq[i, :, 1] = datafile[ticker]['high'][start:start + train_seq_len]
                    train_dummy_seq[i, :, 2] = datafile[ticker]['low'][start:start + train_seq_len]
                    train_dummy_seq[i, :, 3] = datafile[ticker]['close'][start:start + train_seq_len]
                    train_dummy_seq[i, :, 4] = datafile[ticker]['volume'][start:start + train_seq_len]
                    train_dummy_seq[i, :, 5] = trading_minute(datafile[ticker]['datetime'][start:start + train_seq_len])

                    train_sequences = np.concatenate((train_sequences, train_dummy_seq), axis=0)

                    pred_dummy_seq[i, :, 0] = datafile[ticker]['open'][start + train_seq_len:start + tot_seq_len]
                    pred_dummy_seq[i, :, 1] = datafile[ticker]['high'][start + train_seq_len:start + tot_seq_len]
                    pred_dummy_seq[i, :, 2] = datafile[ticker]['low'][start + train_seq_len:start + tot_seq_len]
                    pred_dummy_seq[i, :, 3] = datafile[ticker]['close'][start + train_seq_len:start + tot_seq_len]
                    pred_dummy_seq[i, :, 4] = datafile[ticker]['volume'][start + train_seq_len:start + tot_seq_len]
                    pred_dummy_seq[i, :, 5] = trading_minute(
                        datafile[ticker]['datetime'][start + train_seq_len:start + tot_seq_len])

                    pred_sequences = np.concatenate((pred_sequences, pred_dummy_seq), axis=0)

                    num_seq_recorded += num_seq_per_day

                    if num_seq_recorded >= num_tot_seq:
                        return train_sequences[1:], pred_sequences[1:]


if __name__ == '''__main__''':
    train_dataset, predict_dataset = dataset_creator(filedirectory='D:/StockData/',
                              num_tot_seq=1000,
                              num_seq_per_day=10,
                              train_seq_len=30,
                              pred_seq_len=5)

    print(train_dataset.shape)
    print(predict_dataset.shape)
    #print(train_dataset[20])

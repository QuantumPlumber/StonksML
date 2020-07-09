import DataHandling.DataGenerator as DataGenerator

DataGenerator.datafile_creator(output_file_name='datafile_volume_scale_test.hdf5',
                               filedirectory='D:/StockData/',
                               num_tot_seq=int(500*10*5),
                               num_seq_per_day=10,
                               train_seq_len=30,
                               pred_seq_len=30)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

"""
code from: https://towardsdatascience.com/feature-engineering-on-time-series-data-transforming-signal-data-of-a-smartphone-accelerometer-for-72cbe34b8a60
data from: https://www.cis.fordham.edu/wisdm/dataset.php
"""

"""
input:
    df: dataframe with columns indicating movements in x, y, z and a columns indicating time (unix timestamp
    window_size: number of periods to consider for new group of time unit
    step_size: number of periods of step between each new time unix
    col_x_axis: name of the column of the x_axis
    ...
    
output:
    an object that has a X_train, a df with features common features engineered from accelerometer data
"""
class AccelerometerExtractorSingleAcc:

    # Main dataframe with all data
    df = pd.DataFrame()
    # Name of the columns with the movement in x, y, z axis
    col_x_axis = str
    col_y_axis = str
    col_z_axis = str
    
    # Name of the column with the time id (datetime or timestamp)
    col_id_time = str
    
    # Number of original period in a single new period
    window_size = int
    # Number of original period between each new begining of period
    # Overlap is possible
    step_size = int

    # Dataframe with extracted data
    X_train = pd.DataFrame()
    se_x_list = []
    se_y_list = []
    se_z_list = []
    train_labels = []
    start_window_list = []
    end_window_list = []

    def __init__(self, df: pd.DataFrame(), window_size: int = 100, step_size: int = 50,
                 col_x_axis: str = 'x-axis', col_y_axis: str = 'y-axis', col_z_axis: str = 'z-axis',
                 col_id_time: str = 'timestamp'):

        self.df = df

        self.col_x_axis = col_x_axis
        self.col_y_axis = col_y_axis
        self.col_z_axis = col_z_axis
        self.col_id_time = col_id_time

        self.window_size = window_size
        self.step_size = step_size

        self.generate_list_rolled()

    def generate_list_rolled(self):

        se_x_list = []
        se_y_list = []
        se_z_list = []
        train_labels = []
        start_window_list = []
        end_window_list = []

        df = self.df
        # creating overlapping windows of size window-size 100
        for i in range(0, df.shape[0] - self.window_size, self.step_size):
            xs = df[self.col_x_axis].values[i: i + self.window_size]
            ys = df[self.col_y_axis].values[i: i + self.window_size]
            zs = df[self.col_z_axis].values[i: i + self.window_size]
            label = stats.mode(df['activity'][i: i + self.window_size])[0][0]

            start_window = df[self.col_id_time].values[i]
            end_window = df[self.col_id_time].values[i + self.window_size]

            se_x_list.append(xs)
            se_y_list.append(ys)
            se_z_list.append(zs)
            train_labels.append(label)
            start_window_list.append(start_window)
            end_window_list.append(end_window)

        self.se_x_list = pd.Series(se_x_list)
        self.se_y_list = pd.Series(se_y_list)
        self.se_z_list = pd.Series(se_z_list)
        self.train_labels = train_labels
        self.start_window_list = start_window_list
        self.end_window_list = end_window_list

    def set_col_names(self, col_x_axis='x_axis', col_y_axis='y_axis', col_z_axis='z_axis', col_id_time='time'):

        self.col_x_axis = col_x_axis
        self.col_y_axis = col_y_axis
        self.col_z_axis = col_z_axis
        self.col_id_time = col_id_time

    def gen_acceleration_features(self):

        se_x_list = self.se_x_list
        se_y_list = self.se_y_list
        se_z_list = self.se_z_list

        X_train = pd.DataFrame()

        X_train['start_time'] = pd.Series(self.start_window_list)
        X_train['end_time'] = pd.Series(self.end_window_list)

        # mean
        X_train['x_mean'] = se_x_list.apply(lambda x: x.mean())
        X_train['y_mean'] = se_y_list.apply(lambda x: x.mean())
        X_train['z_mean'] = se_z_list.apply(lambda x: x.mean())

        # std dev
        X_train['x_std'] = se_x_list.apply(lambda x: x.std())
        X_train['y_std'] = se_y_list.apply(lambda x: x.std())
        X_train['z_std'] = se_z_list.apply(lambda x: x.std())

        # avg absolute diff
        X_train['x_aad'] = se_x_list.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_train['y_aad'] = se_y_list.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_train['z_aad'] = se_z_list.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

        # min
        X_train['x_min'] = se_x_list.apply(lambda x: x.min())
        X_train['y_min'] = se_y_list.apply(lambda x: x.min())
        X_train['z_min'] = se_z_list.apply(lambda x: x.min())

        # max
        X_train['x_max'] = se_x_list.apply(lambda x: x.max())
        X_train['y_max'] = se_y_list.apply(lambda x: x.max())
        X_train['z_max'] = se_z_list.apply(lambda x: x.max())

        # max-min diff
        X_train['x_maxmin_diff'] = X_train['x_max'] - X_train['x_min']
        X_train['y_maxmin_diff'] = X_train['y_max'] - X_train['y_min']
        X_train['z_maxmin_diff'] = X_train['z_max'] - X_train['z_min']

        # median
        X_train['x_median'] = se_x_list.apply(lambda x: np.median(x))
        X_train['y_median'] = se_y_list.apply(lambda x: np.median(x))
        X_train['z_median'] = se_z_list.apply(lambda x: np.median(x))

        # median abs dev
        X_train['x_mad'] = se_x_list.apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_train['y_mad'] = se_y_list.apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_train['z_mad'] = se_z_list.apply(lambda x: np.median(np.absolute(x - np.median(x))))

        # interquartile range
        X_train['x_IQR'] = se_x_list.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_train['y_IQR'] = se_y_list.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_train['z_IQR'] = se_z_list.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

        # negative count
        X_train['x_neg_count'] = se_x_list.apply(lambda x: np.sum(x < 0))
        X_train['y_neg_count'] = se_y_list.apply(lambda x: np.sum(x < 0))
        X_train['z_neg_count'] = se_z_list.apply(lambda x: np.sum(x < 0))

        # positive count
        X_train['x_pos_count'] = se_x_list.apply(lambda x: np.sum(x > 0))
        X_train['y_pos_count'] = se_y_list.apply(lambda x: np.sum(x > 0))
        X_train['z_pos_count'] = se_z_list.apply(lambda x: np.sum(x > 0))

        # values above mean
        X_train['x_above_mean'] = se_x_list.apply(lambda x: np.sum(x > x.mean()))
        X_train['y_above_mean'] = se_y_list.apply(lambda x: np.sum(x > x.mean()))
        X_train['z_above_mean'] = se_z_list.apply(lambda x: np.sum(x > x.mean()))

        # number of peaks
        X_train['x_peak_count'] = se_x_list.apply(lambda x: len(find_peaks(x)[0]))
        X_train['y_peak_count'] = se_y_list.apply(lambda x: len(find_peaks(x)[0]))
        X_train['z_peak_count'] = se_z_list.apply(lambda x: len(find_peaks(x)[0]))

        # skewness
        X_train['x_skewness'] = se_x_list.apply(lambda x: stats.skew(x))
        X_train['y_skewness'] = se_y_list.apply(lambda x: stats.skew(x))
        X_train['z_skewness'] = se_z_list.apply(lambda x: stats.skew(x))

        # kurtosis
        X_train['x_kurtosis'] = se_x_list.apply(lambda x: stats.kurtosis(x))
        X_train['y_kurtosis'] = se_y_list.apply(lambda x: stats.kurtosis(x))
        X_train['z_kurtosis'] = se_z_list.apply(lambda x: stats.kurtosis(x))

        # energy
        X_train['x_energy'] = se_x_list.apply(lambda x: np.sum(x**2)/self.window_size)
        X_train['y_energy'] = se_y_list.apply(lambda x: np.sum(x**2)/self.window_size)
        X_train['z_energy'] = se_z_list.apply(lambda x: np.sum(x**2/self.window_size))

        # avg resultant
        X_train['avg_result_accl'] = [i.mean() for i in ((se_x_list**2 + se_y_list**2 + se_z_list**2)**0.5)]

        # signal magnitude area
        X_train['sma'] = se_x_list.apply(lambda x: np.sum(abs(x)/self.window_size)) + se_y_list.apply(lambda x: np.sum(abs(x)/self.window_size)) + se_z_list.apply(lambda x: np.sum(abs(x)/self.window_size))

        self.X_train = X_train.copy()

    def gen_fourier_features(self):

        X_train = self.X_train.copy()

        se_x_list = self.se_x_list
        se_y_list = self.se_y_list
        se_z_list = self.se_z_list

        # converting the signals from time domain to frequency domain using FFT
        end_part_signal = int(self.window_size/2 + 1)
        se_x_list_fft = se_x_list.apply(lambda x: np.abs(np.fft.fft(x))[1:end_part_signal])
        se_y_list_fft = se_y_list.apply(lambda x: np.abs(np.fft.fft(x))[1:end_part_signal])
        se_z_list_fft = se_z_list.apply(lambda x: np.abs(np.fft.fft(x))[1:end_part_signal])

        # Statistical Features on raw x, y and z in frequency domain
        # FFT mean
        X_train['x_mean_fft'] = se_x_list_fft.apply(lambda x: x.mean())
        X_train['y_mean_fft'] = se_y_list_fft.apply(lambda x: x.mean())
        X_train['z_mean_fft'] = se_z_list_fft.apply(lambda x: x.mean())

        # FFT std dev
        X_train['x_std_fft'] = se_x_list_fft.apply(lambda x: x.std())
        X_train['y_std_fft'] = se_y_list_fft.apply(lambda x: x.std())
        X_train['z_std_fft'] = se_z_list_fft.apply(lambda x: x.std())

        # FFT avg absolute diff
        X_train['x_aad_fft'] = se_x_list_fft.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_train['y_aad_fft'] = se_y_list_fft.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_train['z_aad_fft'] = se_z_list_fft.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

        # FFT min
        X_train['x_min_fft'] = se_x_list_fft.apply(lambda x: x.min())
        X_train['y_min_fft'] = se_y_list_fft.apply(lambda x: x.min())
        X_train['z_min_fft'] = se_z_list_fft.apply(lambda x: x.min())

        # FFT max
        X_train['x_max_fft'] = se_x_list_fft.apply(lambda x: x.max())
        X_train['y_max_fft'] = se_y_list_fft.apply(lambda x: x.max())
        X_train['z_max_fft'] = se_z_list_fft.apply(lambda x: x.max())

        # FFT max-min diff
        X_train['x_maxmin_diff_fft'] = X_train['x_max_fft'] - X_train['x_min_fft']
        X_train['y_maxmin_diff_fft'] = X_train['y_max_fft'] - X_train['y_min_fft']
        X_train['z_maxmin_diff_fft'] = X_train['z_max_fft'] - X_train['z_min_fft']

        # FFT median
        X_train['x_median_fft'] = se_x_list_fft.apply(lambda x: np.median(x))
        X_train['y_median_fft'] = se_y_list_fft.apply(lambda x: np.median(x))
        X_train['z_median_fft'] = se_z_list_fft.apply(lambda x: np.median(x))

        # FFT median abs dev
        X_train['x_mad_fft'] = se_x_list_fft.apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_train['y_mad_fft'] = se_y_list_fft.apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_train['z_mad_fft'] = se_z_list_fft.apply(lambda x: np.median(np.absolute(x - np.median(x))))

        # FFT Interquartile range
        X_train['x_IQR_fft'] = se_x_list_fft.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_train['y_IQR_fft'] = se_y_list_fft.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_train['z_IQR_fft'] = se_z_list_fft.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

        # FFT values above mean
        X_train['x_above_mean_fft'] = se_x_list_fft.apply(lambda x: np.sum(x > x.mean()))
        X_train['y_above_mean_fft'] = se_y_list_fft.apply(lambda x: np.sum(x > x.mean()))
        X_train['z_above_mean_fft'] = se_z_list_fft.apply(lambda x: np.sum(x > x.mean()))

        # FFT number of peaks
        X_train['x_peak_count_fft'] = se_x_list_fft.apply(lambda x: len(find_peaks(x)[0]))
        X_train['y_peak_count_fft'] = se_y_list_fft.apply(lambda x: len(find_peaks(x)[0]))
        X_train['z_peak_count_fft'] = se_z_list_fft.apply(lambda x: len(find_peaks(x)[0]))

        # FFT skewness
        X_train['x_skewness_fft'] = se_x_list_fft.apply(lambda x: stats.skew(x))
        X_train['y_skewness_fft'] = se_y_list_fft.apply(lambda x: stats.skew(x))
        X_train['z_skewness_fft'] = se_z_list_fft.apply(lambda x: stats.skew(x))

        # FFT kurtosis
        X_train['x_kurtosis_fft'] = se_x_list_fft.apply(lambda x: stats.kurtosis(x))
        X_train['y_kurtosis_fft'] = se_y_list_fft.apply(lambda x: stats.kurtosis(x))
        X_train['z_kurtosis_fft'] = se_z_list_fft.apply(lambda x: stats.kurtosis(x))

        # FFT energy
        X_train['x_energy_fft'] = se_x_list_fft.apply(lambda x: np.sum(x ** 2) / self.step_size)
        X_train['y_energy_fft'] = se_y_list_fft.apply(lambda x: np.sum(x ** 2) / self.step_size)
        X_train['z_energy_fft'] = se_z_list_fft.apply(lambda x: np.sum(x ** 2 / self.step_size))

        # FFT avg resultant
        X_train['avg_result_accl_fft'] = [i.mean() for i in (
                    (se_x_list_fft ** 2 + se_y_list_fft ** 2 + se_z_list_fft ** 2) ** 0.5)]

        # FFT Signal magnitude area
        X_train['sma_fft'] = se_x_list_fft.apply(lambda x: np.sum(abs(x) / self.step_size)) + se_y_list_fft.apply(lambda x: np.sum(abs(x) / self.step_size)) + se_z_list_fft.apply(lambda x: np.sum(abs(x) / self.step_size))

        self.X_train = X_train.copy()

    def extract_features(self):

        print('Extracting basic acceleration features')
        print('...')
        self.gen_acceleration_features()
        print('Extracting fourier features')
        print('...')
        self.gen_fourier_features()

        return self.X_train

"""
Generalize the application of AccelerometerExtractorSingleAcc with a database with different accelerometers

input:
    see parent class
    col_id_acc: name of column indicating the accelerometer
        if none the class is the same as parent class

"""
class AccelerometerExtractor(AccelerometerExtractorSingleAcc):

    set_id_acc = set()
    dct_ext = dict()
    col_id_acc = str()

    def __init__(self, df: pd.DataFrame(), window_size: int, step_size: int = 1,
                 col_x_axis: str = 'x-axis', col_y_axis: str = 'y-axis', col_z_axis: str = 'z-axis',
                 col_id_time: str = 'timestamp', col_id_acc: str = None):

        self.col_id_acc = col_id_acc

        self.df = df

        self.col_x_axis = col_x_axis
        self.col_y_axis = col_y_axis
        self.col_z_axis = col_z_axis
        self.col_id_time = col_id_time

        self.window_size = window_size
        self.step_size = step_size

        if col_id_acc is None:

            print('Only one accelerometer')

            super().__init__(df=df, window_size=window_size, step_size=step_size,
                             col_x_axis=col_x_axis, col_y_axis=col_y_axis, col_z_axis=col_z_axis,
                             col_id_time=col_id_time)

        else:

            self.set_acc_id = set(df[col_id_acc])

            print(f'Total of {len(self.set_acc_id)} acc')
            print()
            print(f'Creating extractors for each acc ...')
            print('...')

            for temp_id in self.set_acc_id:

                # print(f'Acc: {temp_id}')

                temp_df = df.loc[df[self.col_id_acc] == temp_id, :].copy()

                self.dct_ext[temp_id] = AccelerometerExtractorSingleAcc(df=temp_df, window_size=window_size,
                                                                        step_size=step_size, col_x_axis=col_x_axis,
                                                                        col_y_axis=col_y_axis, col_z_axis=col_z_axis,
                                                                        col_id_time=col_id_time)

    def __gen_acceleration_features(self):

        lst_df = list()

        for key, ext in self.dct_ext.items():

            # Call gen_acceleration_features from parent class
            ext.gen_acceleration_features()

            # assign a column with acc id in df
            ext.X_train[self.col_id_acc] = key

            lst_df.append(ext.X_train)

        self.X_train = pd.concat(lst_df, axis=0)

    def gen_acceleration_features(self):

        if self.col_id_acc is None:

            super().gen_acceleration_features()

        else:

            self.__gen_acceleration_features()

    def __gen_fourier_features(self):

        lst_df = list()

        for key, ext in self.dct_ext.items():
            # Call gen_acceleration_features from parent class
            ext.gen_fourier_features()

            lst_df.append(ext.X_train)

        self.X_train = pd.concat(lst_df, axis=0)

    def gen_fourier_features(self):

        if self.col_id_acc is None:

            super().gen_fourier_features()

        else:

            self.__gen_fourier_features()

    def get_X_train(self):

        return self.X_train
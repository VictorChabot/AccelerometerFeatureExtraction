import pandas as pd
import os
from accelerometer_extractor import AccelerometerExtractor, AccelerometerExtractorSingleAcc
path_df = os.path.join('data', 'df.pkl')

df = pd.read_pickle(path_df)

# ext_single = AccelerometerExtractor(df=df, window_size=100, step_size=50, col_id_acc=None)

ext_multiple = AccelerometerExtractor(df=df, window_size=100, step_size=50,
                                      col_x_axis='x-axis', col_y_axis='y-axis', col_z_axis='z-axis',
                                      col_id_time='timestamp', col_id_acc='user')

df_feat = ext_multiple.extract_features()









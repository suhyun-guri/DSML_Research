import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout, InputLayer, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from sklearn import metrics 

import warnings
warnings.filterwarnings(action='ignore')

#한글설정
import matplotlib.font_manager as fm

font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)
    
# 한글 출력을 위해서 폰트 옵션을 설정합니다.
# "axes.unicode_minus" : 마이너스가 깨질 것을 방지

sns.set(font="NanumBarunGothic",
        rc={"axes.unicode_minus":False},
        style='darkgrid')

#GPU 사용 설정, -1이면 CPU 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:  # gpu가 있다면, 용량 한도를 5GB로 설정
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                                                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)])


import random    
seed_num = 42
random.seed(seed_num)

X = np.load('/project/LSH/x_(7727,10,4068).npy')
y = np.load('/project/LSH/y_(7727,1).npy')

model = tf.keras.models.load_model('./models/ALLFIT_01-0.4865.hdf5')
COLS = list(pd.read_csv('/project/LSH/total_data_7727.csv')['ITEMID'].sort_values().unique())

with tf.device('/device:GPU:0'): 
    result = []
    for i in tqdm(range(X.shape[2])):
        save_cols = X[:,:,i].copy()
        #-----zero2one-----
        X[:,:,i] = 1
        pred1 = model.predict(X)
        mean_pred1 = np.mean(pred1)
        #-----one2zero-----
        X[:,:,i] = 0
        pred2 = model.predict(X)
        mean_pred2 = np.mean(pred2)

        result.append({'feature' : str(COLS[i]), 'one2zero' : mean_pred2,'zero2one' : mean_pred1,
                       'lambda0' : mean_pred2 - mean_pred1, 'lambda1' : (mean_pred2 - mean_pred1) * entropy_dict[COLS[i]]})

        #값 복원
        X[:,:,i] = save_cols
        
df = pd.DataFrame(result)
df.to_csv('./data/Method_allfit.csv', index=False)


print('----------------end----------------')

# top10 = []
# for n in range(10):
#     if len(top10) > 1:
#         print(top10)
#         result = []
#         for i in tqdm(range(X.shape[2])):
#             save_cols = X[:,:,top10].copy()
#             #-----zero2one-----
#             X[:,:,top10] = 1
#             pred1 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
#             mean_pred1 = np.mean(pred1)
#             #-----one2zero-----
#             X[:,:,top10] = 0
#             pred2 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
#             mean_pred2 = np.mean(pred2)
#             result.append({'feature_index' : i, 'one2zero' : mean_pred2,'zero2one' : mean_pred1,
#                            'lambda0' : mean_pred2 - mean_pred1, 'lambda1' : (mean_pred2 - mean_pred1) * entropy_dict[COLS[i]]})
#         df = pd.DataFrame(result).sort_values('lambda0', ascending=False)
#         top10.append(df.feature_index[n])
#     else:
#         result = []
#         for i in tqdm(range(X.shape[2])):
#             save_cols = X[:,:,i].copy()
#             #-----zero2one-----
#             X[:,:,i] = 1
#             pred1 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
#             mean_pred1 = np.mean(pred1)
#             #-----one2zero-----
#             X[:,:,i] = 0
#             pred2 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
#             mean_pred2 = np.mean(pred2)

#             result.append({'feature_index' : i, 'one2zero' : mean_pred2,'zero2one' : mean_pred1,
#                            'lambda0' : mean_pred2 - mean_pred1, 'lambda1' : (mean_pred2 - mean_pred1) * entropy_dict[COLS[i]]})

#         df = pd.DataFrame(result).sort_values('lambda0', ascending=False)
#         top10.append(df.feature_index[0])

# with open('./data/Sequential_top10.txt', 'w') as file:
#     file.write(top10)

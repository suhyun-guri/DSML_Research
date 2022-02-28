import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import Sequentiasl
# from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.layers import SimpleRNN
# from tensorflow.keras.layers import Dropout, InputLayer, Activation
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from sklearn import metrics 

import warnings
warnings.filterwarnings(action='ignore')

#GPU 사용 설정, -1이면 CPU 사용
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#https://www.tensorflow.org/guide/gpu#allowing_gpu_memory_growth
#프로세스의 요구량만큼 메모리 사용 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

import random    
seed_num = 42
random.seed(seed_num)
#-----데이터 로드-----
X = np.load('/project/LSH/x_(7727,10,4068).npy')
y = np.load('/project/LSH/y_(7727,1).npy')
#-----Item name 로드-----

COLS = list(pd.read_csv('/project/LSH/total_data_7727.csv')['ITEMID'].sort_values().unique())
#-----각 feature의 Entropy를 기록한 dictionary 생성-----
def entropy(ratio_list):
    one_ratio, zero_ratio = ratio_list[0], ratio_list[1] 
    return - ((one_ratio * (np.log2(one_ratio))) + (zero_ratio * (np.log2(zero_ratio))))

X = np.load('/project/LSH/x_(7727,10,4068).npy')

entropy_dict = {}
for i in tqdm(range(len(COLS))):
    one_ratio = X[:,:,i].sum() / (X.shape[0]*X.shape[1])
    zero_ratio = 1 - one_ratio
    entropy_num = entropy([one_ratio, zero_ratio])
    entropy_dict[COLS[i]] = entropy_num


#-----FI 측정 시작-----
# with tf.device('/device:GPU:1'): 
#     #-----모델 로드-----
#     model = tf.keras.models.load_model('./models/ALLFIT_17-0.7645.hdf5')
#     #-----RUN-----
#     result = []
#     for i in tqdm(range(X.shape[2])):
#         save_cols = X[:,:,i].copy()
#         #-----zero2one-----
#         X[:,:,i] = 1
#         pred1 = model.predict(X)
#         mean_pred1 = np.mean(pred1)
#         #-----one2zero-----
#         X[:,:,i] = 0
#         pred2 = model.predict(X)
#         mean_pred2 = np.mean(pred2)

#         result.append({'feature' : str(COLS[i]), 'one2zero' : mean_pred2,'zero2one' : mean_pred1,
#                        'lambda0' : mean_pred2 - mean_pred1, 'lambda1' : (mean_pred2 - mean_pred1) * entropy_dict[COLS[i]]})

#         #값 복원
#         X[:,:,i] = save_cols
        
# df = pd.DataFrame(result)
# df.to_csv('./data/Method_allfit.csv', index=False)


print('----------------Sequential----------------')

#-----FI 측정 시작-----
with tf.device('/device:GPU:1'): 
    #-----모델 로드-----
    model = tf.keras.models.load_model('./models/ALLFIT_17-0.7645.hdf5')
    top10 = []
    top10_result = []
    for n in range(2):
        result = []
        if len(top10) > 1:
            print('####################  top10 :', top10)
            for i in tqdm(range(X.shape[2])):
                save_cols = X[:,:,top10].copy()
                #-----zero2one-----
                X[:,:,top10] = 1
                pred1 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
                mean_pred1 = np.mean(pred1)
                #-----one2zero-----
                X[:,:,top10] = 0
                pred2 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
                mean_pred2 = np.mean(pred2)
                result.append({'feature_index' : i, 'one2zero' : mean_pred2,'zero2one' : mean_pred1,
                            'lambda0' : mean_pred2 - mean_pred1, 'lambda1' : (mean_pred2 - mean_pred1) * entropy_dict[COLS[i]]})
            df = pd.DataFrame(result).sort_values('lambda0', ascending=False)
            print(df)
            top10.append(df.feature_index[n])
        else:
            for i in tqdm(range(X.shape[2])):
                save_cols = X[:,:,i].copy()
                #-----zero2one-----
                X[:,:,i] = 1
                pred1 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
                mean_pred1 = np.mean(pred1)
                #-----one2zero-----
                X[:,:,i] = 0
                pred2 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
                mean_pred2 = np.mean(pred2)

                result.append({'feature_index' : i, 'one2zero' : mean_pred2,'zero2one' : mean_pred1,
                            'lambda0' : mean_pred2 - mean_pred1, 'lambda1' : (mean_pred2 - mean_pred1) * entropy_dict[COLS[i]]})

            df = pd.DataFrame(result).sort_values('lambda0', ascending=False)
            top10.append(df.feature_index[0])


    #-----df저장-----
    df.to_csv('./data/Method_Sequential_FI.csv', index=False)
    #-----top10 feature만 저장-----
    top10_list = list(map(lambda x:str(COLS[x]), top10))
    print(top10_list)
    with open('./data/Sequential_top10.txt', 'w') as file:
        file.write(top10_list)

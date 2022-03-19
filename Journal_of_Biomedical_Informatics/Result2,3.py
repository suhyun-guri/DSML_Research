import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score

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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

#---------데이터 로드---------
path = '/project/LSH/'
# model_path = path + 'model/allfit_ep500/allfit_ep500_seed42-17-0.7619.hdf5'

#guri모델
model_path = '/project/guri/Restart/models/ALLFIT_17-0.7645.hdf5'

COLS = list(pd.read_csv(path + 'total_data_7727.csv')['ITEMID'].sort_values().unique())
x = np.load(path + 'x_(7727,10,4068).npy')
y = np.load(path + 'y_(7727,1).npy')

#---------Entropy---------
def entropy(ratio_list):
    one_ratio, zero_ratio = ratio_list[0], ratio_list[1] 
    if (one_ratio == 0) or (zero_ratio == 0):
        return 0.0
    return - ((one_ratio * (np.log2(one_ratio))) + (zero_ratio * (np.log2(zero_ratio))))

X = np.load('/project/LSH/x_(7727,10,4068).npy')

entropy_dict = {}
for i in tqdm(range(len(COLS))):
    one_ratio = X[:,:,i].sum() / (X.shape[0]*X.shape[1])
    zero_ratio = 1 - one_ratio
    entropy_num = entropy([one_ratio, zero_ratio])
    entropy_dict[COLS[i]] = entropy_num

#---------Method1 측정---------
with tf.device('/device:GPU:0'): 
    model = tf.keras.models.load_model(model_path)
    result = []
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

        result.append({'feature' : str(COLS[i]), 'zero2one' : mean_pred1, 'one2zero' : mean_pred2,
                       'lambda0' : mean_pred1 - mean_pred2, 'lambda1' : (mean_pred1 - mean_pred2) * entropy_dict[COLS[i]]})
        #값 복원
        X[:,:,i] = save_cols

#guri 모델 사용
df = pd.DataFrame(result)
df.to_csv('./Result2,3_guri.csv', index=False)
df.sort_values('lambda1', ascending=False)
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

path = '/project/LSH/'
model_path = '/project/guri/ForPaper/models/seed42-06-val_loss:0.5528.hdf5'

COLS = list(pd.read_csv(path + 'total_data_7727.csv')['ITEMID'].sort_values().unique())
x = np.load(path + 'x_(7727,10,4068).npy')
y = np.load(path + 'y_(7727,1).npy')

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

with tf.device('/device:GPU:1'): 
    model = tf.keras.models.load_model(model_path)
    result = []
    for i in tqdm(range(X.shape[2])):
        #-----#원거리#-----
        save_cols = X[:,:,i].copy()
        #-----zero2one-----
        X[:,:5,i] = 1
        원_pred1 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
        원_mean_pred1 = np.mean(원_pred1)
        #-----one2zero-----
        X[:,:5,i] = 0
        원_pred2 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
        원_mean_pred2 = np.mean(원_pred2)
        #-----값 복원-----
        X[:,:,i] = save_cols
        
        #-----#근거리#-----
        #-----zero2one-----
        X[:,5:,i] = 1
        근_pred1 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
        근_mean_pred1 = np.mean(근_pred1)
        #-----one2zero-----
        X[:,5:,i] = 0
        근_pred2 = model.predict(X, batch_size=10000, workers=-1, use_multiprocessing=True)
        근_mean_pred2 = np.mean(근_pred2)
        #-----값 복원-----
        X[:,:,i] = save_cols
        
        result.append({'feature' : str(COLS[i]),'원_lambda1' : (원_mean_pred1 - 원_mean_pred2) * entropy_dict[COLS[i]],
                       '근_lambda1' : (근_mean_pred1 - 근_mean_pred2)* entropy_dict[COLS[i]]})        


df = pd.DataFrame(result).sort_values('feature')
df.to_csv('./Result4_loss.csv', index=False)
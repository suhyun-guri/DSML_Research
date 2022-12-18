# [tf shap version issue](https://www.pythonfixing.com/2021/12/fixed-shap-deepexplainer-with.html)
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import tensorflow as tf
tf.random.set_seed(42)

from typing import List, Set, Dict, Tuple, Optional, Any
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
import shap
shap.initjs()

# 결과 확인을 용이하게 하기 위한 코드
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

print('tf:', tf.__version__)
print('shap:', shap.__version__)

import pandas as pd

# from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.metrics import mean_squared_error as mse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

from tqdm.notebook import tqdm

# import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, InputLayer, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# 결과 확인을 용이하게 하기 위한 코드
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# 한글 출력을 위해서 폰트 옵션을 설정합니다.
# "axes.unicode_minus" : 마이너스가 깨질 것을 방지
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')

#GPU 사용 설정, -1이면 CPU 사용
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야 함
        print(e)
        
# LSTM
def LSTM_model():
    lstm = Sequential()
    lstm.add(InputLayer(input_shape=(x.shape[1],x.shape[2])))
    lstm.add(LSTM(units=128, activation='hard_sigmoid', return_sequences=True))
    lstm.add(LSTM(units=64, activation='hard_sigmoid', return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units=64, activation='hard_sigmoid', return_sequences=True))
    lstm.add(LSTM(units=32, activation='hard_sigmoid', return_sequences=False))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(units=1, activation='sigmoid'))
    lstm.compile(optimizer= keras.optimizers.Adam(learning_rate = 0.001), 
                 loss = "binary_crossentropy", metrics=['acc'])
    return lstm

def aggregate_shap_values(data, shap_values):
    shap_array = np.array(shap_values[0]) # array of shap values computed for time series

    mean_shaps = []

    for i in range(shap_array.shape[0]):
        max_seq_length = shap_array.shape[1]
        for k in range(shap_array.shape[1]):
            if i != 0 and k != 0:  # buggy index I skip over
                # boolean check for zeroed out padded rows
                all_zero = not np.all(data[i, k, ] == 0)
                if all_zero:  # if row k is all zero
                    max_seq_length = k
        mean_shaps.append(shap_array[i, :max_seq_length, :].mean(axis=0).tolist()) 
    shap_values = np.array(mean_shaps)
    return shap_values


COLS = list(pd.read_csv('/project/LSH/total_data.csv')['ITEMID'].sort_values().unique())
print(len(COLS))

path = '/project/LSH/** 해외_Journal of Biomedical Informatics/'
with tf.device('/device:GPU:0'):

    # 1. Load Dataset
    x = np.load(path + 'x_(7727,10,3595).npy')
    y = np.load(path + 'y_(7727,1).npy')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    acc_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []
    result = []
    
    # 2. Load LSTM model
    model = LSTM_model()
    
    # 3. RUN LSTM model and Prediction
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=False)
    model.fit(X_train, y_train, epochs=300, batch_size=516, validation_split=0.25, callbacks=[early_stop])

    y_pred_test = model.predict(X_test)
    y_pred_test[y_pred_test>0.5]=1
    y_pred_test[y_pred_test<=0.5]=0

    acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_pred_test)

    result.append(['LSTM', acc, precision, recall, f1, roc_auc])
    
    ## ---------- APPLY SHAP ----------    
    start_at = datetime.now()
    # 샘플 추출
    background = X_train[np.random.choice(X_train.shape[0], 3000, replace=False)]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(background)
    
    agg_shap_values = aggregate_shap_values(background, shap_values)
    # agg_shap_values = aggregate_shap_values(X_train, shap_values)
    print('agg_shape_values : ', agg_shap_values.shape)
    agg_data = np.array(background.mean(axis=1))
    # agg_data = np.array(X_train.,mean(axis=1))
    print('agg_data : ', agg_data.shape)
    
    fig = plt.figure()
    shap.summary_plot(
                    agg_shap_values, agg_data, 
                    feature_names=COLS, show=False
                )
    plt.savefig("shap_summary_ori.png",dpi=700)
    
    fig = plt.figure()
    shap.summary_plot(
                    agg_shap_values, agg_data, 
                    feature_names=COLS, plot_type="bar", show=False
                )
    plt.savefig("shap_summary_bar.png",dpi=700)
    
    end_at = datetime.now()
    print('소요 시간 : ', (end_at - start_at))
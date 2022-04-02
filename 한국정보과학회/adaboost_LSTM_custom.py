import os
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings 
warnings.filterwarnings(action='ignore')


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, LSTM, InputLayer
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
from tensorflow import keras
import random  
from tensorflow.keras.callbacks import EarlyStopping

seed_num = 42
tf.random.set_seed(seed_num)
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     try:
#         for i in range(len(gpus)):
#             tf.config.experimental.set_memory_growth(gpus[i], True)
#     except RuntimeError as e:
#         # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
#         print(e)
# gpus = tf.config.experimental.list_physical_devices('GPU')

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:  # gpu가 있다면, 용량 한도를 5GB로 설정
#     tf.config.experimental.set_virtual_device_configuration(gpus[1], 
#                                                             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)])
seed_num = 42
random.seed(seed_num)

x = np.load('/project/LSH/x_(7727,10,4068).npy')
y = np.load('/project/LSH/y_(7727,1).npy')

idx = list(range(len(x)))
random.shuffle(idx)

i = round(x.shape[0]*0.8)
X_train, y_train = x[idx[:i],:,:], y[idx[:i]]
X_test, y_test = x[idx[i:],:,:], y[idx[i:]]

X_train.shape, y_train.shape, X_test.shape, y_test.shape


def get_model():
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

from keras.wrappers.scikit_learn import KerasClassifier
import random
class MyKerasClassifier(KerasClassifier):
    def fit(self, x, y, sample_weight=None, validation_split=0.25, **kwargs):
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        #---------------수정---------------
        if sample_weight is not None:
            print('sample weight : ', sample_weight)
            if sample_weight[0] == 0.00016175994823681658:
                return super(MyKerasClassifier, self).fit(x, y, **kwargs)
            
            idx = np.arange(len(x))
            random.shuffle(idx)
            i = int(len(x)*0.75)
            train_x, train_y = x[idx[:i],:,:], y[idx[:i]]
            val_x, val_y = x[idx[i:],:,:], y[idx[i:]]
            
            
            train_sw, val_sw = sample_weight[idx[:i]], sample_weight[idx[i:]]
            train_sw, val_sw = train_sw/sum(train_sw)*len(train_sw), val_sw/sum(val_sw)*len(val_sw)
            
            weights = val_sw / sum(val_sw)
            random_range = [(sum(weights[:i]), sum(weights[:i])+weights[i]) if i!=0 else (0, weights[i]) for i in range(len(weights))]
            random_nums = [random.random() for _ in range(len(weights))]
            idx_list = []
            for i in random_nums:
                for j in random_range:
                    if j[0] < i <= j[1]:
                        idx_list.append(random_range.index(j))
                        break
            new_val_x = val_x[idx_list, :, :]
            new_val_y = val_y[idx_list]
            
            kwargs['validation_data'] = (new_val_x, new_val_y)
            kwargs['sample_weight'] = train_sw
            return super(MyKerasClassifier, self).fit(train_x, train_y, **kwargs)
        
        return super(MyKerasClassifier, self).fit(x, y, **kwargs)
    
    
    
    def predict(self, x, **kwargs):
        return super(MyKerasClassifier, self).predict(x)

# with tf.device('/device:GPU:1'):

batch = 516
estimators_nums = 200
lr = 0.9

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
# base_estimator = MyKerasClassifier(build_fn=get_model, epochs=300, batch_size=batch,
#                                     validation_split=0.25, callbacks=[early_stop])
# boosted_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=estimators_nums, random_state=42, learning_rate=lr)

# print("Adaboost LSTM Start")
# boosted_classifier.fit(X_train, y_train)
# preds = boosted_classifier.predict(X_test)

# precision = precision_score(y_test, preds)
# recall = recall_score(y_test, preds)
# f1 = f1_score(y_test, preds)
# roc_auc = roc_auc_score(y_test, preds)
# acc = accuracy_score(y_test, preds)

# print(f'Adaboost accuracy : {acc}, precision : {precision}, recall : {recall}, f1 : {f1}, roc_auc : {roc_auc}')
# acc_list = [accuracy_score(y_test,i) for i in boosted_classifier.staged_predict(X_test)]
# print(acc_list)
# import time
# now = time.localtime()
# nowtime = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
# with open('./Result_of_Adaboost.txt', 'a') as f:
#     f.write(f'{nowtime} [batch:{batch}, n:{estimators_nums}, lr:{lr}]- accuracy : {acc}, precision : {precision}, recall : {recall}, f1 : {f1}, roc_auc : {roc_auc} acc_list : {acc_list} \n')


model = get_model()
model.fit(X_train, y_train, epochs=300, batch_size=batch, validation_split=0.25, callbacks=[early_stop])
preds = model.predict(X_test)

preds[preds>0.5]=1
preds[preds<=0.5]=0
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, preds)
acc = accuracy_score(y_test, preds)

print(f'accuracy : {acc}, precision : {precision}, recall : {recall}, f1 : {f1}, roc_auc : {roc_auc}')
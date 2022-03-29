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

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

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

class MyKerasClassifier(KerasClassifier):
    def fit(self, x, y, sample_weight=None, **kwargs):
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
#                 kwargs['sample_weight'] = sample_weight
                print('x, y', x.shape, x.sum().sum())
                return super().fit(x, y)
            weights = sample_weight / sum(sample_weight)
            random_range = [(sum(weights[:i]), sum(weights[:i])+weights[i]) if i!=0 else (0, weights[i]) for i in range(len(weights))]
            random_nums = [random.random() for _ in range(len(weights))]
            idx_list = []
            for i in random_nums:
                for j in random_range:
                    if j[0] < i <= j[1]:
                        idx_list.append(random_range.index(j))
                        break
            new_x = x[idx_list, :, :]
            new_y = y[idx_list]
#             kwargs['sample_weight'] = sample_weight
            print(new_x.sum().sum())
            print('new_x, new_y', new_x.shape, new_y.shape)
            return super().fit(new_x, new_y)
            # return super().fit(x, y, sample_weight=sample_weight)
        
    def predict(self, x, **kwargs):
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        classes = self.model.predict_classes(x, **kwargs)
        return self.classes_[classes].flatten()

from tensorflow.keras.callbacks import ModelCheckpoint
with tf.device('/device:GPU:0'):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    cb_checkpoint = ModelCheckpoint(filepath='./models/adaboost_lstm1.h5', monitor='val_acc',
                                    verbose=1, save_best_only=True)
    base_estimator = MyKerasClassifier(build_fn=get_model, epochs=50, batch_size=128, validation_split=0.25, callbacks=[early_stop, cb_checkpoint])
    
    boosted_classifier = AdaBoostClassifier(base_estimator=base_estimator,n_estimators=3, random_state=42, learning_rate=0.8)
    # print("Single LSTM Start")
    # lstm.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2)
    # single_preds = lstm.predict(X_test)
    # single_preds[single_preds>0.5]=1
    # single_preds[single_preds<=0.5]=0
    print("Adaboost LSTM Start")
    boosted_classifier.fit(X_train, y_train)
    preds = boosted_classifier.predict(X_test)

    # single_precision = precision_score(y_test, single_preds)
    # single_recall = recall_score(y_test, single_preds)
    # single_f1 = f1_score(y_test, single_preds)
    # single_roc_auc = roc_auc_score(y_test, single_preds)
    # single_acc = accuracy_score(y_test, single_preds)
    # print(boosted_classifier.estimators_)
    # print(boosted_classifier.estimators_[0].model_.get_weights()[0][0, :5])  # first sub-estimator
    # print(boosted_classifier.estimators_[1].model_.get_weights()[0][0, :5])  # second sub-estimator


    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, preds)

    # print(f'single accuracy : {single_acc}, precision : {single_precision}, recall : {single_recall}, f1 : {single_f1}, roc_auc : {single_roc_auc}')
    print(f'Adaboost accuracy : {acc}, precision : {precision}, recall : {recall}, f1 : {f1}, roc_auc : {roc_auc}')
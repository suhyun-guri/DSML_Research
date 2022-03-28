import numpy as np
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, LSTM, InputLayer
import random  
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping

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

early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=False)
lstm = get_model()
clf = KerasClassifier(
    model=lambda:lstm, epochs=50, batch_size=256, validation_split=0.2, loss='binary_crossentropy',
    optimizer='adam', optimizer__lr=0.001
)

with tf.device('/device:GPU:0'):
    adaboost = AdaBoostClassifier(base_estimator=clf, random_state=0, n_estimators=5, learning_rate=1.2)
    print("Single LSTM Start")
    single_score = clf.fit(X_train, y_train).score(X_train, y_train)
    single_preds = clf.predict(X_test)
    print("Adaboost LSTM Start")
    adaboost_score = adaboost.fit(X_train, y_train).score(X_train, y_train)
    preds = adaboost.predict(X_test)

print(f"Single score: {single_score}")
print(f"AdaBoost score: {adaboost_score}")



single_precision = precision_score(y_test, single_preds)
single_recall = recall_score(y_test, single_preds)
single_f1 = f1_score(y_test, single_preds)
single_roc_auc = roc_auc_score(y_test, single_preds)
single_acc = accuracy_score(y_test, single_preds)

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, preds)
acc = accuracy_score(y_test, preds)

print(f'single accuracy : {single_acc}, precision : {single_precision}, recall : {single_recall}, f1 : {single_f1}, roc_auc : {single_roc_auc}')
print(f'Adaboost accuracy : {acc}, precision : {precision}, recall : {recall}, f1 : {f1}, roc_auc : {roc_auc}')
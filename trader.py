#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import os

iTrainDay = 5
modulePath = 'my_model'

print('------------')
print('Project Name : 2022_MLAI_HW1-Auto_Trading')
print('Student Name : WuKu Yang')
print('Student ID : P77111126')
print('------------')
print('Model Save Path : ' , modulePath)
print('LSTM Train Day : ' , iTrainDay)
print('------------')

def norm(v):
      tva = [ (x - np.min(x)) / (np.max(x) - np.min(x)) for x in v]
      return np.array(tva)

def norm_max_min(v):
      v_max = np.max(v)
      v_min = np.min(v)
      tva = [ (x - np.min(x)) / (np.max(x) - np.min(x)) for x in v]
      return np.array(tva) , v_max , v_min

def de_norm(tva , v_max , v_min):
      v_x = [ y*(v_max-v_min) + v_min for y in tva]
      return np.array(v_x)

def normalize(df):
    norm = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return norm

def train_windows(df, ref_day=3, predict_day=1):
    X_train, Y_train = [], []
    for i in range(df.shape[0]-predict_day-ref_day):
        # print(df.iloc[i])
        X_train.append(np.array(df.iloc[i:i+ref_day,:-1]))
        Y_train.append(np.array(df.iloc[i+ref_day:i+ref_day+predict_day]))
    return np.array(X_train), np.array(Y_train)

def lstm_stock_model(shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5,activation='linear'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
    model.summary()
    return model

def Train(sTrainCSV = './training_data.csv' , sSaveModulePath = 'D:\\LSTM\\my_model'):
        train = pd.read_csv(sTrainCSV , header = None)
        train.head()
        train = train.dropna()
        train.iloc[:,0:4] = normalize(train.iloc[:,0:4])
        X_train, Y_train = train_windows(train, iTrainDay, 1)
        model = lstm_stock_model(X_train.shape)
        callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")
        history = model.fit(X_train, Y_train, epochs=1000, batch_size=5, validation_split=0.1, callbacks=[callback],shuffle=True)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        model.save(sSaveModulePath)
    
def PredictCSV(sInputCSV = './testing_data.csv' , sOutputCSV = 'output.csv'):
    #Testing Datas Preparee
    print("Reading Testing data from " , sInputCSV)
    pd_csv_test_file  = pd.read_csv(sInputCSV , header = None)          #使用panda讀取Testing資料 (標頭需忽略，才不會少一行)
    np_csv_test_file = pd_csv_test_file.to_numpy()                      #Panda to Numpy (備註 : Numpy在維度上比較好操作與轉型)
    print('TestData(Days) : ' , np_csv_test_file[:,0:3].shape[0])
    iMyStockCount = 0                                                   #庫存數量(Debug使用)
    trade_mode = 0                                                      #1買進、0持平、-1放空(區域變數，每次交易行為將存到此變數)
    trade_mode_record = np.empty([np_csv_test_file[:,0:3].shape[0]-1])  #交易行為錄製，錄製大小為 輸入資料數量-1 (因為最後一天不交易)
    

    for i in range(np_csv_test_file[:,0:3].shape[0]-1):
        if(i < iTrainDay):
            np_inputs = np.zeros((1,iTrainDay,3))                           #前五天不交易 (預設取值為0)
        else:
            np_inputs = np_csv_test_file[i-iTrainDay:i,0:3][:,np.newaxis]   #拿前五天的資料來預測
            
        testing_data , v_max , v_min = norm_max_min(np_inputs)              #取完數值要進行正規化
        predict_result = model.predict(testing_data)                        #使用訓練好的模型進行預測
        
        predict_price = de_norm(predict_result , v_max , v_min)[0][0]       #取出預測股價(反正規化)
        current_price = np_csv_test_file[i,3]                               #現在股價(收盤價)
        
        diff_value = predict_price - current_price                          #計算預計 5 天後的股價會跟現在差多少
        trade_keyword = 'No Trade'
        if(diff_value > 0):                                                 #只做多，只要有差異就進行交易(做多方)
            if(iMyStockCount <= 0):
                trade_mode = 1
                trade_keyword = 'Buy'
            else:                                                       
                trade_mode = 0
                trade_keyword = 'No Trade'
            #print('No.' , i , ' predict stock price =>' , predict_price)
        else:                                                               #如果預期五天後有可能會跌，那就出清持股 (本策略只多不空)
            
            if(iMyStockCount > 0):
                trade_mode = -1
                trade_keyword = 'sell'
            else:
                trade_mode = 0
                trade_keyword = 'No Trade'
                
        trade_mode_record[i] = trade_mode                                   #紀錄Action (交易行為)
        iMyStockCount = iMyStockCount + trade_mode                          #印出當下庫存(Debug使用)
        print(i , '-' , 'current Price:' ,current_price , ' Predict Price : ' ,  predict_price ,'  This trade rule is : ' , trade_keyword , ' Inventory : ' , iMyStockCount)

    print('Actions :')
    print(trade_mode_record)
    print('Save csv file...RorCount : ' , trade_mode_record.shape[0])

    #轉型為panda
    DF = pd.DataFrame(trade_mode_record)

    #使用panda存成CSV
    DF.to_csv(sOutputCSV , index = False ,  header=False)   #存檔時需符合助教給的格式，故濾除Index(最左邊一欄)與Header(最上面一列)

    print('Save csv file...Done!!...' , sOutputCSV)
    print('InputDays : ' , np_csv_test_file[:,0:3].shape[0] , 'OutPut Actions :' , trade_mode_record.shape[0])

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    print('Training !!...' , args.training)
    Train(args.training , modulePath)                           # = Step 01 = 訓練模型 (本次參數預設為 5 日)  , 訓練完畢後儲存到 modulePath 底下

    print('read module !!...' , modulePath)
    model = keras.models.load_model(modulePath)                 # = Step 02 = 從 modulePath 讀取該次權重檔案
    print('read module !!...' , modulePath , '....Done')

    #PredictCSV('./testing_data.csv' , 'output.csv')             # = Step 03 = 進行批量CSV檔預測股價，最終輸出到output.csv底下
    PredictCSV(args.testing , args.output)             # = Step 03 = 進行批量CSV檔預測股價，最終輸出到output.csv底下


import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import date, timedelta

from datetime import datetime as dt
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import tensorflow as tf
import os.path
from os import path

def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

def create_model(stock):
    df =yf.download(stock,period='2y')
    cls = df[['Close']]
    ds = cls.values

    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size

    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]
    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='relu'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=53,batch_size=15)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)
    test = np.vstack((train_predict,test_predict))
    ds_scaled=normalizer.inverse_transform(ds_scaled)
    model.save(stock+'.h5')

def lstmprediction(stock, n_days):
    df = yf.download(tickers=stock,period='2y',interval='1d')
    df['Date'] = df.index
    df['MA100']=df.Close.rolling(100).mean()
    df['MA200']=df.Close.rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df = df.assign(VWAP=df.eval('wgtd = Close * Volume', inplace=False).cumsum().eval('wgtd / Volume'))
    cls = df[['Close']]
    ds = cls.values

    if(path.exists(stock+'.h5')):
        model=tf.keras.models.load_model(stock+'.h5')
    else:
        create_model(stock)
        model=tf.keras.models.load_model(stock+'.h5')
    
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
    ds_scaled=normalizer.inverse_transform(ds_scaled)

    fut_inp = ds_scaled[len(ds_scaled)-100:]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    lst_output=[]
    n_steps=100
    i=0
    while(i<n_days):
        
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    ds_new = ds_scaled.tolist()
    final_graph = normalizer.inverse_transform(lst_output)
    dates = []
    current = date.today()

    for i in range(10):
        current += timedelta(days=1)
        dates.append(current)
   
    fig = px.line( final_graph,
                        title="Prediction for next {0} days".format(n_days),
                        labels=dict(variable="Prediction"))
    fig.update_yaxes(title_text='Prices in ₹')
    fig.update_xaxes(title_text='n Day')
    return fig

   
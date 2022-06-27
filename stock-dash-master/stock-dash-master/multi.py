import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
from os import path
import tensorflow as tf

# stock_symbol="SBIN.NS"

# start_date='2020-06-01'
# end_date='2022-06-23'
# df =yf.download(stock_symbol, str(start_date), str(end_date))
def predict(stock_symbol,n_future):
    df =yf.download(stock_symbol, period='2y')

    df.reset_index(inplace=True)

    train_dates = pd.to_datetime(df['Date'])
    cols = list(df)[1:7]
    df_for_training = df[cols].astype(float)
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    df_for_training_scaled

    trainX=[]
    trainY=[]
    n_fut=1
    n_past=14

    for i in range(n_past, len(df_for_training_scaled) - n_fut +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_fut - 1:i + n_fut, 3])
    trainX, trainY = np.array(trainX), np.array(trainY)
  
    if(path.exists(stock_symbol+'.h5')):
        model=tf.keras.models.load_model(stock_symbol+'.h5')
    else:
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))
        model.summary()

        model.compile(optimizer='adam', loss='mse')
        model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
        model.save(stock_symbol+'.h5')

    n_past =1
    # n_future=15  #let us predict past 15 days

    predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_future, freq='1d').tolist()
    prediction = model.predict(trainX[-n_future:])
    predict_period_dates

    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,3]

    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Close':y_pred_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])
    df_forecast.index=df_forecast['Date']

    fig = px.line( df_forecast,
                            x=df_forecast.index,
                            y=['Close'],
                            title="Prediction for next {0} days".format(n_future),
                            labels=dict(variable="Prediction"))
    fig.update_yaxes(title_text='Prices in ₹')
    fig.update_xaxes(title_text='Day')
    return(fig)